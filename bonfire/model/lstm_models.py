from abc import ABC

import torch
from torch import nn

from model.model_base import FullyConnectedStack
from model.nn_models import MultipleInstanceNN


class MiLstm(MultipleInstanceNN, ABC):

    def __init__(self, device, n_classes, n_expec_dims, encoder, aggregator):
        super().__init__(device, n_classes, n_expec_dims)
        self.encoder = encoder
        self.aggregator = aggregator

    def _internal_forward(self, bags):
        bag_predictions = torch.zeros((len(bags), self.n_classes)).to(self.device)
        all_cumulative_bag_predictions = []

        # First pass: get instance embeddings, bag embeddings, and bag predictions.
        for i, instances in enumerate(bags):
            instances = instances.to(self.device)
            instance_embeddings = self.encoder(instances)
            bag_prediction, cumulative_bag_predictions = self.aggregator(instance_embeddings)
            bag_predictions[i] = bag_prediction
            all_cumulative_bag_predictions.append(cumulative_bag_predictions)

        return bag_predictions, all_cumulative_bag_predictions

    def partial_forward(self, instance, hidden_state, cell_state, prev_cumulative_bag_prediction):
        # Embed the instance
        instance = instance.to(self.device)
        instance_embedding = self.encoder(instance)
        # Pass the embedding through the aggregator with the given states
        agg_out = self.aggregator.partial_forward(instance_embedding, hidden_state, cell_state,
                                                  prev_cumulative_bag_prediction)
        # Return instance predictions and new states
        instance_prediction, new_hidden_state, new_cell_state = agg_out
        return instance_prediction, new_hidden_state, new_cell_state

    def get_hidden_states(self, bag):
        instances = bag.to(self.device)
        instance_embeddings = self.encoder(instances)
        _, hidden_states = self.aggregator.lstm_block(torch.unsqueeze(instance_embeddings, 0))
        return hidden_states.squeeze()

    def flatten_parameters(self):
        self.aggregator.flatten_parameters()


class LstmBlock(nn.Module):

    def __init__(self, d_in, d_hid, n_layers, bidirectional, dropout):
        super().__init__()
        # For the LSTM block, a non-zero dropout expects num_layers greater than 1
        self.lstm = nn.LSTM(input_size=d_in, hidden_size=d_hid, num_layers=n_layers,
                            bidirectional=bidirectional, batch_first=True, dropout=0 if n_layers == 1 else dropout)
        self.bidirectional = bidirectional
        self.d_hid = d_hid
        self.dropout = nn.Dropout(p=dropout)
        self.init_hidden = nn.Parameter(torch.zeros(2 * n_layers if bidirectional else n_layers, 1, d_hid))
        self.init_cell = nn.Parameter(torch.zeros(2 * n_layers if bidirectional else n_layers, 1, d_hid))

    def forward(self, x):
        _, n_instances, _ = x.shape

        # Pass through lstm
        out, (ht, _) = self.lstm(x, (self.init_hidden, self.init_cell))

        # Get lstm output
        if self.bidirectional:
            out_split = out.view(1, n_instances, 2, self.d_hid)
            forward_out = out_split[:, :, 0, :]
            backward_out = out_split[:, :, 1, :]
            bag_repr = torch.cat([forward_out[:, -1, :], backward_out[:, 0, :]], dim=1)
        else:
            bag_repr = out[:, -1, :]

        # bag_repr = self.dropout(bag_repr)
        # out = self.dropout(out)
        return bag_repr, out

    def partial_forward(self, instance_embedding, hidden_state, cell_state):
        # Forward pass but we already have a hidden and cell state, and are only doing it for one instance
        instance_embedding = instance_embedding.unsqueeze(0).unsqueeze(0)  # Need a 3D input, currently 1D
        out, (new_hidden_state, new_cell_state) = self.lstm(instance_embedding, (hidden_state, cell_state))
        bag_repr = out[:, -1, :]
        bag_repr = self.dropout(bag_repr)
        return bag_repr, new_hidden_state, new_cell_state

    def flatten_parameters(self):
        self.lstm.flatten_parameters()


class LstmEmbeddingSpaceAggregator(nn.Module):
    """
    An LSTM Aggregator that only makes the bag prediction based on the final hidden state.
    """

    def __init__(self, d_in, d_hid, n_lstm_layers, bidirectional, dropout, ds_hid, n_classes,
                 classifier_final_activation_func=None):
        super().__init__()
        self.lstm_block = LstmBlock(d_in, d_hid, n_lstm_layers, bidirectional, dropout)
        self.embedding_size = d_hid * 2 if bidirectional else d_hid
        self.embedding_classifier = FullyConnectedStack(self.embedding_size, ds_hid, n_classes,
                                                        final_activation_func=classifier_final_activation_func,
                                                        dropout=dropout)

    def forward(self, instance_embeddings):
        # Pass through lstm block. Unsqueeze as lstm block expects a 3D input
        bag_embedding, cumulative_bag_embeddings = self.lstm_block(torch.unsqueeze(instance_embeddings, 0))
        # Get bag prediction
        bag_prediction = self.embedding_classifier(bag_embedding)

        # Get cumulative instance predictions if not training
        cumulative_predictions = None
        if not self.training:
            with torch.no_grad():
                cumulative_predictions = self.embedding_classifier(cumulative_bag_embeddings)
                if self.classifier_activation_func is not None:
                    cumulative_predictions = self.classifier_activation_func(cumulative_predictions)
        return bag_prediction, cumulative_predictions

    def partial_forward(self, instance_embedding, hidden_state, cell_state, prev_cumulative_bag_prediction):
        # Pass instance embedding and states through lstm block
        lstm_out = self.lstm_block.partial_forward(instance_embedding, hidden_state, cell_state)
        # Bag repr is just for the one instance, and we also get new states
        bag_repr, new_hidden_state, new_cell_state = lstm_out
        # Classify bag representation and calculate instance prediction
        cumulative_bag_prediction = self.embedding_classifier(bag_repr)
        if self.classifier_activation_func is not None:
            cumulative_bag_prediction = self.classifier_activation_func(cumulative_bag_prediction)
        instance_prediction = cumulative_bag_prediction - prev_cumulative_bag_prediction
        # Return the instance prediction and new states
        return instance_prediction, new_hidden_state, new_cell_state

    def flatten_parameters(self):
        self.lstm_block.flatten_parameters()


class LstmInstanceSpaceAggregator(nn.Module):
    """
    An LSTM Aggregator that makes the bag prediction by predicting over all cumulative hidden states and then
    performing some aggregation (e.g., mean, sum) over all the instance predictions.
    Assumes forward direction only.
    """

    def __init__(self, d_in, d_hid, n_lstm_layers, dropout, ds_hid, n_classes, agg_func_name,
                 classifier_final_activation_func=None):
        super().__init__()
        self.lstm_block = LstmBlock(d_in, d_hid, n_lstm_layers, False, dropout)
        self.embedding_size = d_hid
        self.embedding_classifier = FullyConnectedStack(self.embedding_size, ds_hid, n_classes,
                                                        final_activation_func=classifier_final_activation_func,
                                                        dropout=dropout)
        self.aggregation_func = self._parse_agg_method(agg_func_name)

    def forward(self, instance_embeddings):
        # Pass through lstm block. Unsqueeze as lstm block expects a 3D input
        _, cumulative_bag_embeddings = self.lstm_block(torch.unsqueeze(instance_embeddings, 0))
        # Get prediction for each instance
        instance_predictions = self.embedding_classifier(cumulative_bag_embeddings)

        # Aggregate to bag prediction
        bag_prediction = self.aggregation_func(instance_predictions.squeeze())
        return bag_prediction, instance_predictions

    def partial_forward(self, instance_embedding, hidden_state, cell_state, prev_cumulative_bag_prediction):
        # Pass instance embedding and states through lstm block
        lstm_out = self.lstm_block.partial_forward(instance_embedding, hidden_state, cell_state)
        # Bag repr is just for the one instance, and we also get new states
        bag_repr, new_hidden_state, new_cell_state = lstm_out
        # Classify bag representation
        instance_prediction = self.embedding_classifier(bag_repr)
        if self.classifier_activation_func is not None:
            instance_prediction = self.classifier_activation_func(instance_prediction)
        # Return the instance prediction and new states
        return instance_prediction, new_hidden_state, new_cell_state

    def flatten_parameters(self):
        self.lstm_block.flatten_parameters()


class LstmASCInstanceSpaceAggregator(nn.Module):
    """
    Additive Skip Connections
    An LSTM Aggregator that makes the bag prediction by predicting over all cumulative hidden states added to
    their respective instance and then performing some aggregation (e.g., mean, sum) over all the instance predictions.
    Assumes forward direction only.
    """

    def __init__(self, d_in, d_hid, n_lstm_layers, dropout, ds_hid, n_classes, agg_func_name,
                 classifier_final_activation_func=None):
        super().__init__()
        self.lstm_block = LstmBlock(d_in, d_hid, n_lstm_layers, False, dropout)
        self.embedding_size = d_hid
        self.embedding_classifier = FullyConnectedStack(self.embedding_size, ds_hid, n_classes,
                                                        final_activation_func=classifier_final_activation_func,
                                                        dropout=dropout)
        self.skip_projection = nn.Linear(d_in, d_hid)
        self.aggregation_func = self._parse_agg_method(agg_func_name)

    def forward(self, instance_embeddings):
        # Pass through lstm block. Unsqueeze as lstm block expects a 3D input
        _, cumulative_bag_embeddings = self.lstm_block(torch.unsqueeze(instance_embeddings, 0))

        # Pass instance embeddings through the skip projection to match the size of the lstm block
        skip_embeddings = self.skip_projection(instance_embeddings)
        # Add together the lstm hidden states and the skip embeddings
        skip_reprs = cumulative_bag_embeddings.squeeze(0) + skip_embeddings

        # Get prediction for each instance
        instance_predictions = self.embedding_classifier(skip_reprs)

        # Aggregate to bag prediction
        bag_prediction = self.aggregation_func(instance_predictions.squeeze())
        return bag_prediction, instance_predictions

    def flatten_parameters(self):
        self.lstm_block.flatten_parameters()


class LstmCSCInstanceSpaceAggregator(nn.Module):
    """
    Concatenated Skip Connections
    An LSTM Aggregator that makes the bag prediction by predicting over all cumulative hidden states concatenated to
    their respective instance and then performing some aggregation (e.g., mean, sum) over all the instance predictions.
    Assumes forward direction only.
    """

    def __init__(self, d_in, d_hid, n_lstm_layers, dropout, ds_hid, n_classes, agg_func_name,
                 classifier_final_activation_func=None):
        super().__init__()
        self.lstm_block = LstmBlock(d_in, d_hid, n_lstm_layers, False, dropout)
        self.embedding_size = d_in + d_hid
        self.embedding_classifier = FullyConnectedStack(self.embedding_size, ds_hid, n_classes,
                                                        final_activation_func=classifier_final_activation_func,
                                                        dropout=dropout)
        # self.skip_projection = nn.Linear(d_in, d_hid)
        self.aggregation_func = self._parse_agg_method(agg_func_name)

    def forward(self, instance_embeddings):
        # Pass through lstm block. Unsqueeze as lstm block expects a 3D input
        _, cumulative_bag_embeddings = self.lstm_block(torch.unsqueeze(instance_embeddings, 0))

        # Concatenate the instance embeddings with the cumulative bag embeddings
        skip_reprs = torch.cat((instance_embeddings, cumulative_bag_embeddings.squeeze(0)), dim=1)

        # Get prediction for each instance
        instance_predictions = self.embedding_classifier(skip_reprs)

        # Aggregate to bag prediction
        bag_prediction = self.aggregation_func(instance_predictions.squeeze())
        return bag_prediction, instance_predictions

    def partial_forward(self, instance_embedding, hidden_state, cell_state, prev_cumulative_bag_prediction):
        # Pass instance embedding and states through lstm block
        lstm_out = self.lstm_block.partial_forward(instance_embedding, hidden_state, cell_state)
        # Bag repr is just for the one instance, and we also get new states
        bag_repr, new_hidden_state, new_cell_state = lstm_out
        # Create the skip representation and classify it
        skip_repr = torch.cat((instance_embedding.unsqueeze(0), bag_repr), dim=1)
        instance_prediction = self.embedding_classifier(skip_repr)
        if self.classifier_activation_func is not None:
            instance_prediction = self.classifier_activation_func(instance_prediction)
        # Return the instance prediction and new states
        return instance_prediction, new_hidden_state, new_cell_state

    def flatten_parameters(self):
        self.lstm_block.flatten_parameters()
