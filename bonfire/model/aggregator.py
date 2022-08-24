from abc import ABC

import torch
from torch import nn

from bonfire.model import modules as mod


class Aggregator(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _parse_agg_method(agg_func_name):
        # TODO ensure these are the correct shape
        if agg_func_name == 'mean':
            def mean_agg(x):
                if len(x.shape) == 1:  # n_instances
                    return torch.mean(x)
                elif len(x.shape) == 2:  # n_instances * encoding_dim
                    return torch.mean(x, dim=0)
                raise NotImplementedError('Check shape!')
            return mean_agg
        if agg_func_name == 'max':
            def max_agg(x):
                if len(x.shape) == 1:  # n_instances
                    return torch.max(x)
                elif len(x.shape) == 2:  # n_instances * encoding_dim
                    return torch.max(x, dim=0)[0]
                raise NotImplementedError('Check shape!')
            return max_agg
        if agg_func_name == 'sum':
            def sum_agg(x):
                if len(x.shape) == 1:   # n_instances
                    return torch.sum(x)
                elif len(x.shape) == 2:  # n_instances * encoding_dim
                    return torch.sum(x, dim=0)
                raise NotImplementedError('Check shape!')
            return sum_agg
        raise ValueError('Invalid aggregation function name for Instance Aggregator: {:s}'.format(agg_func_name))


class InstanceAggregator(Aggregator):

    def __init__(self, d_in, ds_hid, n_classes, dropout, agg_func_name, classifier_raw_last=True,
                 classifier_activation_func=None):
        super().__init__()
        self.instance_classifier = mod.FullyConnectedStack(d_in, ds_hid, n_classes, dropout,
                                                           raw_last=classifier_raw_last)
        self.aggregation_func = self._parse_agg_method(agg_func_name)
        self.classifier_activation_func = classifier_activation_func

    def forward(self, instance_embeddings):
        instance_predictions = self.instance_classifier(instance_embeddings)
        if self.classifier_activation_func is not None:
            instance_predictions = self.classifier_activation_func(instance_predictions)
        bag_prediction = self.aggregation_func(instance_predictions)
        return bag_prediction, instance_predictions


class EmbeddingAggregator(Aggregator):

    def __init__(self, d_in, ds_hid, n_classes, dropout, agg_func_name):
        super().__init__()
        self.aggregation_func = self._parse_agg_method(agg_func_name)
        self.embedding_classifier = mod.FullyConnectedStack(d_in, ds_hid, n_classes, dropout, raw_last=True)

    def forward(self, instance_embeddings):
        bag_embedding = self.aggregation_func(instance_embeddings)
        bag_prediction = self.embedding_classifier(bag_embedding)
        return bag_prediction, None


class MultiHeadAttentionAggregator(Aggregator):

    def __init__(self, n_heads, d_in, ds_hid, d_attn, n_classes, dropout):
        super().__init__()
        self.attention_aggregator = mod.MultiHeadAttentionBlock(n_heads, d_in, d_attn, dropout)
        self.embedding_classifier = mod.FullyConnectedStack(n_heads * d_in, ds_hid, n_classes, dropout, raw_last=True)

    def forward(self, instance_embeddings):
        bag_embedding, attn = self.attention_aggregator(instance_embeddings)
        bag_prediction = self.embedding_classifier(bag_embedding)
        return bag_prediction, attn


class CountAggregator(Aggregator):

    def __init__(self, base_aggregator):
        super().__init__()
        self.base_aggregator = base_aggregator
        self.relu = torch.nn.ReLU()

    def forward(self, instance_embeddings):
        bag_prediction, instance_attributions = self.base_aggregator(instance_embeddings)
        bag_prediction = self.relu(bag_prediction)
        # print(bag_prediction)
        # print(instance_attributions)
        return bag_prediction, instance_attributions


class LstmEmbeddingSpaceAggregator(Aggregator):
    """
    An LSTM Aggregator that only makes the bag prediction based on the final hidden state.
    """

    def __init__(self, d_in, d_hid, n_lstm_layers, bidirectional, dropout, ds_hid, n_classes, classifier_raw_last=True,
                 classifier_activation_func=None):
        super().__init__()
        self.lstm_block = mod.LstmBlock(d_in, d_hid, n_lstm_layers, bidirectional, dropout)
        self.embedding_size = d_hid * 2 if bidirectional else d_hid
        self.embedding_classifier = mod.FullyConnectedStack(self.embedding_size, ds_hid, n_classes,
                                                            dropout, raw_last=classifier_raw_last)
        self.classifier_activation_func = classifier_activation_func

    def forward(self, instance_embeddings):
        # Pass through lstm block. Unsqueeze as lstm block expects a 3D input
        bag_embedding, cumulative_bag_embeddings = self.lstm_block(torch.unsqueeze(instance_embeddings, 0))
        # Get bag prediction
        bag_prediction = self.embedding_classifier(bag_embedding)
        if self.classifier_activation_func is not None:
            bag_prediction = self.classifier_activation_func(bag_prediction)

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


class LstmInstanceSpaceAggregator(Aggregator):
    """
    An LSTM Aggregator that makes the bag prediction by predicting over all cumulative hidden states and then
    performing some aggregation (e.g., mean, sum) over all the instance predictions.
    Assumes forward direction only.
    """

    def __init__(self, d_in, d_hid, n_lstm_layers, dropout, ds_hid, n_classes, agg_func_name, classifier_raw_last=True,
                 classifier_activation_func=None):
        super().__init__()
        self.lstm_block = mod.LstmBlock(d_in, d_hid, n_lstm_layers, False, dropout)
        self.embedding_size = d_hid
        self.embedding_classifier = mod.FullyConnectedStack(self.embedding_size, ds_hid, n_classes,
                                                            dropout, raw_last=classifier_raw_last)
        self.aggregation_func = self._parse_agg_method(agg_func_name)
        self.classifier_activation_func = classifier_activation_func

    def forward(self, instance_embeddings):
        # Pass through lstm block. Unsqueeze as lstm block expects a 3D input
        _, cumulative_bag_embeddings = self.lstm_block(torch.unsqueeze(instance_embeddings, 0))
        # Get prediction for each instance
        instance_predictions = self.embedding_classifier(cumulative_bag_embeddings)
        if self.classifier_activation_func is not None:
            instance_predictions = self.classifier_activation_func(instance_predictions)

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


class LstmASCInstanceSpaceAggregator(Aggregator):
    """
    Additive Skip Connections
    An LSTM Aggregator that makes the bag prediction by predicting over all cumulative hidden states added to
    their respective instance and then performing some aggregation (e.g., mean, sum) over all the instance predictions.
    Assumes forward direction only.
    """

    def __init__(self, d_in, d_hid, n_lstm_layers, dropout, ds_hid, n_classes, agg_func_name, classifier_raw_last=True,
                 classifier_activation_func=None):
        super().__init__()
        self.lstm_block = mod.LstmBlock(d_in, d_hid, n_lstm_layers, False, dropout)
        self.embedding_size = d_hid
        self.embedding_classifier = mod.FullyConnectedStack(self.embedding_size, ds_hid, n_classes,
                                                            dropout, raw_last=classifier_raw_last)
        self.skip_projection = nn.Linear(d_in, d_hid)
        self.aggregation_func = self._parse_agg_method(agg_func_name)
        self.classifier_activation_func = classifier_activation_func

    def forward(self, instance_embeddings):
        # Pass through lstm block. Unsqueeze as lstm block expects a 3D input
        _, cumulative_bag_embeddings = self.lstm_block(torch.unsqueeze(instance_embeddings, 0))

        # Pass instance embeddings through the skip projection to match the size of the lstm block
        skip_embeddings = self.skip_projection(instance_embeddings)
        # Add together the lstm hidden states and the skip embeddings
        skip_reprs = cumulative_bag_embeddings.squeeze(0) + skip_embeddings

        # Get prediction for each instance
        instance_predictions = self.embedding_classifier(skip_reprs)
        if self.classifier_activation_func is not None:
            instance_predictions = self.classifier_activation_func(instance_predictions)

        # Aggregate to bag prediction
        bag_prediction = self.aggregation_func(instance_predictions.squeeze())
        return bag_prediction, instance_predictions

    def flatten_parameters(self):
        self.lstm_block.flatten_parameters()


class LstmCSCInstanceSpaceAggregator(Aggregator):
    """
    Concatenated Skip Connections
    An LSTM Aggregator that makes the bag prediction by predicting over all cumulative hidden states concatenated to
    their respective instance and then performing some aggregation (e.g., mean, sum) over all the instance predictions.
    Assumes forward direction only.
    """

    def __init__(self, d_in, d_hid, n_lstm_layers, dropout, ds_hid, n_classes, agg_func_name, classifier_raw_last=True,
                 classifier_activation_func=None):
        super().__init__()
        self.lstm_block = mod.LstmBlock(d_in, d_hid, n_lstm_layers, False, dropout)
        self.embedding_size = d_in + d_hid
        self.embedding_classifier = mod.FullyConnectedStack(self.embedding_size, ds_hid, n_classes,
                                                            dropout, raw_last=classifier_raw_last)
        # self.skip_projection = nn.Linear(d_in, d_hid)
        self.aggregation_func = self._parse_agg_method(agg_func_name)
        self.classifier_activation_func = classifier_activation_func

    def forward(self, instance_embeddings):
        # Pass through lstm block. Unsqueeze as lstm block expects a 3D input
        _, cumulative_bag_embeddings = self.lstm_block(torch.unsqueeze(instance_embeddings, 0))

        # Concatenate the instance embeddings with the cumulative bag embeddings
        skip_reprs = torch.cat((instance_embeddings, cumulative_bag_embeddings.squeeze(0)), dim=1)

        # Get prediction for each instance
        instance_predictions = self.embedding_classifier(skip_reprs)
        if self.classifier_activation_func is not None:
            instance_predictions = self.classifier_activation_func(instance_predictions)

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
