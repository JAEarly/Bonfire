from abc import ABC

import torch
from torch import nn

from pytorch_mil.model import modules as mod


class Aggregator(nn.Module, ABC):

    def __init__(self):
        super().__init__()

    @staticmethod
    def _parse_agg_method(agg_func_name):
        # TODO ensure these are the correct shape
        if agg_func_name == 'mean':
            raise NotImplementedError('Check shape!')
            # return lambda x: torch.mean(x, dim=0)
        if agg_func_name == 'max':
            raise NotImplementedError('Check shape!')
            # return lambda x: torch.max(x, dim=0)[0]
        if agg_func_name == 'sum':
            def sum_agg(x):
                assert len(x.shape) == 1   # n_instances
                return torch.sum(x)
            return sum_agg
        raise ValueError('Invalid aggregation function name for Instance Aggregator: {:s}'.format(agg_func_name))


class InstanceAggregator(Aggregator):

    def __init__(self, d_in, ds_hid, n_classes, dropout, agg_func_name):
        super().__init__()
        self.instance_classifier = mod.FullyConnectedStack(d_in, ds_hid, n_classes, dropout, raw_last=True)
        self.aggregation_func = self._parse_agg_method(agg_func_name)

    def forward(self, instance_embeddings):
        instance_predictions = self.instance_classifier(instance_embeddings)
        bag_prediction = self.aggregation_func(instance_predictions.squeeze())
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

    def __init__(self, d_in, d_hid, n_lstm_layers, bidirectional, dropout, ds_hid, n_classes):
        super().__init__()
        self.lstm_block = mod.LstmBlock(d_in, d_hid, n_lstm_layers, bidirectional, dropout)
        self.embedding_size = d_hid * 2 if bidirectional else d_hid
        self.embedding_classifier = mod.FullyConnectedStack(self.embedding_size, ds_hid, n_classes,
                                                            dropout, raw_last=True)

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

        return bag_prediction, cumulative_predictions

    def flatten_parameters(self):
        self.lstm_block.flatten_parameters()


class LstmInstanceSpaceAggregator(Aggregator):
    """
    An LSTM Aggregator that makes the bag prediction by predicting over all cumulative hidden states and then
    performing some aggregation (e.g., mean, sum) over all the instance predictions.
    Assumes forward direction only.
    """

    def __init__(self, d_in, d_hid, n_lstm_layers, dropout, ds_hid, n_classes, agg_func_name):
        super().__init__()
        self.lstm_block = mod.LstmBlock(d_in, d_hid, n_lstm_layers, False, dropout)
        self.embedding_size = d_hid
        self.embedding_classifier = mod.FullyConnectedStack(self.embedding_size, ds_hid, n_classes,
                                                            dropout, raw_last=True)
        self.aggregation_func = self._parse_agg_method(agg_func_name)

    def forward(self, instance_embeddings):
        # Pass through lstm block. Unsqueeze as lstm block expects a 3D input
        _, cumulative_bag_embeddings = self.lstm_block(torch.unsqueeze(instance_embeddings, 0))
        # Get prediction for each instance
        instance_predictions = self.embedding_classifier(cumulative_bag_embeddings)
        # Aggregate to bag prediction
        bag_prediction = self.aggregation_func(instance_predictions.squeeze())
        return bag_prediction, instance_predictions

    def flatten_parameters(self):
        self.lstm_block.flatten_parameters()


class LstmASCInstanceSpaceAggregator(Aggregator):
    """
    Additive Skip Connections
    An LSTM Aggregator that makes the bag prediction by predicting over all cumulative hidden states added to
    their respective instance and then performing some aggregation (e.g., mean, sum) over all the instance predictions.
    Assumes forward direction only.
    """

    def __init__(self, d_in, d_hid, n_lstm_layers, dropout, ds_hid, n_classes, agg_func_name):
        super().__init__()
        self.lstm_block = mod.LstmBlock(d_in, d_hid, n_lstm_layers, False, dropout)
        self.embedding_size = d_hid
        self.embedding_classifier = mod.FullyConnectedStack(self.embedding_size, ds_hid, n_classes,
                                                            dropout, raw_last=True)
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


class LstmCSCInstanceSpaceAggregator(Aggregator):
    """
    Concatenated Skip Connections
    An LSTM Aggregator that makes the bag prediction by predicting over all cumulative hidden states concatenated to
    their respective instance and then performing some aggregation (e.g., mean, sum) over all the instance predictions.
    Assumes forward direction only.
    """

    def __init__(self, d_in, d_hid, n_lstm_layers, dropout, ds_hid, n_classes, agg_func_name):
        super().__init__()
        self.lstm_block = mod.LstmBlock(d_in, d_hid, n_lstm_layers, False, dropout)
        self.embedding_size = d_in + d_hid
        self.embedding_classifier = mod.FullyConnectedStack(self.embedding_size, ds_hid, n_classes,
                                                            dropout, raw_last=True)
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

    def partial_forward(self, instance_embedding, hidden_state, cell_state):
        # Pass instance embedding and states through lstm block
        lstm_out = self.lstm_block.partial_forward(instance_embedding, hidden_state, cell_state)
        # Bag repr is just for the one instance, and we also get new states
        bag_repr, new_hidden_state, new_cell_state = lstm_out
        # Create the skip representation and classify it
        skip_repr = torch.cat((instance_embedding.unsqueeze(0), bag_repr), dim=1)
        instance_prediction = self.embedding_classifier(skip_repr)
        # Return the instance prediction and new states
        return instance_prediction, new_hidden_state, new_cell_state

    def flatten_parameters(self):
        self.lstm_block.flatten_parameters()
