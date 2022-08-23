from abc import ABC, abstractmethod

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data.data import Data
from torch_geometric.nn import SAGEConv, dense_diff_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse

from bonfire.model import aggregator as agg
from bonfire.model import modules as mod


class MultipleInstanceModel(nn.Module, ABC):

    def __init__(self, device, n_classes, n_expec_dims):
        super().__init__()
        self.device = device
        self.n_classes = n_classes
        self.n_expec_dims = n_expec_dims

    @classmethod
    @property
    @abstractmethod
    def name(cls):
        pass

    @abstractmethod
    def forward(self, model_input):
        pass

    @abstractmethod
    def forward_verbose(self, model_input):
        pass

    @abstractmethod
    def _internal_forward(self, bags):
        pass

    def suggest_train_params(self):
        return {}


class MultipleInstanceNN(MultipleInstanceModel, ABC):

    def forward(self, model_input):
        # We don't care about any interpretability output here
        bag_predictions, _ = self.forward_verbose(model_input)
        return bag_predictions

    def forward_verbose(self, model_input):
        unbatched_bag = False
        # Model input is tensor
        if torch.is_tensor(model_input):
            input_shape = model_input.shape
            unbatched_bag = len(input_shape) == self.n_expec_dims
            if unbatched_bag:
                # Just a single bag on its own, not in a batch, therefore stick it in a list
                bags = [model_input]
            else:
                # Assume already batched
                bags = model_input
        # Model input is list
        elif type(model_input) == list:
            # Assume already batched
            bags = model_input
        # Invalid input type
        else:
            raise ValueError('Invalid model input type {:}'.format(type(model_input)))

        # Actually pass the input through the model
        #  Note instance interpretations has a different meaning depending on the model
        #  They can be, but are not always, instance predictions (e.g., attention).
        bag_predictions, instance_interpretations = self._internal_forward(bags)

        # If given input was not batched, also un-batch the output
        if unbatched_bag:
            return bag_predictions[0], instance_interpretations[0]
        return bag_predictions, instance_interpretations


class InstanceSpaceNN(MultipleInstanceNN, ABC):

    name = "InstanceSpaceNN"

    def __init__(self, device, n_classes, n_expec_dims, encoder, aggregator):
        super().__init__(device, n_classes, n_expec_dims)
        self.encoder = encoder
        self.aggregator = aggregator

    def _internal_forward(self, bags):
        batch_size = len(bags)
        bag_predictions = torch.zeros((batch_size, self.n_classes)).to(self.device)
        # Bags may be of different sizes, so we can't use a tensor to store the instance predictions
        bag_instance_predictions = []
        for i, instances in enumerate(bags):
            # Embed instances
            instances = instances.to(self.device)
            instance_embeddings = self.encoder(instances)

            # Classify instances and aggregate
            #  Here, the instance interpretations are actually predictions
            bag_prediction, instance_predictions = self.aggregator(instance_embeddings)

            # Update outputs
            bag_predictions[i] = bag_prediction
            bag_instance_predictions.append(instance_predictions)
        return bag_predictions, bag_instance_predictions


class EmbeddingSpaceNN(MultipleInstanceNN, ABC):

    name = "EmbeddingSpaceNN"

    def __init__(self, device, n_classes, n_expec_dims, encoder, aggregator):
        super().__init__(device, n_classes, n_expec_dims)
        self.encoder = encoder
        self.aggregator = aggregator

    def _internal_forward(self, bags):
        bag_predictions = torch.zeros((len(bags), self.n_classes)).to(self.device)
        for i, instances in enumerate(bags):
            # Embed instances
            instances = instances.to(self.device)
            instance_embeddings = self.encoder(instances)

            # Classify instances and aggregate
            #  This model does not produce any instance interpretations
            bag_prediction, _ = self.aggregator(instance_embeddings)

            # Update outputs
            bag_predictions[i] = bag_prediction
        return bag_predictions, None


class AttentionNN(MultipleInstanceNN, ABC):

    name = "AttentionNN"

    def __init__(self, device, n_classes, n_expec_dims, encoder, aggregator):
        super().__init__(device, n_classes, n_expec_dims)
        self.encoder = encoder
        self.aggregator = aggregator

    def _internal_forward(self, bags):
        bag_predictions = torch.zeros((len(bags), self.n_classes)).to(self.device)
        all_attention_values = []
        for i, instances in enumerate(bags):
            # Embed instances
            instances = instances.to(self.device)
            instance_embeddings = self.encoder(instances)

            # Classify instances and aggregate
            #  Instance interpretations is the attention value assigned to each instance in the bag
            bag_prediction, instance_attention_values = self.aggregator(instance_embeddings)

            # Update outputs
            bag_predictions[i] = bag_prediction
            all_attention_values.append(instance_attention_values)
        return bag_predictions, all_attention_values


class ClusterGNN(MultipleInstanceModel, ABC):

    name = "ClusterGNN"

    def __init__(self, device, n_classes, n_expec_dims, encoder, d_enc, d_gnn, ds_gnn_hid, ds_fc_hid, dropout):
        super().__init__(device, n_classes, n_expec_dims)
        self.n_clusters = 1
        self.encoder = encoder
        self.gnn_stack = mod.GNNConvStack(d_enc, ds_gnn_hid, d_gnn, dropout, SAGEConv, raw_last=False)
        # TODO this should be called GNN cluster but the trained models are stuck with gnn_pool
        self.gnn_pool = SAGEConv(d_enc, self.n_clusters)
        self.classifier = mod.FullyConnectedStack(d_gnn, ds_fc_hid, n_classes, dropout, raw_last=True)

    def forward(self, model_input):
        # We don't care about any interpretability output here
        bag_predictions, _ = self.forward_verbose(model_input)
        return bag_predictions

    def forward_verbose(self, model_input):
        # TODO ClusterGNN doesn't deal with batched inputs
        # Unbatched input could be Data type, or raw tensor without Data structure
        unbatched_bag = type(model_input) is Data or \
                        (type(model_input) is torch.Tensor and len(model_input.shape) == self.n_expec_dims)

        # Batch if single bag
        bags = [model_input] if unbatched_bag else model_input

        # Actually pass the input through the model
        bag_predictions, bag_cluster_weights = self._internal_forward(bags)

        # Return single pred if unbatched_bag bag else multiple preds
        if unbatched_bag:
            return bag_predictions.squeeze(), bag_cluster_weights[0]
        return bag_predictions, bag_cluster_weights

    def _internal_forward(self, bags):
        bag_predictions = torch.zeros((len(bags), self.n_classes)).to(self.device)
        bag_cluster_weights = []
        for i, bag in enumerate(bags):
            # If given Tensor and not Data type, add fully connected graph to data
            if type(bag) is not Data:
                bag = self.add_graph(bag)
            bag = bag.to(self.device)
            x, edge_index = bag.x, bag.edge_index

            # Embed instances
            instance_embeddings = self.encoder(x)

            # Run instance embeddings through GNN stack for message passing
            updated_instance_embeddings, _ = self.gnn_stack((instance_embeddings, edge_index))

            # Clustering
            cluster_weights = F.softmax(self.gnn_pool(instance_embeddings, edge_index), dim=0)

            # Reduce to single representation of the graph
            graph_embedding, _, _, _ = dense_diff_pool(updated_instance_embeddings, to_dense_adj(edge_index),
                                                       cluster_weights)
            graph_embedding = graph_embedding.squeeze(dim=0)

            # Classify
            bag_prediction = self.classifier(graph_embedding)

            # Update outputs
            bag_predictions[i] = bag_prediction
            bag_cluster_weights.append(cluster_weights.T)
        return bag_predictions, bag_cluster_weights

    @staticmethod
    def add_graph(bag):
        edge_index = torch.ones((len(bag), len(bag)))
        edge_index, _ = dense_to_sparse(edge_index)
        edge_index = edge_index.long().contiguous()
        graph_bag = Data(x=bag, edge_index=edge_index)
        return graph_bag


class MiLstm(MultipleInstanceNN, ABC):

    def __init__(self, device, n_classes, n_expec_dims, encoder, aggregator):
        super().__init__(device, n_classes, n_expec_dims)
        self.encoder = encoder
        self.aggregator = aggregator

    # -- UNUSED --
    # def create_mi_networks(self):
    #     net_discrim_global = nn.Sequential(
    #         nn.Linear(self.encoder.d_in + self.aggregator.embedding_size, 512),
    #         nn.ReLU(),
    #         nn.Linear(512, 512),
    #         nn.ReLU(),
    #         nn.Linear(512, 1),
    #     )
    #     net_discrim_local = nn.Sequential(
    #         nn.Linear(self.encoder.d_in + self.encoder.d_out, 512),
    #         nn.ReLU(),
    #         nn.Linear(512, 512),
    #         nn.ReLU(),
    #         nn.Linear(512, 1),
    #     )
    #     net_discrim_prior = nn.Sequential(
    #         nn.Linear(self.aggregator.embedding_size, 1000),
    #         nn.ReLU(),
    #         nn.Linear(1000, 200),
    #         nn.ReLU(),
    #         nn.Linear(200, 1),
    #         nn.Sigmoid()
    #     )
    #     return net_discrim_global, net_discrim_local, net_discrim_prior

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

        # -- UNUSED --
        # Second pass: calculate MI scores
        # all_mi_scores = torch.zeros((len(bags), 3))
        # if self.use_mi:
        #     for i, instances in enumerate(bags):
        #         instances = instances.to(self.device)
        #         instance_embeddings = all_instance_embeddings[i]
        #         bag_embedding = all_bag_embedding[i]
        #
        #         if i < len(bags) - 1:
        #             prev_instance_embeddings = all_instance_embeddings[i + 1]
        #             prev_bag_embedding = all_bag_embedding[i + 1]
        #         else:
        #             prev_instance_embeddings = all_instance_embeddings[0]
        #             prev_bag_embedding = all_bag_embedding[0]
        #
        #         mi_scores = self._forward_mi(instances, instance_embeddings, bag_embedding,
        #                                      prev_instance_embeddings, prev_bag_embedding)
        #         all_mi_scores[i] = mi_scores

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

    # -- UNUSED --
    # def _forward_mi(self, instances, instance_embeddings, bag_embedding, alt_instance_embeddings, alt_bag_embedding):
    #
    #     mi_scores = torch.zeros(3)  # Global, local, prior
    #
    #     #  Global: instance + bag representation
    #     actual_global_pairings = torch.cat([instances, bag_embedding.repeat(instances.shape[0], 1)], dim=1)
    #     random_global_pairings = torch.cat([instances, alt_bag_embedding.repeat(instances.shape[0], 1)], dim=1)
    #     global_actual = -F.softplus(-self.net_discrim_global(actual_global_pairings)).mean()
    #     global_random = F.softplus(self.net_discrim_global(random_global_pairings)).mean()
    #     mi_global = global_random - global_actual
    #
    #     # Local: instance + instance embedding
    #     actual_local_pairings = torch.cat([instances, instance_embeddings], dim=1)
    #     # Sample n random indices of our alt instance embeddings, where n is our number of instances in the actual bag
    #     random_perm = np.random.choice(alt_instance_embeddings.shape[0], instances.shape[0])
    #     random_local_pairings = torch.cat([instances, alt_instance_embeddings[random_perm, :]], dim=1)
    #     local_actual = -F.softplus(-self.net_discrim_local(actual_local_pairings)).mean()
    #     local_random = F.softplus(self.net_discrim_local(random_local_pairings)).mean()
    #     mi_local = local_random - local_actual
    #
    #     # Prior
    #     prior = torch.rand_like(bag_embedding)
    #     term_a = torch.log(self.net_discrim_prior(prior)).mean()
    #     term_b = torch.log(1.0 - self.net_discrim_prior(bag_embedding)).mean()
    #     mi_prior = - (term_a + term_b)
    #
    #     mi_scores[0] = mi_global
    #     mi_scores[1] = mi_local
    #     mi_scores[2] = mi_prior
    #
    #     return mi_scores

    def get_hidden_states(self, bag):
        instances = bag.to(self.device)
        instance_embeddings = self.encoder(instances)
        _, hidden_states = self.aggregator.lstm_block(torch.unsqueeze(instance_embeddings, 0))
        return hidden_states.squeeze()

    def flatten_parameters(self):
        self.aggregator.flatten_parameters()


class EmbeddingSpaceLSTM(MiLstm, ABC):

    def __init__(self, device, n_classes, n_expec_dims, encoder, aggregator):
        super().__init__(device, n_classes, n_expec_dims, encoder, aggregator)
        assert type(aggregator) == agg.LstmEmbeddingSpaceAggregator


class InstanceSpaceLSTM(MiLstm, ABC):

    def __init__(self, device, n_classes, n_expec_dims, encoder, aggregator):
        super().__init__(device, n_classes, n_expec_dims, encoder, aggregator)
        assert type(aggregator) == agg.LstmInstanceSpaceAggregator


class CSCInstanceSpaceLSTM(MiLstm, ABC):
    """
    Concatenated skip connection instance space LSTM.
    """

    def __init__(self, device, n_classes, n_expec_dims, encoder, aggregator):
        super().__init__(device, n_classes, n_expec_dims, encoder, aggregator)
        assert type(aggregator) == agg.LstmCSCInstanceSpaceAggregator
