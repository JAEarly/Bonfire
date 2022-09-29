import random

import torch
# from torch_geometric.data import Data
# from torch_geometric.utils import dense_to_sparse


# class GraphDataloader:
#
#     def __init__(self, dataset, shuffle, eta=None):
#         self.dataset = dataset
#         self.shuffle = shuffle
#         self.edge_indices = self.setup_edges(eta=eta)
#
#     def setup_edges(self, eta=None):
#         edge_indices = []
#         for bag in self.dataset.bags:
#             # If eta not set, create fully connected graph
#             if eta is None:
#                 edge_index = torch.ones((len(bag), len(bag)))
#                 edge_index, _ = dense_to_sparse(edge_index)
#                 edge_index = edge_index.long().contiguous()
#             else:
#                 dist = torch.cdist(bag, bag)
#                 b_dist = torch.zeros_like(dist)
#                 b_dist[dist < eta] = 1
#                 edge_index = (dist < eta).nonzero().t()
#             edge_index = edge_index.long().contiguous()
#             edge_indices.append(edge_index)
#         return edge_indices
#
#     def __iter__(self):
#         self.idx = 0
#         self.order = list(range(len(self.dataset)))
#         if self.shuffle:
#             random.shuffle(self.order)
#         return self
#
#     def __next__(self):
#         if self.idx >= len(self.dataset):
#             raise StopIteration
#         o_idx = self.order[self.idx]
#         edge_index = self.edge_indices[o_idx]
#         instances, target, _ = self.dataset[o_idx]
#         data = Data(x=instances, edge_index=edge_index)
#         self.idx += 1
#         return [data], target.unsqueeze(0)
#
#     def __len__(self):
#         return len(self.dataset)
