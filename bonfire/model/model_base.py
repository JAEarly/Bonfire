from abc import ABC, abstractmethod

import torch
from torch import nn


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


class ConvBlock(nn.Module):

    def __init__(self, c_in, c_out, kernel_size, stride, padding):
        super().__init__()
        conv = nn.Conv2d(c_in, c_out, kernel_size=kernel_size, stride=stride, padding=padding)
        relu = nn.ReLU()
        pool = nn.MaxPool2d(kernel_size=2)
        self.block = nn.Sequential(conv, relu, pool)

    def forward(self, x):
        return self.block(x)


class FullyConnectedBlock(nn.Module):

    def __init__(self, d_in, d_out, activation_func=nn.ReLU(), dropout=0):
        super().__init__()
        layers = [nn.Linear(d_in, d_out)]
        if activation_func is not None:
            layers.append(activation_func)
        if dropout != 0:
            layers.append(nn.Dropout(p=dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class FullyConnectedStack(nn.Module):

    def __init__(self, d_in, ds_hid, d_out, activation_func=nn.ReLU(), final_activation_func=nn.ReLU(), dropout=0):
        super().__init__()
        self.d_in = d_in
        self.ds_hid = ds_hid
        self.d_out = d_out
        self.n_blocks = len(ds_hid) + 1
        blocks = []
        for i in range(self.n_blocks):
            in_size = d_in if i == 0 else ds_hid[i - 1]
            out_size = d_out if i == self.n_blocks - 1 else ds_hid[i]
            block_activation = final_activation_func if i == self.n_blocks - 1 else activation_func
            block_dropout = 0 if i == self.n_blocks - 1 else dropout
            blocks.append(FullyConnectedBlock(in_size, out_size,
                                              activation_func=block_activation,
                                              dropout=block_dropout))
        self.stack = nn.Sequential(*blocks)

    def forward(self, x):
        return self.stack(x)


def get_simple_aggregation_function(agg_func_name):
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
