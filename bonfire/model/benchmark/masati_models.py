import torch
from overrides import overrides
from torch import nn

from bonfire.data.benchmark.masati.masati_dataset import MasatiDataset
from bonfire.model import aggregator as agg
from bonfire.model import models
from bonfire.model import modules as mod


def get_model_clzs():
    return [MasatiEmbeddingSpaceNN, MasatiInstanceSpaceNN]


# TODO I think this encoder is shared across a lot of datasets
class MasatiEncoder(nn.Module):

    def __init__(self, ds_enc_hid, d_enc, dropout):
        super().__init__()
        conv1 = mod.ConvBlock(c_in=3, c_out=36, kernel_size=4, stride=1, padding=0)
        conv2 = mod.ConvBlock(c_in=36, c_out=48, kernel_size=3, stride=1, padding=0)
        self.fe = nn.Sequential(conv1, conv2)
        self.fc_stack = mod.FullyConnectedStack(MasatiDataset.d_in, ds_enc_hid, d_enc,
                                                final_activation_func=None, dropout=dropout)

    def forward(self, instances):
        x = self.fe(instances)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x


def clamp_predictions(bag_predictions):
    # TODO Clamping might mean that the sum of the instances predictions no longer equals the bag prediction
    clamped_bag_predictions = torch.empty_like(bag_predictions)
    for bag_idx in range(len(bag_predictions)):
        clamped_bag_predictions[bag_idx, :] = torch.round(bag_predictions[bag_idx])
    return clamped_bag_predictions


class MasatiEmbeddingSpaceNN(models.EmbeddingSpaceNN):

    def __init__(self, device, d_enc=64, ds_enc_hid=(64,), ds_agg_hid=(64,), dropout=0.25, agg_func_name='mean'):
        encoder = MasatiEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.EmbeddingAggregator(d_enc, ds_agg_hid, MasatiDataset.n_classes, dropout, agg_func_name)
        super().__init__(device, MasatiDataset.n_classes, MasatiDataset.n_expected_dims, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 5e-4,
            'weight_decay': 1e-4,
        }

    @overrides
    def _internal_forward(self, bags, bags_metadata=None):
        # Clamp outputs to between 0 and 1 if not training
        bag_predictions, instance_predictions = super()._internal_forward(bags)
        if not self.training:
            bag_predictions = clamp_predictions(bag_predictions)
        return bag_predictions, instance_predictions


class MasatiInstanceSpaceNN(models.InstanceSpaceNN):

    def __init__(self, device, d_enc=64, ds_enc_hid=(64,), ds_agg_hid=(64,), dropout=0.25, agg_func_name='sum'):
        encoder = MasatiEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.InstanceAggregator(d_enc, ds_agg_hid, MasatiDataset.n_classes, dropout, agg_func_name)
        super().__init__(device, MasatiDataset.n_classes, MasatiDataset.n_expected_dims, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-4,
            'weight_decay': 1e-6,
        }

    @overrides
    def _internal_forward(self, bags, bags_metadata=None):
        # Clamp outputs to between 0 and 1 if not training
        bag_predictions, instance_predictions = super()._internal_forward(bags)
        if not self.training:
            bag_predictions = clamp_predictions(bag_predictions)
        return bag_predictions, instance_predictions
