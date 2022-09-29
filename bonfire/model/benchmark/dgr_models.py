import wandb
from overrides import overrides
from torch import nn

from bonfire.data.benchmark.dgr.dgr_dataset import DgrDataset
from bonfire.model import mil_aggregators as agg
from bonfire.model import modules as mod


def get_model_clzs():
    return [DgrInstanceSpaceNN, DgrEmbeddingSpaceNN, DgrAttentionNN]


def get_model_param(key):
    return wandb.config[key]


DGR_D_ENC = 128
DGR_DS_ENC_HID = (512,)
DGR_DS_AGG_HID = (64,)


class DgrEncoder(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        conv1 = mod.ConvBlock(c_in=3, c_out=36, kernel_size=4, stride=1, padding=0)
        conv2 = mod.ConvBlock(c_in=36, c_out=48, kernel_size=3, stride=1, padding=0)
        self.fe = nn.Sequential(conv1, conv2)
        self.fc_stack = mod.FullyConnectedStack(DgrDataset.d_in, DGR_DS_ENC_HID, DGR_D_ENC,
                                                final_activation_func=None, dropout=dropout)

    def forward(self, instances):
        x = self.fe(instances)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x


class DgrInstanceSpaceNN(models.InstanceSpaceNN):

    def __init__(self, device):
        dropout = get_model_param("dropout")
        agg_func = get_model_param("agg_func")
        encoder = DgrEncoder(dropout)
        aggregator = agg.InstanceAggregator(DGR_D_ENC, DGR_DS_AGG_HID, DgrDataset.n_classes, dropout, agg_func)
        super().__init__(device, DgrDataset.n_classes, DgrDataset.n_expected_dims, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 5e-5,
            'weight_decay': 0.01,
        }


class DgrEmbeddingSpaceNN(models.EmbeddingSpaceNN):

    def __init__(self, device):
        dropout = get_model_param("dropout")
        agg_func = get_model_param("agg_func")
        encoder = DgrEncoder(dropout)
        aggregator = agg.EmbeddingAggregator(DGR_D_ENC, DGR_DS_AGG_HID, DgrDataset.n_classes, dropout, agg_func)
        super().__init__(device, DgrDataset.n_classes, DgrDataset.n_expected_dims, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 5e-4,
            'weight_decay': 1e-4,
        }


class DgrAttentionNN(models.AttentionNN):

    def __init__(self, device):
        dropout = get_model_param("dropout")
        d_attn = get_model_param("d_attn")
        encoder = DgrEncoder(dropout)
        aggregator = agg.MultiHeadAttentionAggregator(1, DGR_D_ENC, DGR_DS_AGG_HID, d_attn,
                                                      DgrDataset.n_classes, dropout)
        super().__init__(device, DgrDataset.n_classes, DgrDataset.n_expected_dims, encoder, aggregator)
