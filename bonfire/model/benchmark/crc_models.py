import wandb
from overrides import overrides
from torch import nn

from bonfire.data.benchmark.crc.crc_dataset import CrcDataset
from bonfire.model import aggregator as agg
from bonfire.model import models
from bonfire.model import modules as mod


def get_model_clzs():
    return [CrcInstanceSpaceNN, CrcEmbeddingSpaceNN, CrcAttentionNN, CrcGNN, CrcMiLSTM]


def get_model_param(key):
    return wandb.config[key]


CRC_D_ENC = 256
CRC_DS_ENC_HID = (64,)
CRC_DS_AGG_HID = (128,)


class CrcEncoder(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        conv1 = mod.ConvBlock(c_in=3, c_out=36, kernel_size=4, stride=1, padding=0)
        conv2 = mod.ConvBlock(c_in=36, c_out=48, kernel_size=3, stride=1, padding=0)
        self.fe = nn.Sequential(conv1, conv2)
        self.fc_stack = mod.FullyConnectedStack(CrcDataset.d_in, CRC_DS_ENC_HID, CRC_D_ENC,
                                                final_activation_func=None, dropout=dropout)

    def forward(self, instances):
        x = self.fe(instances)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x


class CrcInstanceSpaceNN(models.InstanceSpaceNN):

    def __init__(self, device):
        dropout = get_model_param("dropout")
        agg_func = get_model_param("agg_func")
        encoder = CrcEncoder(dropout)
        aggregator = agg.InstanceAggregator(CRC_D_ENC, CRC_DS_AGG_HID, CrcDataset.n_classes, dropout, agg_func)
        super().__init__(device, CrcDataset.n_classes, CrcDataset.n_expected_dims, encoder, aggregator)


class CrcEmbeddingSpaceNN(models.EmbeddingSpaceNN):

    def __init__(self, device):
        dropout = get_model_param("dropout")
        agg_func = get_model_param("agg_func")
        encoder = CrcEncoder(dropout)
        aggregator = agg.EmbeddingAggregator(CRC_D_ENC, CRC_DS_AGG_HID, CrcDataset.n_classes, dropout, agg_func)
        super().__init__(device, CrcDataset.n_classes, CrcDataset.n_expected_dims, encoder, aggregator)


class CrcAttentionNN(models.AttentionNN):

    def __init__(self, device):
        dropout = get_model_param("dropout")
        d_attn = get_model_param("d_attn")
        encoder = CrcEncoder(dropout)
        aggregator = agg.MultiHeadAttentionAggregator(1, CRC_D_ENC, CRC_DS_AGG_HID, d_attn,
                                                      CrcDataset.n_classes, dropout)
        super().__init__(device, CrcDataset.n_classes, CrcDataset.n_expected_dims, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-6,
        }


class CrcGNN(models.ClusterGNN):

    def __init__(self, device):
        dropout = get_model_param("dropout")
        encoder = CrcEncoder(dropout)
        # TODO should there be an extra param here to tune at least something other than dropout
        exit(0)
        super().__init__(device, CrcDataset.n_classes, CrcDataset.n_expected_dims, encoder,
                         CRC_D_ENC, CRC_D_ENC, (CRC_D_ENC,), CRC_DS_AGG_HID, dropout)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-3,
            'weight_decay': 1e-2,
        }


class CrcMiLSTM(models.MiLstm):

    def __init__(self, device, d_lstm_hid=256, n_lstm_layers=1,
                 bidirectional=False, ds_fc_hid=(), dropout=0.2):
        encoder = CrcEncoder(dropout)
        aggregator = agg.LstmEmbeddingSpaceAggregator(CRC_D_ENC, d_lstm_hid, n_lstm_layers, bidirectional, dropout, ds_fc_hid,
                                                      CrcDataset.n_classes)
        super().__init__(device, CrcDataset.n_classes, CrcDataset.n_expected_dims, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-4,
            'weight_decay': 1e-6,
        }
