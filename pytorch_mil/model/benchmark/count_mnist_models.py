from overrides import overrides

from pytorch_mil.data.benchmark.mnist.mnist_bags import COUNTMNIST_N_CLASSES, MNIST_N_EXPECTED_DIMS
from pytorch_mil.model import aggregator as agg
from pytorch_mil.model import models
from pytorch_mil.model.benchmark.four_mnist_models import MnistEncoder


class CountMnistInstanceSpaceNN(models.InstanceSpaceNN):

    def __init__(self, device, d_enc=256, ds_enc_hid=(256,), ds_agg_hid=(128,), dropout=0.05, agg_func_name='mean'):
        encoder = MnistEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.InstanceAggregator(d_enc, ds_agg_hid, COUNTMNIST_N_CLASSES, dropout, agg_func_name)
        super().__init__(device, COUNTMNIST_N_CLASSES, MNIST_N_EXPECTED_DIMS, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 0.001,
            'weight_decay': 0.001,
        }


class CountMnistEmbeddingSpaceNN(models.EmbeddingSpaceNN):

    def __init__(self, device, d_enc=64, ds_enc_hid=(128, 128,), ds_agg_hid=(), dropout=0.005, agg_func_name='mean'):
        encoder = MnistEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.EmbeddingAggregator(d_enc, ds_agg_hid, COUNTMNIST_N_CLASSES, dropout, agg_func_name)
        super().__init__(device, COUNTMNIST_N_CLASSES, MNIST_N_EXPECTED_DIMS, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 0.001,
            'weight_decay': 1e-4,
        }


class CountMnistAttentionNN(models.AttentionNN):

    def __init__(self, device, d_enc=256, ds_enc_hid=(128, 256,), ds_agg_hid=(126, 128,), dropout=0.05, d_attn=128):
        encoder = MnistEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.MultiHeadAttentionAggregator(1, d_enc, ds_agg_hid, d_attn, COUNTMNIST_N_CLASSES, dropout)
        super().__init__(device, COUNTMNIST_N_CLASSES, MNIST_N_EXPECTED_DIMS, encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-5,
            'weight_decay': 0.01,
        }


class CountMnistGNN(models.ClusterGNN):

    def __init__(self, device, d_enc=64, ds_enc_hid=(128, 64), d_gnn=256, ds_gnn_hid=(64, 128), ds_fc_hid=(),
                 dropout=0.1):
        encoder = MnistEncoder(ds_enc_hid, d_enc, dropout)
        super().__init__(device, COUNTMNIST_N_CLASSES, MNIST_N_EXPECTED_DIMS, encoder,
                         d_enc, d_gnn, ds_gnn_hid, ds_fc_hid, dropout)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-6,
            'weight_decay': 1e-4,
        }
