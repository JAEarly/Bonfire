from overrides import overrides

from bonfire.data.benchmark.mnist.mnist_bags import CountMnistBagsDataset
from bonfire.model import mil_aggregators as agg
from bonfire.model.benchmark.four_mnist_models import MnistEncoder


def get_model_clzs():
    return [CountMnistInstanceSpaceNN, CountMnistEmbeddingSpaceNN, CountMnistAttentionNN, CountMnistGNN]


class CountMnistInstanceSpaceNN(models.InstanceSpaceNN):

    def __init__(self, device, d_enc=256, ds_enc_hid=(256,), ds_agg_hid=(128,), dropout=0.05, agg_func_name='mean'):
        encoder = MnistEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.InstanceAggregator(d_enc, ds_agg_hid, CountMnistBagsDataset.n_classes, dropout, agg_func_name)
        super().__init__(device, CountMnistBagsDataset.n_classes, CountMnistBagsDataset.n_expected_dims,
                         encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 0.001,
            'weight_decay': 0.001,
        }


class CountMnistEmbeddingSpaceNN(models.EmbeddingSpaceNN):

    def __init__(self, device, d_enc=64, ds_enc_hid=(128, 128,), ds_agg_hid=(), dropout=0.005, agg_func_name='mean'):
        encoder = MnistEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.EmbeddingAggregator(d_enc, ds_agg_hid, CountMnistBagsDataset.n_classes, dropout, agg_func_name)
        super().__init__(device, CountMnistBagsDataset.n_classes, CountMnistBagsDataset.n_expected_dims,
                         encoder, aggregator)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 0.001,
            'weight_decay': 1e-4,
        }


class CountMnistAttentionNN(models.AttentionNN):

    def __init__(self, device, d_enc=256, ds_enc_hid=(128, 256,), ds_agg_hid=(126, 128,), dropout=0.05, d_attn=128):
        encoder = MnistEncoder(ds_enc_hid, d_enc, dropout)
        aggregator = agg.MultiHeadAttentionAggregator(1, d_enc, ds_agg_hid, d_attn,
                                                      CountMnistBagsDataset.n_classes, dropout)
        super().__init__(device, CountMnistBagsDataset.n_classes, CountMnistBagsDataset.n_expected_dims,
                         encoder, aggregator)

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
        super().__init__(device, CountMnistBagsDataset.n_classes, CountMnistBagsDataset.n_expected_dims, encoder,
                         d_enc, d_gnn, ds_gnn_hid, ds_fc_hid, dropout)

    @overrides
    def suggest_train_params(self):
        return {
            'lr': 1e-6,
            'weight_decay': 1e-4,
        }
