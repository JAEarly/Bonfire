import wandb
from torch import nn

from bonfire.data.benchmark.mnist.mnist_bags import FourMnistBagsDataset
from bonfire.model import mil_aggregators as agg
from bonfire.model import modules as mod


def get_model_clzs():
    return [FourMnistInstanceSpaceNN, FourMnistEmbeddingSpaceNN, FourMnistAttentionNN, FourMnistGNN,
            FourMnistEmbeddingSpaceLSTM]


def get_model_param(key):
    return wandb.config[key]


MNIST_D_ENC = 128
MNIST_DS_ENC_HID = (512,)
MNIST_DS_AGG_HID = (64,)


class MnistEncoder(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        conv1 = mod.ConvBlock(c_in=1, c_out=20, kernel_size=5, stride=1, padding=0)
        conv2 = mod.ConvBlock(c_in=20, c_out=50, kernel_size=5, stride=1, padding=0)
        self.fe = nn.Sequential(conv1, conv2)
        self.fc_stack = mod.FullyConnectedStack(FourMnistBagsDataset.d_in, MNIST_DS_ENC_HID, MNIST_D_ENC,
                                                final_activation_func=None, dropout=dropout)

    def forward(self, instances):
        x = self.fe(instances)
        x = x.view(x.size(0), -1)
        x = self.fc_stack(x)
        return x


class FourMnistInstanceSpaceNN(models.InstanceSpaceNN):

    def __init__(self, device):
        dropout = get_model_param("dropout")
        agg_func = get_model_param("agg_func")
        encoder = MnistEncoder(dropout)
        aggregator = agg.InstanceAggregator(MNIST_D_ENC, MNIST_DS_AGG_HID, FourMnistBagsDataset.n_classes,
                                            dropout, agg_func)
        super().__init__(device, FourMnistBagsDataset.n_classes, FourMnistBagsDataset.n_expected_dims,
                         encoder, aggregator)


class FourMnistEmbeddingSpaceNN(models.EmbeddingSpaceNN):

    def __init__(self, device):
        dropout = get_model_param("dropout")
        agg_func = get_model_param("agg_func")
        encoder = MnistEncoder(dropout)
        aggregator = agg.EmbeddingAggregator(MNIST_D_ENC, MNIST_DS_AGG_HID, FourMnistBagsDataset.n_classes,
                                             dropout, agg_func)
        super().__init__(device, FourMnistBagsDataset.n_classes, FourMnistBagsDataset.n_expected_dims,
                         encoder, aggregator)


class FourMnistAttentionNN(models.AttentionNN):

    def __init__(self, device):
        dropout = get_model_param("dropout")
        d_attn = get_model_param("d_attn")
        encoder = MnistEncoder(dropout)
        aggregator = agg.MultiHeadAttentionAggregator(1, MNIST_D_ENC, MNIST_DS_AGG_HID, d_attn,
                                                      FourMnistBagsDataset.n_classes, dropout)
        super().__init__(device, FourMnistBagsDataset.n_classes, FourMnistBagsDataset.n_expected_dims,
                         encoder, aggregator)


class FourMnistGNN(models.ClusterGNN):

    def __init__(self, device):
        dropout = get_model_param("dropout")
        encoder = MnistEncoder(dropout)
        # TODO should there be an extra param here to tune at least something other than dropout
        exit(0)
        super().__init__(device, FourMnistBagsDataset.n_classes, FourMnistBagsDataset.n_expected_dims, encoder,
                         MNIST_D_ENC, MNIST_D_ENC, (MNIST_D_ENC,), MNIST_DS_AGG_HID, dropout)


class FourMnistEmbeddingSpaceLSTM(models.MiLstm):

    def __init__(self, device):
        dropout = get_model_param("dropout")
        bidirectional = get_model_param("bidirectional")
        encoder = MnistEncoder(dropout)
        aggregator = agg.LstmEmbeddingSpaceAggregator(MNIST_D_ENC, (MNIST_D_ENC,), 1, bidirectional,
                                                      dropout, MNIST_DS_AGG_HID, FourMnistBagsDataset.n_classes)
        super().__init__(device, FourMnistBagsDataset.n_classes, FourMnistBagsDataset.n_expected_dims,
                         encoder, aggregator)
