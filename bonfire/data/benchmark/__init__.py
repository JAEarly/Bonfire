from .crc.crc_dataset import CrcDataset
# from .dgr.dgr_dataset import DgrDataset
from .mnist.mnist_bags import FourMnistBagsDataset

dataset_clzs = [CrcDataset, FourMnistBagsDataset]
dataset_names = [d.name for d in dataset_clzs]


def get_dataset_clz(dataset_name):
    for dataset_clz in dataset_clzs:
        if dataset_clz.name == dataset_name:
            return dataset_clz
    raise ValueError('No dataset registered with name {:s}'.format(dataset_name))
