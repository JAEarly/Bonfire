from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import torch
from torch.utils.data import Dataset


# TODO Add abstract method to compute the mean.
class MilDataset(Dataset, ABC):

    def __init__(self, bags, targets, instance_targets, bags_metadata):
        super(Dataset, self).__init__()
        self.bags = bags
        self.targets = torch.as_tensor(targets).float()
        self.instance_targets = instance_targets
        self.bags_metadata = bags_metadata

    @classmethod
    @property
    @abstractmethod
    def name(cls):
        pass

    @classmethod
    @property
    @abstractmethod
    def d_in(cls):
        pass

    @classmethod
    @property
    @abstractmethod
    def n_expected_dims(cls):
        pass

    @classmethod
    @property
    @abstractmethod
    def n_classes(cls):
        pass

    @classmethod
    @property
    @abstractmethod
    def metric_clz(cls):
        pass

    @classmethod
    @abstractmethod
    def create_datasets(cls, num_test_bags=None):
        pass

    @classmethod
    @abstractmethod
    def create_complete_dataset(cls):
        pass

    @classmethod
    @abstractmethod
    def get_target_mask(cls, instance_targets, clz):
        pass

    def summarise(self, out_clz_dist=True):
        clz_dist = Counter(np.asarray(self.targets))
        print('- MIL Dataset Summary -')
        print(' {:d} bags'.format(len(self.bags)))

        if out_clz_dist:
            print(' Class Distribution')
            for clz in sorted(clz_dist.keys()):
                print('  {:d}: {:d} ({:.2f}%)'.format(int(clz), clz_dist[clz], clz_dist[clz]/len(self)*100))

        bag_sizes = [len(b) for b in self.bags]
        print(' Bag Sizes')
        print('  Min: {:d}'.format(min(bag_sizes)))
        print('  Avg: {:.1f}'.format(np.mean(bag_sizes)))
        print('  Max: {:d}'.format(max(bag_sizes)))

    def get_clz_weights(self):
        """
        Calculate clz weights for under/over sampling, i.e., 1/size of classes
        """
        clz_ratio = np.bincount(self.targets)
        clz_weights = 1 / torch.tensor(clz_ratio, dtype=torch.float)
        clz_weights /= clz_weights.sum()
        return clz_weights

    def get_sample_weights(self):
        """
        Calculate weight for each sample based on clz weights.
        """
        clz_weights = self.get_clz_weights()
        sample_weights = torch.zeros(len(self.targets))
        for i, t in enumerate(self.targets):
            sample_weights[i] = clz_weights[t.long()]
        return sample_weights

    def calculate_witness_rate(self):
        if self.instance_targets is None:
            raise ValueError('Cannot calculate witness rate without instance targets.')
        # Witness rate for each bag
        wrs = []
        # Iterate through instance targets for each bag
        for bag_instance_targets in self.instance_targets:
            # Get flat version of instance targets
            flat_targets = []
            for target in bag_instance_targets:
                if type(target) is list:
                    flat_targets.extend(target)
                elif type(target) is torch.Tensor:
                    flat_targets.append(target.item())
                else:
                    flat_targets.append(target)
            # Count instance target classes
            c = Counter(flat_targets)
            # Witness rate is the percentage of non-zero instance targets
            wr = (1 - c[0] / len(flat_targets)) * 100
            wrs.append(wr)
        # Average witness rate over all bags and return
        wr = np.mean(wrs)
        return wr

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        bag = self.bags[index]
        target = self.targets[index]
        instance_targets = self.instance_targets[index] if self.instance_targets is not None else None
        return bag, target, instance_targets
