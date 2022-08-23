import csv
import os
import random

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image
from sklearn.model_selection import StratifiedKFold, train_test_split
from torchvision import transforms

from bonfire.data.mil_dataset import MilDataset
from bonfire.train.metrics import ClassificationMetric

cell_types = ['others', 'inflammatory', 'fibroblast', 'epithelial']
binary_clz_names = ['non-epithelial', 'epithelial']

orig_path = 'data/CRC/orig'
raw_path = 'data/CRC/raw'
csv_path = 'data/CRC/crc_classes.csv'


class Rotate90:

    def __call__(self, x):
        angle = random.choice([0, 90, 190, 270])
        return TF.rotate(x, angle)


basic_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.8035, 0.6499, 0.8348), (0.0858, 0.1079, 0.0731))])

augmentation_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
                                             transforms.RandomVerticalFlip(),
                                             Rotate90(),
                                             transforms.ToTensor(),
                                             transforms.Normalize((0.8035, 0.6499, 0.8348), (0.0858, 0.1079, 0.0731))])


class CrcDataset(MilDataset):

    name = 'crc'
    d_in = 1200
    n_expected_dims = 4  # i x c x h x w
    n_classes = 2
    metric_clz = ClassificationMetric

    def __init__(self, bags, targets, instance_targets, ids, transform=basic_transform):
        # TODO ids as metadata?
        super().__init__(bags, targets, instance_targets, None)
        self.ids = ids
        self.transform = transform

    @classmethod
    def get_dataset_splits(cls, bags, targets, random_state=5):
        # Split using stratified k fold (5 splits)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        splits = skf.split(bags, targets)

        # Split further into train/val/test (60/20/20) as we only have 100 images
        for train_split, test_split in splits:

            # Split train split (currently 80% of data) into 60% and 20% (so 75/25 ratio)
            train_split, val_split = train_test_split(train_split, random_state=random_state, test_size=0.25,
                                                      stratify=targets[train_split])
            # Yield splits
            yield train_split, val_split, test_split

    @classmethod
    def create_datasets(cls, patch_size=27, augment_train=True, random_state=5, verbose=False, num_test_bags=None):
        bags, targets, ids = load_crc_bags(patch_size, verbose=verbose)

        for train_split, val_split, test_split in cls.get_dataset_splits(bags, targets, random_state=random_state):
            # Setup bags, targets, and ids for splits
            train_bags, val_bags, test_bags = [bags[i] for i in train_split],\
                                              [bags[i] for i in val_split],\
                                              [bags[i] for i in test_split]
            train_targets, val_targets, test_targets = targets[train_split], targets[val_split], targets[test_split]
            train_ids, val_ids, test_ids = ids[train_split], ids[val_split], ids[test_split]

            # Setup instance targets for splits
            img_id_to_instance_targets = load_crc_instance_targets(patch_size)
            train_instance_targets = _get_instance_targets_for_bags(train_bags, img_id_to_instance_targets)
            val_instance_targets = _get_instance_targets_for_bags(val_bags, img_id_to_instance_targets)
            test_instance_targets = _get_instance_targets_for_bags(test_bags, img_id_to_instance_targets)

            # Actually create the datasets
            train_dataset = CrcDataset(train_bags, train_targets, train_instance_targets, train_ids,
                                       transform=augmentation_transform if augment_train else basic_transform)
            val_dataset = CrcDataset(val_bags, val_targets, val_instance_targets, val_ids)
            test_dataset = CrcDataset(test_bags, test_targets, test_instance_targets, test_ids)

            # Summarise if requested
            if verbose:
                print('\n-- Train dataset --')
                train_dataset.summarise()
                print('\n-- Val dataset --')
                val_dataset.summarise()
                print('\n-- Test dataset --')
                test_dataset.summarise()

            # Yield three split datasets
            yield train_dataset, val_dataset, test_dataset

    @classmethod
    def create_complete_dataset(cls):
        raise NotImplementedError

    @classmethod
    def get_target_mask(cls, instance_targets, clz):
        mask = []
        for t in instance_targets:
            mask.append(1 if clz in t else (0 if t else None))
        return np.asarray(mask)

    def __getitem__(self, index):
        instances = self._load_instances(index)
        target = self.targets[index]
        instance_targets = self.instance_targets[index]
        return instances, target, instance_targets

    def _load_instances(self, bag_idx):
        instances = []
        bag = self.bags[bag_idx]
        for file_name in bag:
            with open(file_name, 'rb') as f:
                img = Image.open(f)
                instance = img.convert('RGB')
                if self.transform is not None:
                    instance = self.transform(instance)
                instances.append(instance)
        instances = torch.stack(instances)
        return instances


def load_crc_classes():
    binary_clzs = {}
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            image_id, binary_clz, _ = [int(i) for i in row]
            binary_clzs[image_id] = binary_clz
    return binary_clzs


def load_crc_bags(patch_size=27, verbose=False):
    binary_clzs = load_crc_classes()
    if verbose:
        print('Loading CRC data')

    target_dict = binary_clzs
    clz_names = binary_clz_names

    bags = []
    targets = []
    ids = []

    n_r = int(500 / patch_size)
    n_c = int(500 / patch_size)

    for img_id, clz in target_dict.items():
        clz_name = clz_names[clz]
        patch_dir = 'data/CRC/patch_{:d}/{:s}'.format(patch_size, clz_name)

        bag = []
        for r in range(n_r):
            for c in range(n_c):
                file_name = 'img{:d}_{:d}_{:d}.png'.format(img_id, r, c)
                file_path = '{:s}/{:s}'.format(patch_dir, file_name)
                if os.path.exists(file_path):
                    bag.append(file_path)
        if len(bag) == 0:
            if verbose:
                print('Omitting image {:d} as it has zero foreground patches'.format(img_id))
        else:
            bags.append(bag)
            ids.append(img_id)
            targets.append(clz)

    targets = np.asarray(targets)
    ids = np.asarray(ids)

    if verbose:
        print('Loaded {:d} bags'.format(len(bags)))
    return bags, targets, ids


def load_crc_instance_targets(patch_size):
    img_id_to_instance_targets = {}
    for i in range(100):
        img_id = i + 1
        label_csv_path = "data/CRC/patch_{:d}/instance_labels/img{:d}_instance_labels.csv".format(patch_size, img_id)
        bag_instance_targets = {}
        with open(label_csv_path, newline='', mode='r') as f:
            r = csv.reader(f)
            next(r)
            for line in r:
                x = int(line[0])
                y = int(line[1])
                targets = line[2:]
                target_clzs = []
                for target in targets:
                    target_clzs.append(_binary_target_to_id(target))
                target_clzs = list(set(target_clzs))
                bag_instance_targets[(x, y)] = target_clzs
        img_id_to_instance_targets[img_id] = bag_instance_targets
    return img_id_to_instance_targets


def _binary_target_to_id(target):
    if target == 'others' or target == 'inflammatory' or target == 'fibroblast':
        return 0
    if target == 'epithelial':
        return 1
    raise ValueError('Invalid target: {:s}'.format(target))


def _get_instance_targets_for_bags(bags, instance_target_dict):
    all_instance_targets = []
    for bag in bags:
        bag_instance_targets = []
        for file_name in bag:
            info_str = file_name[file_name.rindex('img')+3:-4]
            img_id, x, y = [int(x) for x in info_str.split('_')]
            if (x, y) in instance_target_dict[img_id]:
                instance_targets = instance_target_dict[img_id][(x, y)]
            else:
                instance_targets = []
            bag_instance_targets.append(instance_targets)
        all_instance_targets.append(bag_instance_targets)
    return all_instance_targets


if __name__ == "__main__":
    for _ in CrcDataset.create_datasets(verbose=True):
        exit(0)
