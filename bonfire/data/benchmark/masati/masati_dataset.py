import os
import xml.etree.ElementTree as ET
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition, mark_inset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm

from bonfire.data.mil_dataset import MilDataset


def _setup_loc_df():
    # Check if ship data already exists
    if not os.path.exists(ship_loc_csv_path):
        print('Generating ship location dataframe')
        # Parse data from xml files
        location_data = []
        for label_dir in label_dirs:
            abs_label_dir = RAW_DATA_DIR + "/" + label_dir
            for file in os.listdir(abs_label_dir):
                xml_path = abs_label_dir + "/" + file
                root = ET.parse(xml_path).getroot()
                image_id = file[:-4]
                ship_idx = 0
                for obj in root.findall('object'):
                    truncated = obj.find('truncated').text
                    bbox = obj.find('bndbox')
                    xmin = int(bbox.find('xmin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymin = int(bbox.find('ymin').text)
                    ymax = int(bbox.find('ymax').text)
                    ship_id = '{:}_{:d}'.format(image_id, ship_idx)
                    location_dict = {
                        'image_id': image_id,
                        'xml_path': xml_path,
                        'ship_id': ship_id,
                        'truncated': truncated,
                        'xmin': xmin,
                        'xmax': xmax,
                        'ymin': ymin,
                        'ymax': ymax,
                    }
                    ship_idx += 1
                    location_data.append(location_dict)
        loc_df = pd.DataFrame(location_data)
        loc_df.to_csv(ship_loc_csv_path, index=False)
    else:
        loc_df = pd.read_csv(ship_loc_csv_path)
    return loc_df


def _setup_img_df(loc_df):
    # Check if df csv already exists
    if not os.path.exists(img_data_csv_path):
        print('Generating image label dataframe')
        # Parse data on all the images that we have
        img_data = []
        for img_dir in img_dirs:
            abs_img_dir = RAW_DATA_DIR + "/" + img_dir
            for file in os.listdir(abs_img_dir):
                image_id = file[:-4]
                img_path = abs_img_dir + "/" + file
                img_dict = {
                    'image_id': image_id,
                    'source': img_dir,
                    'image_path': img_path,
                }
                img_data.append(img_dict)
        img_df = pd.DataFrame(img_data)

        # Use the parsed ship location data to count the number of ships per image
        image_ship_counts = []
        for image_id in img_df['image_id']:
            ship_count = {
                'image_id': image_id,
                'ship_count': len(loc_df.loc[loc_df['image_id'] == image_id])
            }
            image_ship_counts.append(ship_count)
        label_df = pd.DataFrame(image_ship_counts)

        # Combine the image data with the label data and save
        img_label_df = pd.merge(img_df, label_df, on='image_id')
        img_label_df.to_csv(img_data_csv_path, index=False)
    else:
        img_label_df = pd.read_csv(img_data_csv_path)
    return img_label_df


RAW_DATA_DIR = 'data/MASATI/raw'
img_data_csv_path = 'data/MASATI/image_data.csv'
ship_loc_csv_path = 'data/MASATI/ship_location.csv'
img_dirs = ['coast', 'coast_ship', 'multi', 'ship', 'water']
label_dirs = ['coast_ship_labels', 'multi_labels', 'ship_labels']

# TODO transform
basic_transform = transforms.Compose([transforms.ToTensor(),
                                      transforms.Normalize((0.3101, 0.3183, 0.2397), (0.1256, 0.1346, 0.1555))])


class MasatiDataset(MilDataset):

    _ship_loc_df = None
    _img_label_df = None

    d_in = 1200
    n_expected_dims = 4  # i x c x h x w
    n_classes = 1

    def __init__(self, bags, targets, bags_metadata):
        super().__init__(bags, targets, None, bags_metadata)
        self.transform = basic_transform

    @staticmethod
    @property
    def ship_loc_df():
        if MasatiDataset._ship_loc_df is None:
            MasatiDataset._ship_loc_df = _setup_loc_df()
        return MasatiDataset._ship_loc_df

    @staticmethod
    @property
    def img_label_df():
        if MasatiDataset._img_label_df is None:
            MasatiDataset._img_label_df = _setup_img_df(MasatiDataset.ship_loc_df)
        return MasatiDataset._img_label_df

    @classmethod
    def create_datasets(cls, random_state=12, grid_size=32, patch_size=28):
        bags, targets, bags_metadata = MasatiDataset.load_masati_bags(grid_size=grid_size, patch_size=patch_size)

        train_bags, test_bags, train_targets, test_targets, train_bags_metadata, test_bags_metadata = \
            train_test_split(bags, targets, bags_metadata, train_size=0.6, random_state=random_state)
        val_bags, test_bags, val_targets, test_targets, val_bags_metadata, test_bags_metadata = \
            train_test_split(test_bags, test_targets, test_bags_metadata, train_size=0.5, random_state=random_state)

        train_dataset = MasatiDataset(train_bags, train_targets, train_bags_metadata)
        val_dataset = MasatiDataset(val_bags, val_targets, val_bags_metadata)
        test_dataset = MasatiDataset(test_bags, test_targets, test_bags_metadata)

        return train_dataset, val_dataset, test_dataset

    def create_complete_dataset(cls):
        raise NotImplementedError

    def __getitem__(self, index):
        instances = self._load_instances(index)
        target = self.targets[index]
        return instances, target

    def _load_instances(self, bag_idx):
        instances = []
        bag = self.bags[bag_idx]
        for patch_path in bag:
            instance = cv2.cvtColor(cv2.imread(patch_path), cv2.COLOR_BGR2RGB)
            if self.transform is not None:
                instance = self.transform(instance)
            instances.append(instance)
        instances = torch.stack(instances)
        return instances

    @staticmethod
    def load_masati_bags(grid_size=32, patch_size=28, ship_count_limit=15):
        patches_df = pd.read_csv("data/MASATI/patch_{:d}_{:d}_data.csv".format(grid_size, patch_size))
        complete_df = pd.merge(patches_df, MasatiDataset.img_label_df, on='image_id')
        bags = [s.split(",") for s in complete_df['patch_paths'].tolist()]
        targets = complete_df['ship_count'].tolist()
        bags_metadata = [{'id': id_} for id_ in complete_df['image_id'].tolist()]
        if ship_count_limit is not None:
            print('Limiting to max {:d} ships'.format(ship_count_limit))
            valid_idxs = np.where(np.asarray(targets) <= ship_count_limit)[0]
            print('Keeping {:d}/{:d} bags'.format(len(valid_idxs), len(targets)))
            bags = [b for idx, b in enumerate(bags) if idx in valid_idxs]
            targets = [t for idx, t in enumerate(targets) if idx in valid_idxs]
            bags_metadata = [m for idx, m in enumerate(bags_metadata) if idx in valid_idxs]
        else:
            print('Using all bags')
        return bags, targets, bags_metadata

    @staticmethod
    def extract_grid_patches(grid_size=32, patch_size=28):
        num_patches = int(512 / grid_size * 512 / grid_size)
        print('{:d} patches per image'.format(num_patches))
        reconstructed_dim = int(num_patches ** 0.5 * patch_size)
        print('{:d} x {:d} effective new size'.format(reconstructed_dim, reconstructed_dim))

        patch_dir = 'data/MASATI/patch_{:d}_{:d}'.format(grid_size, patch_size)
        if not os.path.exists(patch_dir):
            os.makedirs(patch_dir)

        patches_df = MasatiDataset.img_label_df[['image_id']].copy()
        patches_df['patch_paths'] = ""

        for i in tqdm(range(len(MasatiDataset.img_label_df)), desc='Extracting patches'):
            image_id = MasatiDataset.img_label_df['image_id'][i]
            img_path = MasatiDataset.img_label_df['image_path'][i]
            img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

            n_x = int(img.shape[0]/grid_size)
            n_y = int(img.shape[1]/grid_size)

            patch_paths = []
            for i_x in range(n_x):
                for i_y in range(n_y):
                    p_x = i_x * grid_size
                    p_y = i_y * grid_size
                    patch_img = img[p_x:p_x+grid_size, p_y:p_y+grid_size, :]
                    patch_img = cv2.resize(patch_img, (patch_size, patch_size))
                    patch_path = "{:s}/{:s}_{:d}_{:d}.png".format(patch_dir, image_id, i_x, i_y)
                    patch_paths.append(patch_path)
                    cv2.imwrite(patch_path, patch_img)

            patches_df.loc[i, 'patch_paths'] = ",".join(patch_paths)
        patches_df.to_csv("data/MASATI/patch_{:d}_{:d}_data.csv".format(grid_size, patch_size), index=False)

    @classmethod
    def calculate_dataset_normalisation(cls):
        bags, _, _ = cls.load_masati_bags(ship_count_limit=None)
        avgs = []
        transformation = transforms.ToTensor()
        for bag in tqdm(bags, "Calculating dataset norm"):
            for file_name in bag:
                img = cv2.cvtColor(cv2.imread(file_name), cv2.COLOR_BGR2RGB)
                avg = torch.mean(transformation(img), dim=(1, 2))
                avgs.append(avg)
        arrs = torch.stack(avgs)
        print('All data shape:', arrs.shape)
        arrs_mean = torch.mean(arrs, dim=0)
        arrs_std = torch.std(arrs, dim=0)
        print('Dataset mean: {:}'.format(arrs_mean))
        print(' Dataset std: {:}'.format(arrs_std))

    # @classmethod
    # def baseline_performance(cls):
    #     train_dataset, val_dataset, test_dataset = cls.create_datasets()
    #     train_mean_target = train_dataset.targets.mean().item()
    #     train_mean_target = round(train_mean_target)
    #
    #     def performance_for_dataset(pred, dataset):
    #         targets = dataset.targets
    #         preds = torch.ones_like(targets)
    #         preds *= pred
    #         metric = RegressionMetric.calculate_metric(preds, targets, None)
    #         print('MSE Loss: {:.4f}'.format(metric.mse_loss))
    #         print('MAE Loss: {:.4f}'.format(metric.mae_loss))
    #
    #     print('-- Train --')
    #     performance_for_dataset(train_mean_target, train_dataset)
    #     print('-- Val --')
    #     performance_for_dataset(train_mean_target, val_dataset)
    #     print('-- Test --')
    #     performance_for_dataset(train_mean_target, test_dataset)

    @staticmethod
    def plot_ship_count_data():
        counts = MasatiDataset.img_label_df['ship_count']
        counts = np.asarray(counts)
        lower_bound = 0
        mid_split = 5
        upper_bound = 30
        counts = counts[(counts >= lower_bound) * (counts <= upper_bound)]
        density_counts = Counter(counts)

        all_xs = list(range(lower_bound, upper_bound + 1))
        all_densities = [density_counts[x] for x in all_xs]
        zoomed_xs = list(range(mid_split, upper_bound + 1))
        zoomed_densities = [density_counts[x] for x in zoomed_xs]

        fig, axis = plt.subplots(nrows=1, ncols=1, figsize=(5, 3))
        axis.bar(all_xs, all_densities)
        axis.set_xlabel('Count')
        axis.set_ylabel('Density')

        i_axis = plt.axes([0, 0, 1, 1])
        i_axis.bar(zoomed_xs, zoomed_densities)
        i_axis.set_axes_locator(InsetPosition(i_axis, [0.25, 0.25, 0.65, 0.65]))
        mark_inset(axis, i_axis, loc1=3, loc2=4, fc="none", ec='0.5')

        plt.tight_layout()
        fig_path = "data/MASATI/ship_count_histogram.png"
        fig.savefig(fig_path, format='png', dpi=fig.dpi)
        plt.show()
