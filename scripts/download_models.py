import argparse

import wandb

from bonfire.data.benchmark import dataset_names
from bonfire.model.benchmark import model_names
from bonfire.util import get_device

device = get_device()


def parse_args():
    parser = argparse.ArgumentParser(description='Utility script for downloading models from WandB.')
    parser.add_argument('dataset_name', choices=dataset_names, help='The dataset to use.')
    parser.add_argument('model_names', choices=model_names, nargs='+', help='The models to evaluate.')
    parser.add_argument('-r', '--n_repeats', default=5, type=int, help='The number of models to evaluate (>=1).')
    args = parser.parse_args()
    return args.dataset_name, args.model_names, args.n_repeats


def run_download():
    dataset_name, models, n_repeats = parse_args()

    print('Downloading models for dataset {:s}'.format(dataset_name))
    print('Models: {:}'.format(models))

    for model_idx, model_name in enumerate(models):
        api = wandb.Api()
        for repeat_idx in range(n_repeats):
            print('  Downloading {:s} {:d}'.format(model_name, repeat_idx))
            artifact = api.artifact('Train_{:s}/{:s}_{:d}.pkl:latest'.format(dataset_name, model_name, repeat_idx))
            artifact.file("models/{:s}/{:s}".format(dataset_name, model_name))


if __name__ == "__main__":
    run_download()
