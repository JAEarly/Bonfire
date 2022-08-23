import argparse

from bonfire.data.benchmark import dataset_names
from bonfire.model.benchmark import model_names
from bonfire.train.trainer import create_trainer_from_names
from bonfire.util import get_device
from bonfire.util.yaml_util import parse_yaml_config, parse_training_config

device = get_device()


def parse_args():
    parser = argparse.ArgumentParser(description='Builtin PyTorch MIL training script.')
    parser.add_argument('dataset_name', choices=dataset_names, help='The dataset to use.')
    parser.add_argument('model_name', choices=model_names, help='The model to train.')
    parser.add_argument('-r', '--n_repeats', default=5, type=int, help='The number of models to train (>=1).')
    args = parser.parse_args()
    return args.dataset_name, args.model_name, args.n_repeats


def run_training():
    # Parse args
    dataset_name, model_name, n_repeats = parse_args()

    # Parse wandb config and get training config for this model
    config = parse_yaml_config(dataset_name)
    training_config = parse_training_config(config['training'], model_name)

    # Create trainer
    trainer = create_trainer_from_names(device, model_name, dataset_name)

    # Log
    print('Starting {:s} training'.format(dataset_name))
    print('  Using model {:}'.format(model_name))
    print('  Using device {:}'.format(device))
    print('  Training {:d} models'.format(n_repeats))

    # Start training
    trainer.train_multiple(training_config, n_repeats=n_repeats)


if __name__ == "__main__":
    run_training()
