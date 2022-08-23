from bonfire.data.benchmark import get_dataset_clz
from bonfire.model.benchmark import get_model_clz
from bonfire.util.yaml_util import parse_yaml_config, parse_training_config, parse_tuning_config
from .tuner import Tuner


def create_tuner_from_config(device, model_name, dataset_name, study_name, n_trials):
    # Get model and dataset classes
    model_clz = get_model_clz(dataset_name, model_name)
    dataset_clz = get_dataset_clz(dataset_name)

    # Load training and tuning configs
    config = parse_yaml_config(dataset_name)
    training_config = parse_training_config(config['training'], model_name)
    tuning_config = parse_tuning_config(config['tuning'], model_name)

    # Create tuner
    return Tuner(device, model_clz, dataset_clz, study_name, training_config, tuning_config, n_trials)
