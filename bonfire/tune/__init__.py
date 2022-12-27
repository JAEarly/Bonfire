from bonfire.data.benchmark import get_dataset_clz
from bonfire.model.benchmark import get_model_clz
from bonfire.util.yaml_util import parse_yaml_benchmark_config, parse_training_config, parse_tuning_config
from .tuner import Tuner


def create_benchmark_tuner(device, model_name, dataset_name, study_name, n_trials):
    # Get model and dataset classes
    model_clz = get_model_clz(dataset_name, model_name)
    dataset_clz = get_dataset_clz(dataset_name)

    # Load training and tuning configs
    config = parse_yaml_benchmark_config(dataset_name)

    return create_tuner_from_config(device, model_clz, dataset_clz, config, study_name, n_trials)


def create_tuner_from_config(device, model_clz, dataset_clz, config, study_name, n_trials,
                             dataloader_func=None, project_name=None, trainer_clz=None, model_name=None):
    # Load training and tuning configs
    if model_name is None:
        # Fall back on model clz if specific model name not provided
        model_name = model_clz.name
        print('Model name not provided to create tuning config. Instead, using model clz name: {:s}'.format(model_name))
    training_config = parse_training_config(config['training'], model_name)
    tuning_config = parse_tuning_config(config['tuning'], model_name)

    # Create tuner
    return Tuner(device, model_clz, dataset_clz, study_name, training_config, tuning_config, n_trials,
                 dataloader_func=dataloader_func, project_name=project_name, trainer_clz=trainer_clz)
