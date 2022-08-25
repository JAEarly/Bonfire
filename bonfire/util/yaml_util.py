import yaml


def parse_yaml_benchmark_config(dataset_name):
    path = "bonfire/config/{:s}_config.yaml".format(dataset_name)
    parse_yaml_config(path)


def parse_yaml_config(path):
    stream = open(path, 'r')
    config = {}
    for config_group, params in yaml.safe_load(stream).items():
        # Parse all training configs
        if config_group == 'training':
            all_training_configs = {}
            for training_config_name, training_config_params in params.items():
                all_training_configs[training_config_name] = training_config_params
            config[config_group] = all_training_configs
        # Parse all tuning configs
        elif config_group == 'tuning':
            all_tuning_configs = {}
            for tuning_config_name, tuning_config_params in params.items():
                all_tuning_configs[tuning_config_name] = tuning_config_params
            config[config_group] = all_tuning_configs
        else:
            raise ValueError('Unknown config group {:s}'.format(config_group))
    return config


def parse_training_config(all_training_configs, model_name):
    # Start with the default config
    default_config = all_training_configs['default']
    # Override (or add) model specific param values if they exist
    if model_name in all_training_configs:
        return combine_configs(default_config, all_training_configs[model_name])
    return default_config


def parse_tuning_config(all_tuning_configs, model_name):
    # Start with the default config
    default_config = all_tuning_configs['default']
    # Override with model specific param values if they exist
    if model_name in all_tuning_configs:
        return combine_configs(default_config, all_tuning_configs[model_name])
    return default_config


def combine_configs(config_base, config_override):
    config_combined = config_base.copy()
    for param_name, param_value in config_override.items():
        config_combined[param_name] = param_value
    return config_combined
