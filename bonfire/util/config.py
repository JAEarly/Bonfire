import wandb
import yaml


class Config:

    def __init__(self, config_dict, use_wandb):
        self._config_dict = config_dict
        self._use_wandb = use_wandb

    def __getitem__(self, param_name):
        if self._use_wandb:
            return wandb.config[param_name]
        return self._config_dict[param_name]


config: Config = Config({}, False)


def init_config_from_yaml(path, model_name, use_wandb=False):
    stream = open(path, 'r')
    config_dict = {}
    for name, value in yaml.safe_load(stream).items():
        if name == 'best':
            model_best_dict = value
            if model_name in model_best_dict:
                config_dict.update(model_best_dict[model_name])
            else:
                print('Model best parameters for {:s} not found in {:s}'.format(model_name, path))
        elif name == 'range':
            model_range_dict = value
            # Add shared range parameters as they're used by all models
            config_dict.update(model_range_dict['shared'])
            # Add range parameters just for this particular model
            if model_name in model_range_dict:
                config_dict.update(model_range_dict[model_name])
            else:
                print('Model range parameters for {:s} not found in {:s}'.format(model_name, path))
        else:
            config_dict[name] = value
    config._config_dict = config_dict
    config._use_wandb = use_wandb
