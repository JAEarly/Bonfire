import os

import optuna
import wandb
from optuna import visualization as viz

from bonfire.util.config_util import combine_configs
from bonfire.train.trainer import Trainer

TUNE_ROOT_DIR = "out/tune"


class Tuner:

    def __init__(self, device, model_clz, dataset_clz, study_name, training_config, tuning_config, n_trials,
                 project_name=None):
        self.device = device
        self.model_clz = model_clz
        self.dataset_clz = dataset_clz
        self.project_name = project_name if project_name is not None else 'Tune_{:s}'.format(self.dataset_clz.name)
        self.study_name = study_name
        self.training_config = training_config
        self.tuning_config = tuning_config
        self.n_trials = n_trials
        self.study = self.create_study()

    @property
    def direction(self):
        return self.dataset_clz.metric_clz.optimise_direction

    def run_study(self):
        self.study.optimize(self, n_trials=self.n_trials)

        # Init a new (empty) run just so we can log the optuna plots
        wandb.init(
            project=self.project_name,
            reinit=True,
            group=self.study_name,
        )
        wandb.log(
            {
                "optuna_optimization_history": viz.plot_optimization_history(self.study),
                "optuna_param_importance": viz.plot_param_importances(self.study),
                "optuna_slice": viz.plot_slice(self.study),
            }
        )

    def __call__(self, trial):
        # First configure params in Optuna
        tuning_params = self.generate_params(trial)

        # Combine selected tuning params with default training params (deals with params that we're not tuning but need)
        config = combine_configs(self.training_config, tuning_params)

        # Also include trial number in wandb config
        config["trial_num"] = trial.number

        # Initialise a wandb run (one to one with optuna trials)
        wandb.init(
            project=self.project_name,
            config=config,
            reinit=True,
            group=self.study_name,
        )

        # Create trainer based on params and actually run training
        trainer = Trainer(self.device, self.model_clz, self.dataset_clz)
        model, _, val_results, _ = trainer.train_single(verbose=False, trial=trial)

        # Get final val key metric (the one that we're optimising for) and finish wandb
        key_metric = val_results.key_metric()
        wandb.summary["key_metric"] = key_metric

        # Finish wandb and return key metric
        wandb.finish(quiet=True)
        return key_metric

    def generate_params(self, trial):
        params = {}
        for param_name, param_range in self.tuning_config.items():
            params[param_name] = trial.suggest_categorical(param_name, param_range)
        return params

    def create_study(self):
        storage_path = "{:s}/{:s}.db".format(TUNE_ROOT_DIR, self.study_name)
        if os.path.exists(storage_path):
            os.remove(storage_path)
        storage = "sqlite:///{:s}".format(storage_path)
        pruner = self.create_pruner()
        return optuna.create_study(direction=self.direction, study_name=self.study_name, storage=storage, pruner=pruner)

    def create_pruner(self):
        # Pruner only activates after 10 trials.
        # Also only evaluates a run for pruning at the first point for early stopping
        return optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=self.training_config['patience'])
