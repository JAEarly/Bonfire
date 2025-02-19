import os
from datetime import datetime

import optuna

TUNE_ROOT_DIR = "out/tune"


def _create_pruner():
    # return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=25, interval_steps=5)
    return optuna.pruners.SuccessiveHalvingPruner(min_early_stopping_rate=2)


def load_study(study_uid, study_dir_path, direction='Maximize'):
    study_path = TUNE_ROOT_DIR + "/" + study_dir_path + "/" + study_uid
    if not os.path.exists(study_path):
        raise FileNotFoundError('No study directory found at path: {:s}'.format(study_path))
    storage_uid = "sqlite:///{}/study.db".format(study_path)
    pruner = _create_pruner()
    return optuna.create_study(study_name=study_uid, storage=storage_uid,
                               pruner=pruner, load_if_exists=True, direction=direction)


def generate_figure(plot_func, study_dir_path, study, auto_open=True, **plot_args):
    study_path = TUNE_ROOT_DIR + "/" + study_dir_path + "/" + study.study_name
    fig = plot_func(study, **plot_args)
    name = plot_func.__name__
    name = name[name.index('_') + 1:]
    fig.write_html("{:s}/{:s}.html".format(study_path, name), auto_open=auto_open)


class TqdmOptunaCallBack:

    def __init__(self, pbar):
        self.pbar = pbar

    def __call__(self, study, trial):
        self.pbar.update()
