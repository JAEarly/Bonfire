import os

import torch
import wandb


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_default_save_path(dataset_name, model_name, modifier=None):
    # Save directory
    save_dir = "models/{:s}/{:s}".format(dataset_name, model_name)

    # File name with optional additions
    file_name = "{:s}".format(model_name)
    if modifier is not None:
        file_name += "_" + str(modifier)
    file_name += ".pkl"

    # Full path
    path = "{:s}/{:s}".format(save_dir, file_name)

    # Return all
    return path, save_dir, file_name


def load_model(device, dataset_name, model_clz, modifier=None):
    path, _, _ = get_default_save_path(dataset_name, model_clz.name, modifier=modifier)
    return load_model_from_path(device, model_clz, path)


def load_model_from_path(device, model_clz, path):
    model = model_clz(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def save_model(dataset_name, model, modifier=None, verbose=True):
    path, save_dir, file_name = get_default_save_path(dataset_name, model.name, modifier=modifier)
    if verbose:
        print('Saving model to {:s}'.format(path))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    torch.save(model.state_dict(), path)
    wandb.log_artifact(path, name=file_name, type='model')
