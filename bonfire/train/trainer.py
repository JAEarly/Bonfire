import copy
import inspect
import os

import numpy as np
import optuna
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from bonfire.data.benchmark import get_dataset_clz
from bonfire.data.mil_graph_dataset import GraphDataloader
from bonfire.model import models
from bonfire.model.benchmark import get_model_clz
from bonfire.train import metrics, get_default_save_path


# -- UNUSED --
# def info_loss(outputs, targets, mi_scores):
#     prediction_loss = nn.CrossEntropyLoss()(outputs, targets.long())
#     mean_mi_scores = torch.mean(mi_scores, dim=0)
#     mi_global, mi_local, mi_prior = mean_mi_scores[0], mean_mi_scores[1], mean_mi_scores[2]
#     mi_loss = 0.8 * mi_global + 0.1 * mi_local + 0.1 * mi_prior
#     return prediction_loss, mi_loss, mean_mi_scores


def mil_collate_function(batch):
    # Collate function with variable size inputs
    #  Taken from https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/2?u=ptrblck
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.as_tensor(target)
    return [data, target]


def create_trainer_from_names(device, model_name, dataset_name):
    # Parse model and dataset classes
    model_clz = get_model_clz(dataset_name, model_name)
    dataset_clz = get_dataset_clz(dataset_name)

    # Create the trainer
    return create_trainer_from_clzs(device, model_clz, dataset_clz)


def create_trainer_from_clzs(device, model_clz, dataset_clz):
    # Util function for checking if a model clz (m_clz) inherits from a list of base model classes (b_clzs)
    def check_clz_base_in(m_clz, b_clzs):
        return any([base_clz in b_clzs for base_clz in inspect.getmro(m_clz)])

    # Get dataloader based on what the model clz inherits from
    normal_model_clzs = [models.InstanceSpaceNN, models.EmbeddingSpaceNN, models.AttentionNN, models.MiLstm]
    graph_model_clzs = [models.ClusterGNN]
    if check_clz_base_in(model_clz, normal_model_clzs):
        dataloader_func = create_normal_dataloader
    elif check_clz_base_in(model_clz, graph_model_clzs):
        dataloader_func = create_graph_dataloader
    else:
        raise ValueError('No dataloader registered for model class {:}'.format(model_clz))

    # Actually create the trainer
    return Trainer(device, model_clz, dataset_clz, dataloader_func)


def create_normal_dataloader(dataset, shuffle, n_workers):
    # TODO not using batch size
    return DataLoader(dataset, shuffle=shuffle, batch_size=1, num_workers=n_workers)


def create_graph_dataloader(dataset, shuffle, n_workers):
    # TODO batch_size and n_workers for Graph data loader
    return GraphDataloader(dataset, shuffle)


class Trainer:

    def __init__(self, device, model_clz, dataset_clz, dataloader_func, dataset_params=None):
        self.device = device
        self.model_clz = model_clz
        self.dataset_clz = dataset_clz
        self.dataset_params = dataset_params
        self.dataloader_func = dataloader_func

    @property
    def model_name(self):
        return self.model_clz.__name__

    @property
    def dataset_name(self):
        return self.dataset_clz.__name__

    @property
    def metric_clz(self):
        return self.dataset_clz.metric_clz

    @property
    def criterion(self):
        return self.metric_clz.criterion()

    def load_datasets(self, seed=None):
        if self.dataset_params is not None:
            return self.dataset_clz.create_datasets(seed=seed, **self.dataset_params)
        return self.dataset_clz.create_datasets(seed=seed)

    def create_dataloader(self, dataset, shuffle, n_workers):
        return self.dataloader_func(dataset, shuffle, n_workers)

    def create_model(self):
        return self.model_clz(self.device)

    def create_optimizer(self, model):
        lr = self.get_train_param('lr')
        wd = self.get_train_param('wd')
        return torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=wd)

    def get_train_param(self, key):
        return wandb.config[key]

    def train_epoch(self, model, optimizer, criterion, train_dataloader, val_dataloader):
        model.train()
        epoch_train_loss = 0
        # epoch_prediction_loss = 0
        # epoch_mi_loss = 0
        # epoch_mi_sub_losses = torch.zeros(3)
        for data in tqdm(train_dataloader, desc='Epoch Progress', leave=False):
            bags, targets = data[0], data[1].to(self.device)
            optimizer.zero_grad()
            outputs = model(bags)
            # outputs, mi_scores = model(bags)
            loss = criterion(outputs, targets)
            # prediction_loss, mi_loss, mi_sub_losses = criterion(outputs, targets, mi_scores)
            # loss = prediction_loss + mi_loss
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            # epoch_prediction_loss += prediction_loss.item()
            # epoch_mi_loss += mi_loss.item()
            # epoch_mi_sub_losses += mi_sub_losses.detach()

        epoch_train_loss /= len(train_dataloader)
        # epoch_prediction_loss /= len(train_dataloader)

        # epoch_mi_loss /= len(train_dataloader)
        # epoch_mi_sub_losses /= len(train_dataloader)

        epoch_train_metrics = self.metric_clz.from_train_loss(epoch_train_loss)
        # epoch_train_metrics = metrics.eval_model(model, train_dataloader.dataset, criterion, self.metric_clz)
        epoch_val_metrics = None
        if val_dataloader is not None:
            epoch_val_metrics = metrics.eval_model(model, val_dataloader, self.metric_clz)

        return epoch_train_metrics, epoch_val_metrics

    def train_model(self, model, train_dataloader, val_dataloader, trial=None):
        model.to(self.device)
        model.train()

        optimizer = self.create_optimizer(model)

        # Early stopping based on patience
        early_stopped = False
        # Pruned by Optuna when performing hyper param opt
        pruned = False

        n_epochs = self.get_train_param('n_epochs')
        patience = self.get_train_param('patience')
        patience_interval = self.get_train_param('patience_interval')

        train_metrics = []
        val_metrics = []

        best_model = None
        patience_tracker = 0

        if self.metric_clz.optimise_direction == 'maximize':
            best_key_metric = float("-inf")
        elif self.metric_clz.optimise_direction == 'minimize':
            best_key_metric = float("inf")
        else:
            raise ValueError('Invalid optimise direction {:}'.format(self.metric_clz.optimise_direction))

        print('Starting model training')
        if patience is not None:
            print('  Using patience of {:d} with interval {:d} (num patience epochs = {:d})'
                  .format(patience, patience_interval, patience * patience_interval))
        for epoch in range(n_epochs):
            print('Epoch {:d}/{:d}'.format(epoch + 1, n_epochs))
            # Train model for an epoch
            epoch_outputs = self.train_epoch(model, optimizer, self.criterion, train_dataloader,
                                             val_dataloader if epoch % patience_interval == 0 else None)
            epoch_train_metrics, epoch_val_metrics = epoch_outputs

            # Early stopping
            if patience is not None:
                if epoch_val_metrics is not None:
                    new_key_metric = epoch_val_metrics.key_metric()
                    if self.metric_clz.optimise_direction == 'maximize' and new_key_metric > best_key_metric or \
                            self.metric_clz.optimise_direction == 'minimize' and new_key_metric < best_key_metric:
                        best_key_metric = new_key_metric
                        best_model = copy.deepcopy(model)
                        patience_tracker = 0
                    else:
                        patience_tracker += 1
                        if patience_tracker == patience:
                            early_stopped = True

                    # Update wandb tracking
                    wandb.log({'val_loss': epoch_val_metrics.loss,
                               'val_acc': epoch_val_metrics.accuracy},
                              commit=False)

                    # Update optuna tracking
                    if trial is not None:
                        trial.report(epoch_val_metrics.key_metric(), epoch)
                        # Handle pruning based on the intermediate value.
                        if trial.should_prune():
                            pruned = True
            else:
                best_model = copy.deepcopy(model)

            wandb.log({'train_loss': epoch_train_metrics.loss}, commit=True)

            # Update progress
            train_metrics.append(epoch_train_metrics)
            val_metrics.append(epoch_val_metrics)
            print(' Train: {:s}'.format(epoch_train_metrics.short_string_repr()))
            print('   Val: {:s}'.format(epoch_val_metrics.short_string_repr() if epoch_val_metrics else 'None'))

            if early_stopped:
                wandb.summary["finish_case"] = "early stopped"
                wandb.summary["final_epoch"] = epoch
                break

            if pruned:
                wandb.summary["finish_case"] = "pruned"
                wandb.summary["final_epoch"] = epoch
                wandb.finish(quiet=True)
                raise optuna.exceptions.TrialPruned()
        else:
            wandb.summary["finish_case"] = "epoch limit"
            wandb.summary["final_epoch"] = n_epochs - 1

        return best_model, train_metrics, val_metrics

    def train_single(self, seed=5, save_model=True, verbose=True, trial=None):
        train_dataset, val_dataset, test_dataset = self.load_datasets(seed)
        train_dataloader = self.create_dataloader(train_dataset, True, 0)
        val_dataloader = self.create_dataloader(val_dataset, False, 0)
        test_dataloader = self.create_dataloader(test_dataset, False, 0)
        model = self.create_model()
        train_outputs = self.train_model(model, train_dataloader, val_dataloader, trial=trial)
        del model
        best_model, train_metrics, val_metrics = train_outputs
        if hasattr(best_model, 'flatten_parameters'):
            best_model.flatten_parameters()
        train_results, val_results, test_results = metrics.eval_complete(best_model, train_dataloader, val_dataloader,
                                                                         test_dataloader, self.metric_clz,
                                                                         verbose=verbose)
        if save_model:
            path, save_dir = self.get_model_save_path(best_model, None)
            if verbose:
                print('Saving model to {:s}'.format(path))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(best_model.state_dict(), path)

        train_results.add_wandb_summary('train')
        val_results.add_wandb_summary('val')
        test_results.add_wandb_summary('test')

        return best_model, train_results, val_results, test_results

    def train_multiple(self, n_repeats=10, seeds=None):
        if seeds is not None and len(seeds) != n_repeats:
            print('Provided incorrect number of seeds: {:d} given but expected {:d}'.format(len(seeds), n_repeats))

        # Train multiple models
        results = []
        for i in range(n_repeats):
            print('Repeat {:d}/{:d}'.format(i + 1, n_repeats))
            repeat_seed = np.random.randint(low=1, high=1000) if seeds is None else seeds[i]
            print('Seed: {:d}'.format(repeat_seed))
            train_dataset, val_dataset, test_dataset = self.load_datasets(seed=repeat_seed)
            train_dataloader = self.create_train_dataloader(train_dataset, batch_size=1)
            model = self.create_model()
            best_model, _, _, _ = self.train_model(model, train_dataloader, val_dataset)
            del model

            if hasattr(best_model, 'flatten_parameters'):
                best_model.flatten_parameters()
            final_results = metrics.eval_complete(best_model, train_dataset, val_dataset, test_dataset,
                                                  self.metric_clz, verbose=False)
            results.append(final_results)

            # Save model
            path, save_dir = self.get_model_save_path(best_model, i)
            print('Saving model to {:s}'.format(path))
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(best_model.state_dict(), path)

        metrics.output_results(results)

    def get_model_save_path(self, model, repeat):
        path, save_dir, _ = get_default_save_path(self.dataset_name, self.model_name,
                                                  param_save_string=model.get_param_save_string(), repeat=repeat)
        return path, save_dir
