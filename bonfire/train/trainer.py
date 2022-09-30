import copy
from time import sleep

import numpy as np
import optuna
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm

from bonfire.train import metrics
from bonfire.util import save_model


def mil_collate_function(batch):
    # Collate function with variable size inputs
    #  Taken from https://discuss.pytorch.org/t/how-to-create-a-dataloader-with-variable-size-input/8278/2?u=ptrblck
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.as_tensor(target)
    return [data, target]


class Trainer:

    def __init__(self, device, model_clz, dataset_clz, project_name=None, group_name=None):
        self.device = device
        self.model_clz = model_clz
        self.dataset_clz = dataset_clz
        self.model_name = self.model_clz.name
        self.dataset_name = self.dataset_clz.name
        self.project_name = project_name if project_name is not None else 'Train_{:s}'.format(self.dataset_name)
        self.group_name = group_name if group_name is not None else 'Train_{:s}'.format(self.model_name)

    @property
    def metric_clz(self):
        return self.dataset_clz.metric_clz

    @property
    def criterion(self):
        return self.metric_clz.criterion()

    @staticmethod
    def create_dataloader(dataset, shuffle, n_workers):
        # TODO not using batch size
        return DataLoader(dataset, shuffle=shuffle, batch_size=1, num_workers=n_workers)

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

        for data in tqdm(train_dataloader, desc='Epoch Progress', leave=False):
            bags, targets = data[0], data[1].to(self.device)
            optimizer.zero_grad()
            outputs = model(bags)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()

        # Compute train metrics (from running loss)
        epoch_train_loss /= len(train_dataloader)
        epoch_train_metrics = self.metric_clz.from_train_loss(epoch_train_loss)

        # Compute val metrics
        epoch_val_metrics = None
        if val_dataloader is not None:
            bag_metrics, _ = metrics.eval_model(model, val_dataloader, bag_metrics=(self.metric_clz,))
            epoch_val_metrics = bag_metrics[0]

        return epoch_train_metrics, epoch_val_metrics

    def train_model(self, train_dataloader, val_dataloader, test_dataloader, verbose=True, trial=None):
        model = self.create_model()
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
            sleep(0.1)
            epoch_outputs = self.train_epoch(model, optimizer, self.criterion, train_dataloader,
                                             val_dataloader if epoch % patience_interval == 0 else None)
            sleep(0.1)
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

                    # Update wandb tracking for val
                    epoch_val_metrics.wandb_log('val', commit=False)

                    # Update optuna tracking
                    if trial is not None:
                        trial.report(epoch_val_metrics.key_metric(), epoch)
                        # Handle pruning based on the intermediate value.
                        if trial.should_prune():
                            pruned = True
            else:
                best_model = copy.deepcopy(model)

            # Updated wandb tracking for train
            epoch_train_metrics.wandb_log('train', commit=True)

            # Update progress
            train_metrics.append(epoch_train_metrics)
            val_metrics.append(epoch_val_metrics)
            print(' Train: {:s}'.format(epoch_train_metrics.short_string_repr()))
            print('   Val: {:s}'.format(epoch_val_metrics.short_string_repr() if epoch_val_metrics else 'None'))

            if early_stopped:
                print('Training Finished - Early Stopping')
                wandb.summary["finish_case"] = "early stopped"
                wandb.summary["final_epoch"] = epoch
                break

            if pruned:
                print('Training Finished - Pruned')
                wandb.summary["finish_case"] = "pruned"
                wandb.summary["final_epoch"] = epoch
                wandb.finish(quiet=True)
                raise optuna.exceptions.TrialPruned()
        else:
            print('Training Finished - Epoch Limit')
            wandb.summary["finish_case"] = "epoch limit"
            wandb.summary["final_epoch"] = n_epochs - 1

        # Delete model and only use best model from here on
        del model
        if hasattr(best_model, 'flatten_parameters'):
            best_model.flatten_parameters()

        # Perform final eval and log with wandb
        sleep(0.1)
        results = metrics.eval_complete(best_model, train_dataloader, val_dataloader, test_dataloader,
                                        bag_metrics=(self.metric_clz,), verbose=verbose)
        train_results, _, val_results, _, test_results, _ = results
        train_results = train_results[0]
        val_results = val_results[0]
        test_results = test_results[0]
        train_results.wandb_summary('train')
        val_results.wandb_summary('val')
        test_results.wandb_summary('test')

        return best_model, train_results, val_results, test_results

    def train_single(self, verbose=True, trial=None, random_state=5):
        train_dataset, val_dataset, test_dataset = next(self.dataset_clz.dataset_folder_iter(random_state=random_state))
        train_dataloader = self.create_dataloader(train_dataset, True, 0)
        val_dataloader = self.create_dataloader(val_dataset, False, 0)
        test_dataloader = self.create_dataloader(test_dataset, False, 0)
        return self.train_model(train_dataloader, val_dataloader, test_dataloader, verbose=verbose, trial=trial)

    def train_multiple(self, config, n_repeats=5, verbose=True, random_state=5):
        best_models = []
        results_arr = np.empty((1, n_repeats, 3), dtype=object)
        for fold, datasets in enumerate(self.dataset_clz.dataset_folder_iter(n_repeats, random_state=random_state)):
            train_dataset, val_dataset, test_dataset = datasets
            print('Repeat {:d}/{:d}'.format(fold + 1, n_repeats))

            config['dataset_fold'] = fold
            wandb.init(
                project=self.project_name,
                group=self.group_name,
                config=config,
                reinit=True,
            )
            train_dataloader = self.create_dataloader(train_dataset, True, 0)
            val_dataloader = self.create_dataloader(val_dataset, False, 0)
            test_dataloader = self.create_dataloader(test_dataset, False, 0)
            train_outputs = self.train_model(train_dataloader, val_dataloader, test_dataloader, verbose=verbose)
            model = train_outputs[0]
            repeat_results = train_outputs[1:]
            best_models.append(model)
            results_arr[:, fold] = repeat_results

            # Save model
            save_model(self.dataset_name, model, modifier=fold, verbose=verbose)

        if verbose:
            metrics.output_results([self.model_name], results_arr)

        return best_models, results_arr
