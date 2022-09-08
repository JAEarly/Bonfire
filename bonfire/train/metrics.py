from abc import ABC, abstractmethod

import latextable
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import wandb
from sklearn.metrics import accuracy_score, confusion_matrix
from texttable import Texttable
from torch import nn
from tqdm import tqdm


class Metric(ABC):

    @abstractmethod
    def key_metric(self):
        """
        This is the metric that we're going to maximise/minimise when doing hyperparameter tuning
        :return:
        """

    @property
    @abstractmethod
    def optimise_direction(self):
        """
        The direction that we want to optimise when doing hyperparameter tuning
        maximise for classification (accuracy)
        minimise for regression (loss)
        """

    @staticmethod
    @abstractmethod
    def criterion():
        pass

    @staticmethod
    @abstractmethod
    def calculate_metric(probas, targets, labels):
        pass

    @staticmethod
    @abstractmethod
    def from_train_loss(train_loss):
        pass

    @abstractmethod
    def short_string_repr(self):
        pass

    @abstractmethod
    def out(self):
        pass

    @abstractmethod
    def wandb_log(self, dataset_split, commit):
        pass

    @abstractmethod
    def wandb_summary(self, dataset_split):
        pass


class ClassificationMetric(Metric):

    optimise_direction = 'maximize'

    def __init__(self, accuracy, loss, conf_mat):
        self.accuracy = accuracy
        self.loss = loss
        self.conf_mat = conf_mat

    def key_metric(self):
        return self.accuracy

    @staticmethod
    def criterion():
        return lambda outputs, targets: nn.CrossEntropyLoss()(outputs, targets.long())

    @staticmethod
    def calculate_metric(preds, targets, labels):
        _, probas = torch.max(F.softmax(preds, dim=1), dim=1)
        acc = accuracy_score(targets.long(), probas)
        loss = ClassificationMetric.criterion()(preds, targets).item()
        conf_mat = pd.DataFrame(
            confusion_matrix(targets.long(), probas, labels=labels),
            index=pd.Index(labels, name='Actual'),
            columns=pd.Index(labels, name='Predicted')
        )
        return ClassificationMetric(acc, loss, conf_mat)

    @staticmethod
    def from_train_loss(train_loss):
        return ClassificationMetric(None, train_loss, None)

    def short_string_repr(self):
        return "{{Loss: {:.3f}; ".format(self.loss) + \
               ("Acc: {:.3f}}}".format(self.accuracy) if self.accuracy is not None else "Acc: None}")

    def out(self):
        print('Acc: {:.3f}'.format(self.accuracy))
        print('Loss: {:.3f}'.format(self.loss))
        print(self.conf_mat)

    def wandb_log(self, dataset_split, commit):
        log_dict = {}
        if self.loss is not None:
            log_dict['{:s}_loss'.format(dataset_split)] = self.loss
        if self.accuracy is not None:
            log_dict['{:s}_acc'.format(dataset_split)] = self.accuracy
        wandb.log(log_dict, commit=commit)

    def wandb_summary(self, dataset_split):
        wandb.summary["{:s}_acc".format(dataset_split)] = self.accuracy
        wandb.summary["{:s}_loss".format(dataset_split)] = self.loss


class RegressionMetric(Metric):

    optimise_direction = 'minimize'

    def __init__(self, rmse_loss, mae_loss):
        self.rmse_loss = rmse_loss
        self.mae_loss = mae_loss

    def key_metric(self):
        return self.rmse_loss

    @staticmethod
    def criterion():
        return lambda outputs, targets: torch.sqrt(nn.MSELoss()(outputs.squeeze(), targets.squeeze()))

    @staticmethod
    def calculate_metric(preds, targets, labels):
        rmse_loss = RegressionMetric.criterion()(preds, targets).item()
        mae_loss = nn.L1Loss()(preds.squeeze(), targets.squeeze()).item()
        return RegressionMetric(rmse_loss, mae_loss)

    @staticmethod
    def from_train_loss(train_loss):
        return RegressionMetric(train_loss, None)

    def short_string_repr(self):
        return "{{RMSE Loss: {:.3f}; ".format(self.rmse_loss) + \
               ("MAE Loss: {:.3f}}}".format(self.mae_loss) if self.mae_loss is not None else "MAE Loss: None}")

    def out(self):
        print('RMSE Loss: {:.3f}'.format(self.rmse_loss))
        print('MAE Loss: {:.3f}'.format(self.mae_loss))

    def wandb_log(self, dataset_split, commit):
        log_dict = {}
        if self.rmse_loss is not None:
            log_dict['{:s}_rmse'.format(dataset_split)] = self.rmse_loss
        if self.mae_loss is not None:
            log_dict['{:s}_mae'.format(dataset_split)] = self.mae_loss
        wandb.log(log_dict, commit=commit)

    def wandb_summary(self, dataset_split):
        wandb.summary["{:s}_rmse".format(dataset_split)] = self.rmse_loss
        wandb.summary["{:s}_mae".format(dataset_split)] = self.mae_loss


class IoUMetric(Metric):

    optimise_direction = 'maximize'

    def __init__(self, mean_iou, clz_iou):
        self.mean_iou = mean_iou
        self.clz_iou = clz_iou

    def key_metric(self):
        return self.mean_iou

    @staticmethod
    def criterion():
        pass

    @staticmethod
    def calculate_metric(predictions, targets, labels):
        mean_iou, clz_iou, _, _ = IoUMetric.intersection_over_union(predictions, targets, len(labels))
        return IoUMetric(mean_iou, clz_iou)

    @staticmethod
    def intersection_over_union(true_labels, pred_labels, num_classes, eps=1e-6):
        mask = (true_labels >= 0) & (true_labels < num_classes)
        hist = torch.bincount(
            num_classes * true_labels[mask] + pred_labels[mask],
            minlength=num_classes ** 2,
        ).reshape(num_classes, num_classes).float()
        intersection = torch.diag(hist)
        union = hist.sum(dim=0) + hist.sum(dim=1) - intersection + eps
        clz_iou = intersection / (union + eps)
        mean_iou = torch.nanmean(clz_iou)
        return mean_iou, clz_iou, intersection, union

    @staticmethod
    def calculate_from_cumulative(intersections, unions):
        sum_intersection = torch.sum(intersections, dim=0)
        sum_union = torch.sum(unions, dim=0)
        clz_iou = sum_intersection / sum_union
        mean_iou = torch.nanmean(clz_iou)
        return mean_iou, clz_iou

    @staticmethod
    def from_train_loss(train_loss):
        pass

    def short_string_repr(self):
        return "Mean IoU: {:.3f}".format(self.mean_iou)

    def out(self):
        print("Mean IoU: {:.3f}".format(self.mean_iou))

    def wandb_log(self, dataset_split, commit):
        raise NotImplementedError()

    def wandb_summary(self, dataset_split):
        raise NotImplementedError()


# class CountRegressionMetric(RegressionMetric):
#
#     def __init__(self, mse_loss, mae_loss, conf_mat=None):
#         super().__init__(mse_loss, mae_loss)
#         self.conf_mat = conf_mat
#
#     @staticmethod
#     def calculate_metric(preds, targets, labels):
#         regression_metric = RegressionMetric.calculate_metric(preds, targets, labels)
#         max_count = int(max(max(targets), max(preds)))
#         labels = list(range(max_count + 1))
#         conf_mat = pd.DataFrame(
#             confusion_matrix(targets.long(), torch.round(preds), labels=labels),
#             index=pd.Index(labels, name='Actual'),
#             columns=pd.Index(labels, name='Predicted')
#         )
#         return CountRegressionMetric(regression_metric.rmse_loss, regression_metric.mae_loss, conf_mat)
#
#     def out(self):
#         print('MSE Loss: {:.3f}'.format(self.rmse_loss))
#         print('MAE Loss: {:.3f}'.format(self.mae_loss))
#         if self.conf_mat is not None:
#             print(self.conf_mat)


def eval_complete(model, train_dataloader, val_dataloader, test_dataloader,
                  bag_metrics=(), instance_metrics=(), verbose=False):
    train_bag_res, train_inst_res = eval_model(model, train_dataloader, bag_metrics=bag_metrics,
                                               instance_metrics=instance_metrics, verbose=verbose)
    val_bag_res, val_inst_res = eval_model(model, val_dataloader, bag_metrics=bag_metrics,
                                           instance_metrics=instance_metrics, verbose=verbose)
    test_bag_res, test_inst_res = eval_model(model, test_dataloader, bag_metrics=bag_metrics,
                                             instance_metrics=instance_metrics, verbose=verbose)
    return train_bag_res, train_inst_res, val_bag_res, val_inst_res, test_bag_res, test_inst_res


def eval_model(model, dataloader, bag_metrics=(), instance_metrics=(), verbose=False):
    # Iterate through data loader and gather preds and targets
    all_preds = []
    all_targets = []
    all_instance_preds = []
    all_instance_targets = []
    labels = list(range(model.n_classes))
    model.eval()
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Evaluating', leave=False):
            bags, targets, instance_targets = data[0], data[1], data[2]
            bag_pred, instance_pred = model.forward_verbose(bags)
            all_preds.append(bag_pred.cpu())
            all_targets.append(targets.cpu())

            instance_pred = instance_pred[0]
            if instance_pred is not None:
                all_instance_preds.append(instance_pred.squeeze().cpu())
            all_instance_targets.append(instance_targets.squeeze().cpu())

    # Calculate bag results
    bag_results = None
    if bag_metrics:
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)
        bag_results = [bm.calculate_metric(all_preds, all_targets, labels) for bm in bag_metrics]
        if verbose:
            for bag_result in bag_results:
                bag_result.out()

    # Calculate instance results
    instance_results = None
    if instance_metrics:
        all_instance_preds = torch.cat(all_instance_preds)
        all_instance_targets = torch.cat(all_instance_targets)
        instance_results = [im.calculate_metric(all_instance_preds, all_instance_targets, labels)
                            for im in instance_metrics]
        if verbose:
            for instance_result in instance_results:
                instance_result.out()

    return bag_results, instance_results


def output_results(model_names, results_arr, sort=True, latex=False):
    n_models, n_repeats, n_splits = results_arr.shape
    assert n_models == len(model_names)
    assert n_splits == 3

    results_type = type(results_arr[0][0][0])
    if results_type == ClassificationMetric:
        output_classification_results(model_names, results_arr, sort=sort, latex=latex)
    elif issubclass(results_type, RegressionMetric):
        output_regression_results(model_names, results_arr, sort=sort, latex=latex)
    elif results_type == IoUMetric:
        output_iou_results(model_names, results_arr, sort=sort, latex=latex)
    else:
        raise NotImplementedError('No results output for metrics {:}'.format(results_type))


def output_classification_results(model_names, results_arr, sort=True, latex=False):
    n_models, n_repeats, _ = results_arr.shape
    results = np.empty((n_models, 6), dtype=object)
    mean_test_accuracies = []
    for model_idx in range(n_models):
        model_results = results_arr[model_idx]
        expanded_model_results = np.empty((n_repeats, 6), dtype=float)
        for repeat_idx in range(n_repeats):
            train_results, val_results, test_results = model_results[repeat_idx]
            expanded_model_results[repeat_idx, :] = [train_results.loss, train_results.accuracy,
                                                     val_results.loss, val_results.accuracy,
                                                     test_results.loss, test_results.accuracy]
        mean = np.mean(expanded_model_results, axis=0)
        sem = np.std(expanded_model_results, axis=0) / np.sqrt(len(expanded_model_results))
        mean_test_accuracies.append(mean[5])
        for metric_idx in range(6):
            results[model_idx, metric_idx] = '{:.4f} +- {:.4f}'.format(mean[metric_idx], sem[metric_idx])
    model_order = np.argsort(mean_test_accuracies)[::-1] if sort else list(range(len(model_names)))
    rows = [['Model Name', 'Train Loss', 'Train Accuracy', 'Val Loss', 'Val Accuracy', 'Test Loss', 'Test Accuracy']]
    for model_idx in model_order:
        rows.append([model_names[model_idx]] + list(results[model_idx, :]))
    table = Texttable()
    table.set_cols_dtype(['t'] * 7)
    table.set_cols_align(['c'] * 7)
    table.add_rows(rows)
    table.set_max_width(0)
    print(table.draw())
    if latex:
        print(latextable.draw_latex(table))


def output_regression_results(model_names, results_arr, sort=True, latex=False):
    n_models, n_repeats, _ = results_arr.shape
    results = np.empty((n_models, 6), dtype=object)
    mean_test_mae_losses = []
    for model_idx in range(n_models):
        model_results = results_arr[model_idx]
        expanded_model_results = np.empty((n_repeats, 6), dtype=float)
        for repeat_idx in range(n_repeats):
            train_results, val_results, test_results = model_results[repeat_idx]
            expanded_model_results[repeat_idx, :] = [train_results.rmse_loss, train_results.mae_loss,
                                                     val_results.rmse_loss, val_results.mae_loss,
                                                     test_results.rmse_loss, test_results.mae_loss]
        mean = np.mean(expanded_model_results, axis=0)
        sem = np.std(expanded_model_results, axis=0) / np.sqrt(len(expanded_model_results))
        mean_test_mae_losses.append(mean[5])
        for metric_idx in range(6):
            results[model_idx, metric_idx] = '{:.4f} +- {:.4f}'.format(mean[metric_idx], sem[metric_idx])
    model_order = np.argsort(mean_test_mae_losses) if sort else list(range(len(model_names)))
    rows = [['Model Name', 'Train RMSE', 'Train MAE', 'Val RMSE', 'Val MAE', 'Test RMSE', 'Test MAE']]
    for model_idx in model_order:
        rows.append([model_names[model_idx]] + list(results[model_idx, :]))
    table = Texttable()
    table.set_cols_dtype(['t'] * 7)
    table.set_cols_align(['c'] * 7)
    table.add_rows(rows)
    table.set_max_width(0)
    print(table.draw())
    if latex:
        print(latextable.draw_latex(table))


def output_iou_results(model_names, results_arr, sort=True, latex=False):
    n_models, n_repeats, _ = results_arr.shape
    results = np.empty((n_models, 3), dtype=object)
    mean_test_ious = []
    for model_idx in range(n_models):
        model_results = results_arr[model_idx]
        expanded_model_results = np.empty((n_repeats, 3), dtype=float)
        for repeat_idx in range(n_repeats):
            train_results, val_results, test_results = model_results[repeat_idx]
            expanded_model_results[repeat_idx, :] = [train_results.mean_iou,
                                                     val_results.mean_iou,
                                                     test_results.mean_iou]
        mean = np.mean(expanded_model_results, axis=0)
        sem = np.std(expanded_model_results, axis=0) / np.sqrt(len(expanded_model_results))
        mean_test_ious.append(mean[2])
        for metric_idx in range(3):
            results[model_idx, metric_idx] = '{:.4f} +- {:.4f}'.format(mean[metric_idx], sem[metric_idx])
    model_order = np.argsort(mean_test_ious) if sort else list(range(len(model_names)))
    rows = [['Model Name', 'Train IoU', 'Val IoU', 'Test IoU']]
    for model_idx in model_order:
        rows.append([model_names[model_idx]] + list(results[model_idx, :]))
    table = Texttable()
    table.set_cols_dtype(['t'] * 4)
    table.set_cols_align(['c'] * 4)
    table.add_rows(rows)
    table.set_max_width(0)
    print(table.draw())
    if latex:
        print(latextable.draw_latex(table))
