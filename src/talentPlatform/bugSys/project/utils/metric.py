from collections.abc import Iterable

import torch
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error
)


class Metric(object):
    """
    Base class of Metric
    Overwrite function: compute_metric, return metric of updated samples
    Overwrite function: update_state, save listed data into pred_record and true_record
    """
    NAME = "metric"

    def __init__(self):
        self.pred_record = []
        self.true_record = []
        self.cur_metric = None

    @property
    def name(self):
        return self.NAME

    def update_state(self, y_pred, y_true):
        raise NotImplementedError

    def compute_metric(self) -> float:
        raise NotImplementedError

    def result(self):
        self.cur_metric = self.compute_metric()
        return self.cur_metric

    def display(self):
        cur_metric = self.result()
        return "Metric {}: {}".format(self.name, cur_metric)

    def clear(self):
        self.pred_record = []
        self.true_record = []
        self.cur_metric = None


class MetricList(Metric):
    NAME = "metric_list"

    def __init__(self, metrics):
        super(MetricList, self).__init__()
        self.metrics = {m.name: m for m in metrics} if isinstance(metrics, Iterable) else {metrics.name: metrics}

    def update_state(self, y_pred, y_true):
        for metric in self.metrics.values():
            metric.update_state(y_pred, y_true)

    def result(self):
        metrics_dict = dict()
        for metric in self.metrics.values():
            metrics_dict[metric.metric_name] = metric.result()
        return metrics_dict

    def display(self):
        return "\t".join(metric.display() for metric in self.metrics.values())

    def clear(self):
        for metric in self.metrics.values():
            metric.clear()


class AccuracyMetric(Metric):
    """
    Compute accuracy of all sample updated in
    """
    NAME = "accuracy"

    def __init__(self, normalize=True, sample_weight=None):
        super(AccuracyMetric, self).__init__()
        self.normalize = normalize
        self.sample_weight = sample_weight

    def update_state(self, y_pred, y_true, threshold: float=0.5):
        y_pred = y_pred.cpu().view(-1).tolist()
        y_true = y_true.cpu().view(-1).tolist()
        self.pred_record += y_pred
        self.true_record += y_true

    def compute_metric(self):
        score = accuracy_score(
            self.true_record,
            self.pred_record,
            normalize=self.normalize,
            sample_weight=self.sample_weight
        )
        return score


class F1Metric(Metric):
    """
    Compute F1 score of all sample updated in
    """
    NAME = "F1"

    def __init__(self, labels=None, pos_label=1, average='binary', sample_weight=None):
        super(F1Metric, self).__init__()
        self.labels = labels
        self.pos_label = pos_label
        self.average = average
        self.sample_weight = sample_weight

    def update_state(self, y_pred, y_true):
        y_pred = y_pred.argmax(dim=-1).cpu().view(-1).tolist()
        y_true = y_true.cpu().view(-1).tolist()
        self.pred_record += y_pred
        self.true_record += y_true

    def compute_metric(self):
        score = f1_score(
            self.true_record,
            self.pred_record,
            labels=self.labels,
            pos_label=self.pos_label,
            average=self.average,
            sample_weight=self.sample_weight
        )
        return score


class AUCMetric(Metric):
    """
    Compute roc_auc score of all sample updated in
    """
    NAME = "auc"

    def __init__(self, average="macro", sample_weight=None, max_fpr=None, multi_class="raise", labels=None):
        super(AUCMetric, self).__init__()
        self.average = average
        self.sample_weight = sample_weight
        self.max_fpr = max_fpr
        self.multi_class = multi_class
        self.labels = labels

    def update_state(self, y_pred, y_true):
        y_pred = y_pred.cpu().view(-1).tolist()
        y_true = y_true.cpu().view(-1).tolist()
        self.pred_record += y_pred
        self.true_record += y_true

    def compute_metric(self):
        score = roc_auc_score(
            self.true_record,
            self.pred_record,
            average=self.average,
            sample_weight=self.sample_weight,
            max_fpr=self.max_fpr,
            multi_class=self.multi_class,
            labels=self.labels
        )
        return score


class MSEMetric(Metric):
    """
    Compute Mean Square Error for regression task
    """
    NAME = "mse"

    def __init__(self, sample_weight=None, multioutput='uniform_average', squared=True):
        super(MSEMetric, self).__init__()
        self.sample_weight = sample_weight
        self.multioutput = multioutput
        self.squared = squared

    def update_state(self, y_pred, y_true):
        y_pred = y_pred.cpu().view(-1).tolist()
        y_true = y_true.cpu().view(-1).tolist()
        self.pred_record += y_pred
        self.true_record += y_true

    def compute_metric(self):
        score = mean_squared_error(
            self.true_record,
            self.pred_record,
            sample_weight=self.sample_weight,
            multioutput=self.multioutput,
            squared=self.squared
        )
        return score


class MAEMetric(Metric):
    """
    Compute Mean Absolute Error for regression task
    """
    NAME = "mae"

    def __init__(self, sample_weight=None, multioutput='uniform_average'):
        super(MAEMetric, self).__init__()
        self.sample_weight = sample_weight
        self.multioutput = multioutput

    def update_state(self, y_pred, y_true):
        y_pred = y_pred.cpu().view(-1).tolist()
        y_true = y_true.cpu().view(-1).tolist()
        self.pred_record += y_pred
        self.true_record += y_true

    def compute_metric(self):
        score = mean_absolute_error(
            self.true_record,
            self.pred_record,
            sample_weight=self.sample_weight,
            multioutput=self.multioutput
        )
        return score


METRIC_DICT = {
    "f1": F1Metric,
    "accuracy": AccuracyMetric,
    "roc_auc": AUCMetric,
    "mse": MSEMetric,
    "mae": MAEMetric
}


def get_metric(name: str, *args, **kwargs) -> Metric:
    """Return specific metric instance by given name.
    """
    return METRIC_DICT.get(name, None)(*args, **kwargs)
