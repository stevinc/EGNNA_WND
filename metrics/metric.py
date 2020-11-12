from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def accuracy(labels, pred):
    """
    Compute the accuracy, given the labels and the predictions for all images
    :param labels: array of labels
    :param pred: array of prediction
    :return: the accuracy value
    """
    return accuracy_score(labels, pred)


def precision(labels, pred):
    """
    Compute the precision, given the labels and the predictions for all images
    The precision is the ratio tp / (tp + fp)
    --When tp + fp == 0, precision returns 0 and raises UndefinedMetricWarning--
    :param labels: array of labels
    :param pred: array of predictions
    :return: the precision value
    """
    return precision_score(labels, pred)


def recall(labels, pred):
    """
    Compute the recall, given the labels and the predictions for all images
    The recall is the ratio tp / (tp + fn)-->ability of find all the positive samples
    --When tp + fp == 0, precision returns 0 and raises UndefinedMetricWarning--
    :param labels: array of labels
    :param pred: array of predictions
    :return: the recall value
    """
    return recall_score(labels, pred)


def f1(labels, pred):
    """
    Compute the F1 score, also known as balanced F-score or F-measure
    The F1 score can be interpreted as a weighted average of the precision and recall,
    where an F1 score reaches its best value at 1 and worst score at 0.
    ---F1 = 2 * (precision * recall) / (precision + recall) ---
    :param labels: array of labels
    :param pred: array of predictions
    :return: the f1 value
    """
    return f1_score(labels, pred)


metrics_def = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1': f1
}

