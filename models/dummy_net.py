from sklearn.dummy import DummyClassifier
import numpy as np
from matplotlib import pyplot as plt
import scikitplot as skplt


def dummy_classifier(test_labels, metrics, seed):
    test_size = len(test_labels)
    lab = [t[1] for t in test_labels]
    non_zero_count = np.count_nonzero(lab)
    zero_count = test_size - non_zero_count

    train_values_0 = np.zeros(shape=(zero_count, 2), dtype=int)
    train_values_1 = np.ones(shape=(non_zero_count, 2), dtype=int)
    train_values = np.concatenate((train_values_0, train_values_1), axis=0)
    labels = np.concatenate((np.zeros(shape=zero_count, dtype=int),
                             np.ones(shape=non_zero_count, dtype=int)))

    v = np.column_stack((train_values, labels))
    np.random.seed(seed)
    np.random.shuffle(v)

    v_data = v[:, :2]
    v_labels = v[:, 2]

    # train a dummy classifier to make predictions based on the class values
    new_dummy_classifier = DummyClassifier(strategy="stratified") # , random_state=seed)
    new_dummy_classifier.fit(v_data, v_labels)
    prediction = new_dummy_classifier.predict(v_data)
    prob_pred = new_dummy_classifier.predict_proba(v_data)
    # prob = prob_pred[:, 1]
    metrics_calc = {metric: metrics[metric](v_labels, prediction) for metric in metrics}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_calc.items())
    print("- Train metrics Stratified: " + metrics_string)
    print("---------")

    new_dummy_classifier = DummyClassifier(strategy="most_frequent")  # , random_state=seed)
    new_dummy_classifier.fit(v_data, v_labels)
    prediction = new_dummy_classifier.predict(v_data)
    metrics_calc = {metric: metrics[metric](v_labels, prediction) for metric in metrics}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_calc.items())
    print("- Train metrics Most frequent: " + metrics_string)
    print("---------")
    new_dummy_classifier = DummyClassifier(strategy="uniform")  # , random_state=seed)
    new_dummy_classifier.fit(v_data, v_labels)
    prediction = new_dummy_classifier.predict(v_data)
    metrics_calc = {metric: metrics[metric](v_labels, prediction) for metric in metrics}
    metrics_string = " ; ".join("{}: {:05.3f}".format(k, v) for k, v in metrics_calc.items())
    print("- Train metrics Uniform: " + metrics_string)
    print("---------")

    # plt.figure()
    # skplt.metrics.plot_precision_recall(labels, prob_pred)
    # plt.savefig(params.log_dir + '/precision_recall_curve_dummy_plot.png')dummy2.fit(v_data, v_labels)
    # plt.show()
    # plt.close()


