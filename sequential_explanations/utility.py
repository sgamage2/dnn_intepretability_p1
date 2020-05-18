import logging
from pprint import pformat

import pandas as pd
import numpy as np
import pickle
import time, os, csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from matplotlib.colors import ListedColormap
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve


# A global list to keep figures (to be saved at the end of the program)
figures_list = []


def add_figure_to_save(fig, name=None):
    figures_list.append((fig, name))


def clear_all_figures():
    figures_list.clear()


def save_all_figures(dir):
    for i, (fig, name) in enumerate(figures_list):
        figname = ''
        if name: figname = '_' + name

        filename = dir + '/fig_' + str(i) + figname + '.png'
        fig.savefig(filename)
    logging.info('All figures saved to {}'.format(dir))


def initialize(params):
    results_dir = params['results_dir'] + '/' + params['exp_id'] + '_' + time.strftime("%b%d-%H_%M_%S")
    params['results_dir'] = results_dir

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    setup_logging(results_dir)
    clear_all_figures()

    logging.info('Experiment parameters below')
    logging.info('\n{}'.format(pformat(params)))


def setup_logging(output_dir):
    log_filename = output_dir + '/' + 'run_log.log'

    # Remove any previous log handlers (or the re-init won't work)
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # noinspection PyArgumentList
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(log_filename, 'w+')],
        level=logging.INFO
    )


def load_dataset(filename):
    logging.info("Loading data from file: {}".format(filename))
    data_df = pd.read_csv(filename)
    return data_df


def get_filename_suffix(params):
    fname_suffix = '_win_' + str(params['window_size'])

    return fname_suffix


def print_class_distribution(y, dataset_name):
    unique, counts = np.unique(y, return_counts=True)
    unique_counts = np.asarray((unique, counts)).T
    np.set_printoptions(suppress=True)
    logging.info('Class distribution of dataset: {}\n{}'.format(dataset_name, unique_counts))
    ratio = counts[0] / counts[1]
    logging.info('Dataset: {}, damaged to undamaged ratio = {}'.format(dataset_name, ratio))


def load_prepared_dataset(params, use_catch22=False):
    logging.info('Loading saved datasets')

    fname_suffix = get_filename_suffix(params)
    if use_catch22:
        fname_suffix += '_catch22'

    X_filename = params['data_dir'] + '/X_all' + fname_suffix + '.npy'
    y_filename = params['data_dir'] + '/y_all' + fname_suffix + '.npy'

    X_all = np.load(X_filename)
    y_all = np.load(y_filename)

    print_class_distribution(y_all, 'Full set')

    return X_all, y_all


def plot_training_history(history):
    if history is None or 'loss' not in history.history:
        return

    fig = plt.figure()
    add_figure_to_save(fig, 'training_history')
    plt.title("Training history")

    plt.plot(history.history['loss'], label='training_loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='validation_loss')

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')

    if 'val_f1' in history.history:
        fig = plt.figure()
        add_figure_to_save(fig, 'validation_f1')
        plt.plot(history.history['val_f1'], label='validation_f1_score')

        plt.xlabel("Epoch")
        plt.ylabel("F1-score")
        plt.legend(loc='upper right')


def print_evaluation_report(y_true, y_pred, dataset_name):
    report_str = classification_report(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    pd.set_option('display.float_format', '{:.4f}'.format)
    logging.info('Dataset: {}. Classification report below'.format(dataset_name))

    logging.info('\n{}'.format(report_str))
    logging.info('Overall accuracy (micro avg): {}'.format(accuracy))


def compute_metrics(conf_mat):
    TP = conf_mat[0][0]
    FP = conf_mat[0][1]
    FN = conf_mat[1][0]
    TN = conf_mat[1][1]

    metrics = {}
    metrics['accuracy'] = (TP+TN)/(TP+FP+FN+TN)
    metrics['precision'] = TP / (TP + FP)
    metrics['recall'] = TP / (TP + FN)
    metrics['false_pos_rate'] = FP / (FP + TN)
    metrics['false_neg_rate'] = FN / (TP + FN)
    metrics['f1'] = (2 * metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])

    return metrics


def scale_training_set(X_train):
    scaler_obj = StandardScaler()
    # scaler_obj = MinMaxScaler()

    scaler_obj.fit(X_train)
    scaled_X = scaler_obj.transform(X_train)
    scaled_X_df = pd.DataFrame(scaled_X)

    return scaled_X_df, scaler_obj


def plot_confusion_matrix(y_true, y_pred, title_suffix):
    conf_mat = confusion_matrix(y_true, y_pred)
    n = conf_mat.shape[0]
    row_index = ['True-' + str(n) for n in range(n)]
    col_index = ['Pred-' + str(n) for n in range(n)]
    conf_mat_df = pd.DataFrame(conf_mat, row_index, col_index)

    fig = plt.figure(figsize=(5, 5))
    add_figure_to_save(fig, 'confusion_matrix_' + title_suffix)
    plt.title('Confusion matrix: ' + title_suffix)
    # sns.set(font_scale=1.4)  # for label size
    sns.heatmap(conf_mat_df, annot=True, cbar=False, cmap=ListedColormap(['gray']), annot_kws={"size": 15}, fmt="0")  # font size
    plt.xlabel('Predicted class')
    plt.ylabel('True class')

    return conf_mat_df


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def plot_roc_curve(y_true, y_pred_prob, title_suffix):
    fpr, tpr, threshold = roc_curve(y_true, y_pred_prob)
    idx_thresh = find_nearest(threshold, 0.5)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    add_figure_to_save(fig, 'ROC_' + title_suffix)
    plt.title('ROC curve: ' + title_suffix)
    plt.plot(fpr, tpr, 'b', label='ROC AUC = %0.2f' % roc_auc)
    plt.scatter(fpr[idx_thresh], tpr[idx_thresh], marker='x', s=50, c='red', label='Threshold = 0.5')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.legend(loc='lower right')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    return roc_auc


def plot_multiclass_roc_curves(y_true, y_pred_prob, num_classes, title_suffix):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fig = plt.figure()
    add_figure_to_save(fig, 'ROC_multiclass_' + title_suffix)
    plt.title('ROC curve: ' + title_suffix)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')

    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true[:, i], y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        plt.plot(fpr[i], tpr[i], label='Class {} (AUC = {:.2f})'.format(i, roc_auc[i]))

    plt.legend(loc='lower right')


def plot_precision_recall_curve(y_true, y_pred_prob, title_suffix):
    prec, rec, threshold = precision_recall_curve(y_true, y_pred_prob)
    idx_thresh = find_nearest(threshold, 0.5)
    prec_rec_auc = auc(rec, prec)
    no_skill = len(y_true[y_true == 1]) / len(y_true)

    fig = plt.figure()
    add_figure_to_save(fig, 'prec_recall_curve_' + title_suffix)
    plt.title('Precision-recall curve: ' + title_suffix)
    plt.plot(rec, prec, 'b', label='AUC = %0.2f' % prec_rec_auc)
    plt.scatter(rec[idx_thresh], prec[idx_thresh], marker='x', s=50, c='red', label='Threshold = 0.5')
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.legend(loc='upper right')
    plt.xlim([-0.05, 1.05])
    plt.ylim([0.0, 1.05])
    plt.ylabel('Precision')
    plt.xlabel('Recall (sensitivity)')

    return prec_rec_auc


def save_obj_to_disk(obj, filepath):
    with open(filepath, 'wb') as file:
        pickle.dump(obj, file)


# Following helpers are for parsing input experiment parameters csv file

def convert(val):
    constructors = [int, float, str]
    for c in constructors:
        try:
            return c(val)
        except ValueError:
            pass


def convert_params_to_correct_types(params):
    converted_params = {}

    for key, val in params.items():
        if val == '-':  # Irrelevant config
            continue

        new_val = convert(val)

        if type(new_val) == str:    # If param is a comma separated string, convert it to a list of the elements
            elements = new_val.split(",")
            if len(elements) > 1:
                new_val = [convert(x) for x in elements]

                if new_val[-1] == '':   # One-element string (nothing after the comma)
                    new_val = new_val[:-1]

        converted_params[key] = new_val

    return converted_params


def get_experiments(filename):
    experiments = []

    with open(filename, mode='r') as experiments_file:
        csv_dict_reader = csv.DictReader(filter(lambda row: row[0]!='#', experiments_file))

        for row in csv_dict_reader:
            exp_params = convert_params_to_correct_types(row)
            experiments.append(exp_params)

    logging.info('Read {} experiments from file: {}'.format(len(experiments), filename))

    return experiments

