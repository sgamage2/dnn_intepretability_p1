import numpy as np
import os, math
import matplotlib.pyplot as plt

import utility
from models import ann_toy_problem
from feature_significance.random_feature_sig import get_random_feature_sig_scores
from feature_significance.gradient_saliency import get_gradient_saliency_scores


exp_params = {}
exp_params['results_dir'] = 'output'
exp_params['exp_id'] = 'random_sig'
exp_params['model_location'] = 'models/output/ann_toy_May21-11_08_45/ann_toy_model.pickle'
exp_params['X_data_file'] = 'models/output/ann_toy_May21-11_08_45/X_test.npy'
exp_params['y_data_file'] = 'models/output/ann_toy_May21-11_08_45/y_test.npy'

# Options: random, gradient, occlusion, lrp, shap, lime, grad_cam, ig, etc.
exp_params['feature_sig_estimator'] = 'random'


def plot_feature_sig(X_sig_scores, title_suffix=''):
    num_samples = X_sig_scores.shape[0]
    num_features = X_sig_scores.shape[1]
    n = math.ceil(num_samples ** 0.5)   # n x n grid

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Feature signficance scores: {}'.format(title_suffix))
    utility.add_figure_to_save(fig, 'feature_sig_' + title_suffix)

    labels = [str(num) for num in range(1, num_features+1)]

    for j in range(num_samples):
        plt.subplot(n, n, j + 1)
        sample_sig_scores = X_sig_scores[j]
        plt.bar(labels, sample_sig_scores, width=0.5)
        plt.xlabel('Feature no.')
        plt.ylabel('Significance score')


def plot_feature_sig_rand_samples(X_sig_scores, y, num_samples, class_label):
    X_sig_scores = X_sig_scores[y == class_label]
    rand_idx = np.random.choice(X_sig_scores.shape[0], num_samples, replace=False)
    X_rand_sig_scores = X_sig_scores[rand_idx, :]
    title_suffix = 'estimator=' + exp_params['feature_sig_estimator'] + ' - class = ' + str(class_label)
    plot_feature_sig(X_rand_sig_scores, title_suffix)


def plot_feature_sig_average(X_sig_scores, y):
    X_avg_sig_scores_class_0 = np.mean(X_sig_scores[y == 0], axis=0)
    X_avg_sig_scores_class_1 = np.mean(X_sig_scores[y == 1], axis=0)
    title_suffix = 'estimator=' + exp_params['feature_sig_estimator']

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('Average feature signficance scores: {}'.format(title_suffix))
    utility.add_figure_to_save(fig, 'avg_feature_sig_' + title_suffix)

    num_features = X_sig_scores.shape[1]
    labels = [str(num) for num in range(1, num_features + 1)]

    plt.subplot(1, 2, 1)
    plt.bar(labels, X_avg_sig_scores_class_0, width=0.5)
    plt.gca().set_title('class=0')
    plt.xlabel('Feature no.')
    plt.ylabel('Significance score')

    plt.subplot(1, 2, 2)
    plt.bar(labels, X_avg_sig_scores_class_1, width=0.5)
    plt.gca().set_title('class=1')
    plt.xlabel('Feature no.')
    plt.ylabel('Significance score')


def main():
    utility.initialize(exp_params)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # This line disables GPU

    # --------------------------------------
    # Load model and data. Test it works by evaluating the model on data

    model = utility.load_obj_from_disk(exp_params['model_location'])

    X_test = np.load(exp_params['X_data_file'])
    y_test = np.load(exp_params['y_data_file'])

    ann_toy_problem.evaluate_model(model, X_test, y_test, "Test set")   # Check that model and datasets were loaded properly


    # --------------------------------------
    # Get feature significance scores

    sig_estimator = exp_params['feature_sig_estimator']

    if sig_estimator == 'random':
        X_sig_scores = get_random_feature_sig_scores(X_test)
    elif sig_estimator == 'gradient':
        X_sig_scores = get_gradient_saliency_scores(model.ann, X_test, -2)
    elif sig_estimator == 'shap':
        assert False
    elif sig_estimator == 'grad_cam':
        assert False        # Not implemented yet
    else:
        assert False    # Unknown feature significance method


    # --------------------------------------
    # Plot feature significance scores of some examples (class=0 and class=1)
    n = 4
    plot_feature_sig_rand_samples(X_sig_scores, y_test, num_samples=n, class_label=0)
    plot_feature_sig_rand_samples(X_sig_scores, y_test, num_samples=n, class_label=1)

    # Plot the average feature significance scores (average across samples) of each class
    plot_feature_sig_average(X_sig_scores, y_test)


    # --------------------------------------
    # Evaluation metrics for feature significance
    # Call evaluation metrics functions in 'metrics' directory here


    utility.save_all_figures(exp_params['results_dir'])
    plt.show()


if __name__ == '__main__':
    main()
