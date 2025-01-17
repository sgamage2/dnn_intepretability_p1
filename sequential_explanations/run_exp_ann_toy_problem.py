import numpy as np
import os, math, logging
import matplotlib.pyplot as plt
import shap
import utility
import pandas as pd
from models import ann_toy_problem

from feature_significance.random_feature_sig import get_random_feature_sig_scores
from feature_significance.gradient_saliency import get_gradient_saliency_scores
from feature_significance.Intergrated_Grad import get_ig_sig_scores
from feature_significance.LIME import get_lime_feature_sig_scores
from feature_significance.shapley import get_shapley_feature_sig_scores
from feature_significance.occlusion import get_occlusion_scores

from metrics.occlusion_metrics import get_occlusion_metric_tabular


exp_params = {}
exp_params['results_dir'] = 'output'
exp_params['exp_id'] = 'ann_toy'
exp_params['model_location'] = 'models/output/ann_toy_good'

# Options: random, gradient, occlusion, lrp, shap, lime, grad_cam, ig, etc.



# exp_params['feature_sig_estimator'] = 'random'
# exp_params['feature_sig_estimator'] = 'IG'
# exp_params['feature_sig_estimator'] = 'lime'
# exp_params['feature_sig_estimator'] = 'gradient'
# exp_params['feature_sig_estimator'] = 'occlusion'
exp_params['feature_sig_estimator'] = 'shap'

def plot_feature_sig(X_sig_scores, title_suffix=''):
    num_samples = X_sig_scores.shape[0]
    num_features = X_sig_scores.shape[1]
    n = math.ceil(num_samples ** 0.5)   # n x n grid

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Feature signficance scores: {}'.format(title_suffix))
    utility.add_figure_to_save(fig, 'feature_sig_' + title_suffix)

    x = np.arange(0, num_features)

    for j in range(num_samples):
        ax = plt.subplot(n, n, j + 1)
        sample_sig_scores = X_sig_scores[j]
        plt.bar(x, sample_sig_scores, width=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(x)
        plt.xlabel('Feature no.')
        plt.ylabel('Significance score')


def plot_feature_sig_rand_samples(X_sig_scores, y, num_samples, class_label):
    X_sig_scores = X_sig_scores[y == class_label]
    rand_idx = np.random.choice(X_sig_scores.shape[0], num_samples, replace=True)
    X_rand_sig_scores = X_sig_scores[rand_idx, :]
    title_suffix = 'estimator=' + exp_params['feature_sig_estimator'] + ' - class = ' + str(class_label)
    plot_feature_sig(X_rand_sig_scores, title_suffix)


def plot_feature_sig_distribution(X_sig_scores, X, y):
    # title_suffix = 'estimator=' + exp_params['feature_sig_estimator']
    # fig.suptitle('Feature signficance distribution: '.format(title_suffix))
    # utility.add_figure_to_save(fig, 'feature_sig_dist_' + title_suffix)
    #
    # shap.summary_plot(X_sig_scores, X, show=False)

    X_avg_sig_scores_class_0 = X_sig_scores[y == 0]
    X_avg_sig_scores_class_1 = X_sig_scores[y == 1]
    X_class_0 = X[y == 0]
    X_class_1 = X[y == 1]

    fig = plt.figure(figsize=(2, 1))
    title_suffix = 'estimator=' + exp_params['feature_sig_estimator']
    fig.suptitle('Feature signficance distribution: {}'.format(title_suffix))
    utility.add_figure_to_save(fig, 'feature_sig_dist_' + title_suffix)

    plt.subplot(1, 2, 1)
    plt.gca().set_title('class=0')
    shap.summary_plot(X_avg_sig_scores_class_0, X_class_0, show=False, sort=False, plot_size=(10,10))

    plt.subplot(1, 2, 2)
    plt.gca().set_title('class=1')
    shap.summary_plot(X_avg_sig_scores_class_1, X_class_1, show=False, sort=False, plot_size=(10,10))


def plot_feature_sig_average_abs(X_sig_scores, y):
    # title_suffix = 'estimator=' + exp_params['feature_sig_estimator']
    # fig.suptitle('Feature signficance distribution: '.format(title_suffix))
    # utility.add_figure_to_save(fig, 'feature_sig_dist_' + title_suffix)
    #
    # shap.summary_plot(X_sig_scores, X, show=False)

    X_avg_sig_scores_class_0 = X_sig_scores[y == 0]
    X_avg_sig_scores_class_1 = X_sig_scores[y == 1]
    # X_class_0 = X[y == 0]
    # X_class_1 = X[y == 1]

    fig = plt.figure(figsize=(2, 1))
    title_suffix = 'estimator=' + exp_params['feature_sig_estimator']
    fig.suptitle('Average(|feature significance|): {}'.format(title_suffix))
    utility.add_figure_to_save(fig, 'feature_sig_dist_' + title_suffix)

    plt.subplot(1, 2, 1)
    plt.gca().set_title('class=0')
    shap.summary_plot(X_avg_sig_scores_class_0, plot_type='bar', show=False, sort=False, plot_size=(10,10))

    plt.subplot(1, 2, 2)
    plt.gca().set_title('class=1')
    shap.summary_plot(X_avg_sig_scores_class_1, plot_type='bar', show=False, sort=False, plot_size=(10,10))


def plot_feature_sig_average_signed(X_sig_scores, y):
    X_avg_sig_scores_class_0 = np.mean(X_sig_scores[y == 0], axis=0)
    X_avg_sig_scores_class_1 = np.mean(X_sig_scores[y == 1], axis=0)
    title_suffix = 'estimator=' + exp_params['feature_sig_estimator']

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('Average feature significance (signed): {}'.format(title_suffix))
    utility.add_figure_to_save(fig, 'avg_feature_sig_' + title_suffix)

    num_features = X_sig_scores.shape[1]
    x = np.arange(0, num_features)

    ax = plt.subplot(1, 2, 1)
    plt.bar(x, X_avg_sig_scores_class_0, width=0.5)
    ax.set_title('class=0')
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    plt.xlabel('Feature no.')
    plt.ylabel('Significance score')

    ax = plt.subplot(1, 2, 2)
    plt.bar(x, X_avg_sig_scores_class_1, width=0.5)
    ax.set_title('class=1')
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    plt.xlabel('Feature no.')
    plt.ylabel('Significance score')


def main():

    utility.initialize(exp_params)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # This line disables GPU
    
    # --------------------------------------
    # Load model and data. Test it works by evaluating the model on data
    
    model = utility.load_model(exp_params['model_location'])

    X_test = np.load(exp_params['model_location'] + '/X_test.npy')
    y_test = np.load(exp_params['model_location'] + '/y_test.npy')
    X_train = np.load(exp_params['model_location'] + '/X_train.npy')
    y_train = np.load(exp_params['model_location'] + '/y_train.npy')
    
    ann_toy_problem.evaluate_model(model, X_test, y_test, "Test set")   # Check that model and datasets were loaded properly
    
    
    # --------------------------------------
    # Get feature significance scores
    
    sig_estimator = exp_params['feature_sig_estimator']
    logging.info('Running feature significance estimator: {}'.format(sig_estimator))

    if sig_estimator == 'random':
        X_sig_scores = get_random_feature_sig_scores(X_test)
    elif sig_estimator == 'gradient':
        X_sig_scores = get_gradient_saliency_scores(model.ann, X_test, -2)
    elif sig_estimator == 'occlusion':
        X_sig_scores = get_occlusion_scores(model.ann, X_test, output_layer_idx=-1, mask_size=1, stride=1, fill_value=0)
    elif sig_estimator == 'shap':
        X_sig_scores = get_shapley_feature_sig_scores(model.ann, X_train, X_test)
    elif sig_estimator == 'grad_cam':
        assert False  # Not implemented yet
    elif sig_estimator == 'IG':
        X_sig_scores = get_ig_sig_scores(model.ann, X_test)
    elif sig_estimator == 'lime':
        X_test = X_test[0:10]
        y_test = y_test[0:10]
        X_sig_scores = get_lime_feature_sig_scores(model.ann, X_train, X_test, y_train)
    else:
        assert False    # Unknown feature significance method

    logging.info('Plotting feature significance values')

    # --------------------------------------
    # Plot feature significance scores of some examples (class=0 and class=1)
    n = 4
    plot_feature_sig_rand_samples(X_sig_scores, y_test, num_samples=n, class_label=0)
    plot_feature_sig_rand_samples(X_sig_scores, y_test, num_samples=n, class_label=1)

    # Plot the distribution of significance scores
    plot_feature_sig_distribution(X_sig_scores, X_test, y_test)

    # Plot the average feature significance scores (average across samples) of each class
    plot_feature_sig_average_abs(X_sig_scores, y_test)
    plot_feature_sig_average_signed(X_sig_scores, y_test)

    # --------------------------------------
    # Evaluation metrics for feature significance
    # Call evaluation metrics functions in 'metrics' directory here

    remove_ratios, function_vals, accuracies = get_occlusion_metric_tabular(model.ann, X_test, y_test, X_sig_scores,
                                                                    metric='accuracy', fill_value=0)

    f_auc = utility.get_auc(remove_ratios, function_vals)
    a_auc = utility.get_auc(remove_ratios, accuracies)

    utility.plot_occlusion_curve(remove_ratios, function_vals, f_auc, "avg_function_val")
    utility.plot_occlusion_curve(remove_ratios, accuracies, a_auc, "accuracy")

    logging.info('Occlusion metrics: f_auc = {:.2f}, a_auc = {:.2f}'.format(f_auc, a_auc))

    # --------------------------------------
    # Final actions

    utility.save_all_figures(exp_params['results_dir'])
    plt.show()


if __name__ == '__main__':
    main()
