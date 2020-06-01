import numpy as np
import os, math, logging
import matplotlib.pyplot as plt

import utility
from models import lstm_adding_problem
from feature_significance.random_feature_sig import get_random_feature_sig_scores_lstm
from feature_significance.gradient_saliency import get_gradient_saliency_scores
from feature_significance.LIME import get_lime_feature_sig_scores_lstm
from feature_significance.Intergrated_Grad import get_ig_sig_scores
from feature_significance.occlusion import get_occlusion_scores_lstm

from feature_significance.shapley import get_shapley_feature_sig_scores

import pandas as pd


exp_params = {}
exp_params['results_dir'] = 'output'
exp_params['exp_id'] = 'lstm_adding'
exp_params['model_location'] = 'models/output/lstm_adding_good'

# Options: random, gradient, occlusion, lrp, shap, lime, grad_cam, ig, etc.

# exp_params['feature_sig_estimator'] = 'IG'
# exp_params['feature_sig_estimator'] = 'random'
exp_params['feature_sig_estimator'] = 'occlusion'
# exp_params['feature_sig_estimator'] = 'lime'
# exp_params['feature_sig_estimator'] = 'gradient'



def plot_feature_sig(X_sig_scores, X, title_suffix=''):
    num_samples = X_sig_scores.shape[0]
    num_timesteps = X_sig_scores.shape[1]
    num_features = X_sig_scores.shape[2]
    assert num_features == 2   # For now, we are calling plt.bar twice for the two features
    n = math.ceil(num_samples ** 0.5)   # n x n grid

    fig = plt.figure(figsize=(10, 10))
    fig.suptitle('Feature signficance scores: {}'.format(title_suffix))
    utility.add_figure_to_save(fig, 'feature_sig_' + title_suffix)

    width = 0.5
    gap = 1
    step = (width * 2 + gap)
    end = step * num_timesteps - gap
    mid_points = np.arange(0, end, step)  # label locations
    labels = [str(t) for t in range(0, num_timesteps)]

    for j in range(num_samples):
        ax = plt.subplot(n, n, j + 1)
        feat1_sample_sig_scores = X_sig_scores[j][:, 0]
        feat2_sample_sig_scores = X_sig_scores[j][:, 1]
        markers = X[j][:, 1]
        marker_1s = np.where(markers == 1.0)[0]

        plt.bar(mid_points - width/2, feat1_sample_sig_scores, width=width, label='Random No.')
        plt.bar(mid_points + width/2, feat2_sample_sig_scores, width=width, label='Marker')

        # print(markers)
        for t in marker_1s:
            # print(t)
            max_score = max(feat1_sample_sig_scores[t], feat2_sample_sig_scores[t])
            x = mid_points[t]
            plt.annotate('', xy=(x, 0), xytext=(x, max_score/4), arrowprops=dict(facecolor='black', width=0.25, shrink=0.005))

        ax.set_xticks(mid_points)
        ax.set_xticklabels(labels)

        plt.xlabel('Timestep (t)')
        plt.ylabel('Significance score')

    plt.legend()


def plot_feature_sig_rand_samples(X_sig_scores, X, num_samples):
    rand_idx = np.random.choice(X_sig_scores.shape[0], num_samples, replace=False)
    X_rand_sig_scores = X_sig_scores[rand_idx, :]
    X = X[rand_idx, :]
    title_suffix = 'estimator=' + exp_params['feature_sig_estimator']
    plot_feature_sig(X_rand_sig_scores, X, title_suffix)


def plot_feature_sig_average(X_sig_scores, y):
    X_avg_sig_scores = np.mean(X_sig_scores, axis=0)
    title_suffix = 'estimator=' + exp_params['feature_sig_estimator']

    fig = plt.figure(figsize=(7, 5))
    fig.suptitle('Average feature signficance scores: {}'.format(title_suffix))
    utility.add_figure_to_save(fig, 'avg_feature_sig_' + title_suffix)

    num_timesteps = X_sig_scores.shape[1]
    num_features = X_sig_scores.shape[2]
    assert num_features == 2  # For now, we are calling plt.bar twice for the two features

    width = 0.5
    gap = 1
    step = (width * 2 + gap)
    end = step * num_timesteps - gap
    mid_points = np.arange(0, end, step)  # label locations
    labels = [str(t) for t in range(1, num_timesteps+1)]

    feat1_sample_sig_scores = X_avg_sig_scores[:, 0]
    feat2_sample_sig_scores = X_avg_sig_scores[:, 1]

    ax = plt.subplot(1, 1, 1)

    plt.bar(mid_points - width / 2, feat1_sample_sig_scores, width=width, label='Feature-1')
    plt.bar(mid_points + width / 2, feat2_sample_sig_scores, width=width, label='Feature-2')

    ax.set_xticks(mid_points)
    ax.set_xticklabels(labels)
    plt.xlabel('Timestep (t)')
    plt.ylabel('Significance score')
    plt.legend()


def main():
    utility.initialize(exp_params)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # This line disables GPU

    # --------------------------------------
    # Load model and data. Test it works by evaluating the model on data

    model = utility.load_model(exp_params['model_location'])
    X_train = np.load(exp_params['model_location'] + '/X_train.npy')
    y_train = np.load(exp_params['model_location'] + '/y_train.npy')
    X_test = np.load(exp_params['model_location'] + '/X_test.npy')
    y_test = np.load(exp_params['model_location'] + '/y_test.npy')
    X_train = np.load(exp_params['model_location'] + '/X_train.npy')
    y_train = np.load(exp_params['model_location'] + '/y_train.npy')

    lstm_adding_problem.evaluate_model(model, X_train, y_train, "Train set")
    lstm_adding_problem.evaluate_model(model, X_test, y_test, "Test set")   # Check that model and datasets were loaded properly


    # --------------------------------------
    # Get feature significance scores

    sig_estimator = exp_params['feature_sig_estimator']

    logging.info('Running feature significance estimator: {}'.format(sig_estimator))

    if sig_estimator == 'random':
        X_sig_scores = get_random_feature_sig_scores_lstm(X_test)
    elif sig_estimator == 'gradient':
        X_sig_scores = get_gradient_saliency_scores(model.lstm, X_test, -1)
    elif sig_estimator == 'occlusion':
        X_sig_scores = get_occlusion_scores_lstm(model.lstm, X_test, output_layer_idx=-1, mask_full_timestep=True, fill_value=0)
    elif sig_estimator == 'shap':
        # X_sig_scores = get_shapley_feature_sig_scores(model.lstm, X_train, X_test)
        assert False
    elif sig_estimator == 'grad_cam':
        assert False        # Not implemented yet
    elif sig_estimator == 'IG':
        X_sig_scores = get_ig_sig_scores(model.lstm, X_test)
    elif sig_estimator == 'lime':
        feature_names = ['randnum', 'mask']
        X_test = X_test[0:10]
        y_test = y_test[0:10]
        X_sig_scores = get_lime_feature_sig_scores_lstm(model.lstm, X_train, X_test, y_train, feature_names)
    else:
        assert False    # Unknown feature significance method


    logging.info('Plotting feature significance values')

    # --------------------------------------
    # Plot feature significance scores of some examples (class=0 and class=1)
    n = 4
    plot_feature_sig_rand_samples(X_sig_scores, X_test, num_samples=n)

    # Plot the average feature significance scores (average across samples) of each class
    plot_feature_sig_average(X_sig_scores, y_test)


    # --------------------------------------
    # Evaluation metrics for feature significance
    # Call evaluation metrics functions in 'metrics' directory here


    utility.save_all_figures(exp_params['results_dir'])
    plt.show()


if __name__ == '__main__':
    main()
