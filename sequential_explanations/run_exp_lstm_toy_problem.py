import numpy as np
import os, math
import matplotlib.pyplot as plt

import utility
from models import lstm_adding_problem
from feature_significance.random_feature_sig import get_random_feature_sig_scores_lstm
from feature_significance.gradient_saliency import get_gradient_saliency_scores
from feature_significance.Intergrated_Grad import *
from feature_significance.shapley import *

exp_params = {}
exp_params['results_dir'] = 'output'
exp_params['exp_id'] = 'random_sig'
exp_params['model_location'] = 'models/output/lstm_adding_good'

# Options: random, gradient, occlusion, lrp, shap, lime, grad_cam, ig, etc.
exp_params['feature_sig_estimator'] = 'random'


def plot_feature_sig(X_sig_scores, title_suffix=''):
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
    labels = [str(t) for t in range(1, num_timesteps+1)]

    for j in range(num_samples):
        ax = plt.subplot(n, n, j + 1)
        feat1_sample_sig_scores = X_sig_scores[j][:, 0]
        feat2_sample_sig_scores = X_sig_scores[j][:, 1]

        plt.bar(mid_points - width/2, feat1_sample_sig_scores, width=width, label='Feature-1')
        plt.bar(mid_points + width/2, feat2_sample_sig_scores, width=width, label='Feature-2')

        ax.set_xticks(mid_points)
        ax.set_xticklabels(labels)

        plt.xlabel('Timestep (t)')
        plt.ylabel('Significance score')

    plt.legend()


def plot_feature_sig_rand_samples(X_sig_scores, num_samples):
    rand_idx = np.random.choice(X_sig_scores.shape[0], num_samples, replace=False)
    X_rand_sig_scores = X_sig_scores[rand_idx, :]
    title_suffix = 'estimator=' + exp_params['feature_sig_estimator']
    plot_feature_sig(X_rand_sig_scores, title_suffix)


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

    X_test = np.load(exp_params['model_location'] + '/X_test.npy')
    y_test = np.load(exp_params['model_location'] + '/y_test.npy')
    X_train = np.load(exp_params['model_location'] + '/X_train.npy')
    y_train = np.load(exp_params['model_location'] + '/y_train.npy')

    lstm_adding_problem.evaluate_model(model, X_test, y_test, "Test set")   # Check that model and datasets were loaded properly


    # --------------------------------------
    # Get feature significance scores

    sig_estimator = exp_params['feature_sig_estimator']

    if sig_estimator == 'random':
        X_sig_scores = get_random_feature_sig_scores_lstm(X_test)
    elif sig_estimator == 'gradient':
        X_sig_scores = get_gradient_saliency_scores(model.ann, X_test, -2)
    elif sig_estimator == 'shap':
#       X_sig_scores = get_shapley_feature_sig_scores(model.lstm, X_train, X_test)
        assert False
    elif sig_estimator == 'grad_cam':
        assert False        # Not implemented yet
    elif sig_estimator == 'IG':
        #X_sig_scores = integrated_gradients((model.ann, X_test)
        ig = integrated_gradients(model.ann)
        X_sig_scores = ig.explain(X_test[0], num_steps=100) #Call explain() on the integrated_gradients instance with a sample to explain(scores)
        X_sig_scores = X_sig_scores[:, np.newaxis].T
        for i in range(1,2500):
            scores2 = ig.explain(X_test[i], num_steps=100)
            scores2 = scores2[:, np.newaxis].T
            X_sig_scores = np.concatenate((X_sig_scores, scores2), axis=0)
        print(X_sig_scores)
        print(X_sig_scores.shape)
        print(X_sig_scores.ndim)
    else:
        assert False    # Unknown feature significance method


    # --------------------------------------
    # Plot feature significance scores of some examples (class=0 and class=1)
    n = 4
    plot_feature_sig_rand_samples(X_sig_scores, num_samples=n)

    # Plot the average feature significance scores (average across samples) of each class
    plot_feature_sig_average(X_sig_scores, y_test)


    # --------------------------------------
    # Evaluation metrics for feature significance
    # Call evaluation metrics functions in 'metrics' directory here


    utility.save_all_figures(exp_params['results_dir'])
    plt.show()


if __name__ == '__main__':
    main()
