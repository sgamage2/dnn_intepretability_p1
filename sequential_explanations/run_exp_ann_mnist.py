import numpy as np
import os, math, logging
import matplotlib.pyplot as plt
import shap
import utility
from models import ann_mnist

from feature_significance.random_feature_sig import get_random_feature_sig_scores
from feature_significance.gradient_saliency import get_gradient_saliency_scores
from feature_significance.shapley import get_shapley_feature_sig_scores
from feature_significance.Intergrated_Grad import get_ig_sig_scores
from feature_significance.LIME import get_lime_feature_sig_scores
from feature_significance.occlusion import get_occlusion_scores


exp_params = {}
exp_params['results_dir'] = 'output'
exp_params['exp_id'] = 'ann_mnist'
exp_params['model_location'] = 'models/output/ann_mnist_good'
exp_params['img_width'] = 28
exp_params['num_classes'] = 10

# Options: random, gradient, occlusion, lrp, shap, lime, grad_cam, IG, etc.
exp_params['feature_sig_estimator'] = 'occlusion'


def plot_feature_sig(ax, x_sig_scores, x, digit):
    xmin = np.min(x_sig_scores)
    xmax = np.max(x_sig_scores)
    # x_sig_scores = (x_sig_scores - xmin) / (xmax - xmin)
    im = ax.imshow(x, alpha=0.5, cmap=plt.cm.binary)
    hm = ax.imshow(x_sig_scores, cmap=plt.cm.jet, alpha=0.5, interpolation='bilinear')

    return im, hm


def plot_feature_sig_rand_samples(X_sig_scores, X, y):
    title_suffix = 'estimator=' + exp_params['feature_sig_estimator']

    w = exp_params['img_width']
    X_sig_scores = X_sig_scores.reshape(-1, w, w)
    X = X.reshape(-1, w, w)

    fig = plt.figure(figsize=(10, 5))
    utility.add_figure_to_save(fig, 'feature_sig_' + title_suffix)
    axes = fig.subplots(3, 4)
    axes = axes.flatten()

    for i in range(11): # 11 plots
        digit = i % exp_params['num_classes']   # 10 digits
        X_sig_scores_digit = X_sig_scores[y == digit]
        X_digit = X[y == digit]
        rand_idx = np.random.randint(0, X_sig_scores_digit.shape[0])
        X_sig_scores_digit = X_sig_scores_digit[rand_idx]
        X_digit = X_digit[rand_idx]

        im, hm = plot_feature_sig(axes[i], X_sig_scores_digit, X_digit, i)

    fig.colorbar(hm, cax=axes[11], aspect=10)


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


def get_distributed_sample(X, y, num_samples):
    assert num_samples >= exp_params['num_classes']
    X_dist = np.zeros((num_samples, X.shape[1]))
    y_dist = np.zeros((num_samples,))

    for i in range(num_samples):
        digit = i % exp_params['num_classes']  # 10 digits
        X_digit = X[y == digit]
        rand_idx = np.random.randint(0, X_digit.shape[0])
        X_dist[i] = X_digit[rand_idx]
        y_dist[i] = digit

    return X_dist, y_dist


def main():

    utility.initialize(exp_params)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # This line disables GPU
    
    # --------------------------------------
    # Load model and data. Test it works by evaluating the model on data

    dataset = ann_mnist.get_mnist_dataset()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset
    
    model = utility.load_model(exp_params['model_location'])

    ann_mnist.evaluate_model(model, X_test, y_test, "Test set")   # Check that model and datasets were loaded properly
    
    y_train = y_train.argmax(axis=1)  # Integer labels
    y_test = y_test.argmax(axis=1)  # Integer labels

    # Reduce test set size for fast score computation
    X_train = X_test[:len(X_train) // 120]
    y_train = X_test[:len(y_train) // 120]
    X_test = X_test[:len(X_test) // 12]
    y_test = y_test[:len(y_test) // 12]

    # --------------------------------------
    # Get feature significance scores
    
    sig_estimator = exp_params['feature_sig_estimator']
    logging.info('Running feature significance estimator: {}'.format(sig_estimator))

    if sig_estimator == 'random':
        X_sig_scores = get_random_feature_sig_scores(X_test)
    elif sig_estimator == 'gradient':
        X_sig_scores = get_gradient_saliency_scores(model.ann, X_test, -2)
    elif sig_estimator == 'occlusion':
        X_digit_sig_scores_list = []
        X_digit_list = []
        y_digit_list = []
        for digit in range(exp_params['num_classes']):  # Get scores for each class separately
            X_digit = X_test[y_test == digit]
            y_digit = np.full((X_digit.shape[0], ), digit)
            X_digit_sig_scores = get_occlusion_scores(model.ann, X_digit, output_layer_idx=-1, output_node=digit, mask_size=20, stride=4, fill_value=0)
            X_digit_sig_scores_list.append(X_digit_sig_scores)
            X_digit_list.append(X_digit)
            y_digit_list.append(y_digit)
        X_sig_scores = np.concatenate(X_digit_sig_scores_list)
        X_test = np.concatenate(X_digit_list)
        y_test = np.concatenate(y_digit_list)
    elif sig_estimator == 'shap':
        X_sig_scores = get_shapley_feature_sig_scores(model.ann, X_train, X_test)
    elif sig_estimator == 'grad_cam':
        assert False  # Not implemented yet
    elif sig_estimator == 'IG':
        X_sig_scores = get_ig_sig_scores(model.ann, X_test)
    elif sig_estimator == 'lime':
        X_test, y_test = get_distributed_sample(X_test, y_test, 12)
        X_sig_scores = get_lime_feature_sig_scores(model.ann, X_train, X_test, y_train, verbose=True)
    else:
        assert False  # Unknown feature significance method
    
    logging.info('Plotting feature significance values')

    # --------------------------------------
    # Plot feature significance scores of some examples (class=0 and class=1)
    plot_feature_sig_rand_samples(X_sig_scores, X_test, y_test)

    # Plot the distribution of significance scores
    # plot_feature_sig_distribution(X_sig_scores, X_test, y_test)
    #
    # # Plot the average feature significance scores (average across samples) of each class
    # plot_feature_sig_average_abs(X_sig_scores, y_test)
    # plot_feature_sig_average_signed(X_sig_scores, y_test)

    # --------------------------------------
    # Evaluation metrics for feature significance
    # Call evaluation metrics functions in 'metrics' directory here

    utility.save_all_figures(exp_params['results_dir'])
    plt.show()


if __name__ == '__main__':
    main()
