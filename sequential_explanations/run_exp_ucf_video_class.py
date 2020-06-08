# Similar structure to other exp scripts
# eg: run_exp_lstm_toy_problem.py

# But load only part of the dataset (because it's too big)

# Challenge: model has both CNN and LSTM parts. We will likely not be able to simply call the feature sig methods (will need custom versions of current functions)

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
exp_params['exp_id'] = 'lrcn_ucf_video_classification'
exp_params['model_location'] = 'models/output/lrcn_video_good'


# Options: random, gradient, occlusion, lrp, shap, lime, grad_cam, IG, etc.
#exp_params['feature_sig_estimator'] = 'occlusion'
exp_params['feature_sig_estimator'] = 'IG'

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



def main():
    utility.initialize(exp_params)
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # This line disables GPU

    # --------------------------------------
    # Load model and data. Test it works by evaluating the model on data

    dataset = ann_mnist.get_mnist_dataset()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset

    # XXXXXXXXXXX:  [ToDo Sunanda] Load models.lrcn here

    # XXXXXXXXXXX: [ToDo Jiazhi] Load 3 random videos (img sequences) = X_test
    # Possible steps (see the for loop in extract_features.py)
    # Get 3 random rows from data.data
    # Call frames = data.get_frames_for_sample(video) --> frames = data.rescale_list(frames, 40)
    # Create a list of these frame seqs without calling model.predict(x) --> shape of a single frame seq will be (40, 299, 299)
    # Make that a numpy array --> So for 3 videos, the array X_test will have shape (3, 40, 299, 299)

    # XXXXXXXXXXX: [ToDo Jiazhi] Plot the above 3 videos
    # Create a plot_video(video) function where the input video is an img seq of shape (40, 299, 299)

    # XXXXXXXXXXX: [ToDo Jiazhi] Cretae a new version of random_feature_sig_scores_video(X_test)
    # returns X_sig_scores with the same shape as X_test --> (3, 40, 299, 299)

    # XXXXXXXXXXX: [ToDo Jiazhi] Plot those feature sig values as a heatmap on top of the video plots


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
            y_digit = np.full((X_digit.shape[0],), digit)
            X_digit_sig_scores = get_occlusion_scores(model.ann, X_digit, output_layer_idx=-1, output_node=digit,
                                                      mask_size=20, stride=4, fill_value=0)
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
        print(X_sig_scores)
        print(X_sig_scores.shape)
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