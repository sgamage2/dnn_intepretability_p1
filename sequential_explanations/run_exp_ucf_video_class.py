# Similar structure to other exp scripts
# eg: run_exp_lstm_toy_problem.py

# But load only part of the dataset (because it's too big)

# Challenge: model has both CNN and LSTM parts. We will likely not be able to simply call the feature sig methods (will need custom versions of current functions)

import numpy as np
import os, math, logging
import matplotlib.pyplot as plt
import utility
from video_utility.data import DataSet


from feature_significance.random_feature_sig import get_random_feature_sig_scores
from feature_significance.gradient_saliency import get_gradient_saliency_scores
from feature_significance.shapley import get_shapley_feature_sig_scores
from feature_significance.Intergrated_Grad import get_ig_sig_scores
from feature_significance.LIME import get_lime_feature_sig_scores_video
from feature_significance.occlusion import get_occlusion_scores

exp_params = {}
exp_params['random_seed'] = 0
exp_params['data_base_path'] = '/home/jiazhi/videoucftest/data'
exp_params['sequences_path'] = '/home/jiazhi/videoucftest/data/sequences111'
exp_params['results_dir'] = 'output'
exp_params['exp_id'] = 'lrcn_ucf_video_classification'
exp_params['model_location'] = 'models/output/lrcn_video_good'
exp_params['image_shape'] = (224, 224, 3)
exp_params['lstm_layer_units'] = [256, 256]
exp_params['lstm_time_steps'] = 40   # No. of frames to classify at one time (LSTM unrolling)
exp_params['output_nodes'] = 70  # No. of classes


# Options: random, gradient, occlusion, lrp, shap, lime, grad_cam, IG, etc.



# exp_params['feature_sig_estimator'] = 'IG'
#exp_params['feature_sig_estimator'] = 'gradient'
#exp_params['feature_sig_estimator'] = 'random'
#exp_params['feature_sig_estimator'] = 'occlusion'
exp_params['feature_sig_estimator'] = 'lime'

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])


def plot_feature_sig_img(ax, x_img_sig_scores, x_img):
    xmin = np.min(x_img_sig_scores)
    xmax = np.max(x_img_sig_scores)
    x_img_sig_scores = (x_img_sig_scores - xmin) / (xmax - xmin)

    im = ax.imshow(x_img, alpha=0.5, cmap=plt.cm.binary)
    hm = ax.imshow(x_img_sig_scores, cmap=plt.cm.jet, alpha=0.5, interpolation='bilinear')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    return im, hm


def plot_feature_sig_video(x_video_sig_scores, x_video):
    title_suffix = 'estimator=' + exp_params['feature_sig_estimator']

    fig = plt.figure(figsize=(10, 8))
    utility.add_figure_to_save(fig, 'feature_sig_' + title_suffix)
    axes = fig.subplots(4, 5)
    axes = axes.flatten()

    time_steps = x_video.shape[0]
    assert time_steps == 40     # For now, we only do the 5 x 8 = 40 grid (downsampled to 4 x 5 = 20 grid)

    for t in range(0, time_steps, 2):
        x_img = x_video[t]
        x_img_sig_scores = x_video_sig_scores[t]

        # Pre-processing for a good visualization
        x_img_sig_scores = rgb2gray(x_img_sig_scores)   # Otherwise the heatmap appears in weird colors
        x_img_sig_scores = x_img_sig_scores[::4, ::4]   # Otherwise the resolution is too high: heatmap has no regions
        x_img = x_img[::4, ::4, :]

        ax_ind = t // 2
        im, hm = plot_feature_sig_img(axes[ax_ind], x_img_sig_scores, x_img)

    fig.colorbar(hm, cax=axes[-1], aspect=40)
    axes[-1].get_yaxis().set_visible(True)


def plot_feature_sig_video_single_frame(x_video_sig_scores, x_video, timestep):
    title_suffix = 'estimator=' + exp_params['feature_sig_estimator']

    fig = plt.figure(figsize=(8, 8))
    utility.add_figure_to_save(fig, 'feature_sig_' + title_suffix)
    ax = fig.subplots()

    x_img_sig_scores = x_video_sig_scores[timestep]
    x_img = x_video[timestep]

    # x_img_sig_scores = x_img_sig_scores[:, :, 0]    # Single-channel (also works)
    x_img_sig_scores = rgb2gray(x_img_sig_scores)

    x_img_sig_scores = x_img_sig_scores[::4, ::4]
    x_img = x_img[::4, ::4, :]

    im, hm = plot_feature_sig_img(ax, x_img_sig_scores, x_img)


def main():
    np.random.seed(exp_params['random_seed'])  # To get same result every time

    utility.initialize(exp_params)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # This line disables GPU (disable for gradient saliency)


    # --------------------------------------
    # Load model and data. Test it works by evaluating the model on data

    dataset = DataSet(seq_length=exp_params['lstm_time_steps'],
                      class_limit=exp_params['output_nodes'],
                      image_shape=exp_params['image_shape'],
                      base_path=exp_params['data_base_path'],
                      sequences_path=exp_params['sequences_path'])


    X_test, y_test = dataset.get_frames_for_sample_set('test', num_samples=1, random_seed=exp_params['random_seed'])

    # X_test = X_test[:, :10, ::10, :]   # Down sample!!!


    logging.info('X_test.shape = {}, y_test.shape = {}'.format(X_test.shape, y_test.shape))
    # X_test has shape: (num_samples, seq_length, width, height, channels=3)

    model = utility.load_model(exp_params['model_location'])


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
        X_sig_scores = get_gradient_saliency_scores(model.lrcn_model, X_test, -2)
    elif sig_estimator == 'occlusion':
        X_digit_sig_scores_list = []
        X_digit_list = []
        y_digit_list = []
        for digit in range(exp_params['num_classes']):  # Get scores for each class separately
            X_digit = X_test[y_test == digit]
            y_digit = np.full((X_digit.shape[0],), digit)
            X_digit_sig_scores = get_occlusion_scores(model.lrcn_model, X_digit, output_layer_idx=-1, output_node=digit,
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
        X_sig_scores = get_ig_sig_scores(model.lrcn_model, X_test)
        print(X_sig_scores.shape)
    elif sig_estimator == 'lime':
        plots = get_lime_feature_sig_scores_video(model, X_test)
    else:
        assert False  # Unknown feature significance method

    logging.info('X_sig_scores.shape = {}'.format(X_sig_scores.shape))

    logging.info('Plotting feature significance values')

    # --------------------------------------
    # Plot feature significance scores of some examples (class=0 and class=1)


    #plot_video(X_test[0])
    #plot_feature_sig_rand_samples(X_sig_scores, X_test)
    # plot_feature_sig_video_single_frame(X_sig_scores[0], X_test[0], timestep=0)
    plot_feature_sig_video(X_sig_scores[0], X_test[0])
    #plot_feature_sig_rand_samples(X_sig_scores, X_test)

    # plot_feature_sig_rand_samples(X_sig_scores, X_test, y_test)


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