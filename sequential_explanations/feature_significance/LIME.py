import numpy as np

import lime
import lime.lime_tabular


def get_lime_feature_sig_scores(model, X_train, X_test, y_train, verbose=False):
    #If the feature is numerical, compute the mean and std, and discretize it into quartiles.
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train, mode='classification', training_labels=y_train,
                                                       class_names=[0, 1], discretize_continuous=True, verbose=verbose)

    num_samples = X_test.shape[0]
    num_features = X_train.shape[1]
    X_sig_scores = np.zeros(shape=(num_samples, num_features))

    for j in range(num_samples):
        if verbose:
            print('Running LIME on sample {}/{}'.format(j, num_samples))

        exp = explainer.explain_instance(X_test[0], model.predict, num_features=num_features, labels=(0,))
        scores = exp.as_map()[0]    # A list of score tuples: (feature_num, score)

        for (feature_num, score) in scores:
            # print('{} - {}'.format(feature_num, score))
            X_sig_scores[j][feature_num] = score

    return X_sig_scores


def arrange_rnn_scores(scores_list, num_timesteps, num_features):
    arranged_scores = np.zeros((num_timesteps, num_features))

    for elem in scores_list:
        feature_num = elem[0]
        feature_idx = feature_num // num_timesteps
        timestep = feature_num % num_timesteps
        # print('feature_num = {} --> (timestep, feature_idx) = ({}, {})'.format(feature_num, timestep, feature_idx))
        score = elem[1]
        arranged_scores[timestep, feature_idx] = score

    return arranged_scores


def get_lime_feature_sig_scores_lstm(model, X_train, X_test, y_train, feature_names):
    #If the feature is numerical, compute the mean and std, and discretize it into quartiles.
    explainer = lime.lime_tabular.RecurrentTabularExplainer(X_train, mode='regression', training_labels=y_train,
                                       feature_names=feature_names, discretize_continuous=True, discretizer='decile')

    num_samples = X_test.shape[0]
    num_timesteps = X_train.shape[1]
    num_features = X_train.shape[2]
    assert len(feature_names) == num_features
    num_features_to_explain = num_timesteps * num_features  # All features (at all time steps) must be explained

    X_sig_scores = np.zeros(shape=(num_samples, num_timesteps, num_features))

    for j in range(num_samples):
        exp = explainer.explain_instance(X_test[j], model.predict, num_features=num_features_to_explain)

        scores = exp.as_map()[1]  # For regression, the output are predictions for class 1
        # scores is a list of tuples: (feature_num, score)  where feature_num in range (0, num_features_to_explain)

        # Arrange the scores into proper shape (num_timesteps, num_features)
        arranged_scores = arrange_rnn_scores(scores, num_timesteps, num_features)

        X_sig_scores[j] = arranged_scores

    # print(X_sig_scores)
    # print(X_sig_scores.shape)
    # print(X_sig_scores.ndim)


    return X_sig_scores