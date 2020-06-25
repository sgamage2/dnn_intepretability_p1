import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error


def get_occlusion_metric(model, X, y, X_sig_scores, metric='accuracy', fill_value=0):
    num_samples, num_features = X.shape[0], X.shape[1]

    y_orig_func_vals = model.predict(X)
    y_output_node = y_orig_func_vals.argmax(axis=1)    # Predicted original classes

    # Column indices of features sorted by descending order of significance
    sorted_idx = np.argsort(X_sig_scores, axis=1)[:, ::-1]

    remove_ratio = np.arange(0, 1.0001, 0.1)
    occluded_avg_func_vals = []
    occluded_metrics_list = []

    for r in remove_ratio:
        num_features_removed = int(num_features * r)
        # num_features_kept = num_features - num_features_removed
        remove_idx = sorted_idx[:, :num_features_removed]

        X_occluded = np.array(X, copy=True)

        for i in range(num_samples):    # Destroy the most significant r% features
            X_occluded[i, remove_idx[i]] = fill_value

        y_occluded_prob = model.predict(X_occluded)  # This has multiple probs for the multiclass case
        # Pick the new prediction (probability) for original classes
        y_occluded_prob_output_node = y_occluded_prob[np.arange(num_samples), y_output_node].reshape(-1,)

        avg_y_occluded = np.mean(y_occluded_prob_output_node)
        occluded_avg_func_vals.append(avg_y_occluded)

        occluded_metric = -1
        if metric == 'accuracy':
            y_occluded_classes = model.predict_classes(X_occluded)
            occluded_metric = accuracy_score(y, y_occluded_classes)
        elif metric == 'mse':
            y_occluded_classes = model.predict(X_occluded)
            occluded_metric = mean_squared_error(y, y_occluded_classes)

        occluded_metrics_list.append(occluded_metric)

    return remove_ratio, np.array(occluded_avg_func_vals), np.array(occluded_metrics_list)


# Thin wrapper class to hold a keras RNN model and do the reshaping before calls
class RNNModel:
    def __init__(self, keras_model, num_timsteps, num_features):
        self.keras_model = keras_model
        self.num_timsteps = num_timsteps
        self.num_features = num_features

    def predict(self, X):
        X = X.reshape((-1, self.num_timsteps, self.num_features))
        return self.keras_model.predict(X)


def get_occlusion_metric_lstm(model, X, y, X_sig_scores, fill_value=0):
    num_samples, num_timesteps, num_features = X.shape[0], X.shape[1], X.shape[2],

    rnn_model = RNNModel(model, num_timesteps, num_features)

    X_new = X.reshape((-1, num_timesteps * num_features))
    X_sig_scores_new = X_sig_scores.reshape((-1, num_timesteps * num_features))

    return get_occlusion_metric(rnn_model, X_new, y, X_sig_scores_new, metric='mse', fill_value=fill_value)


def get_occlusion_metric_lstm_masked_timesteps(model, X, y, X_sig_scores, fill_value):
    num_samples, num_timesteps, num_features = X.shape[0], X.shape[1], X.shape[2],

    X_sig_scores_sum = np.sum(X_sig_scores, axis=2) # Sum across all features in a timestep

    y_orig_func_vals = model.predict(X)
    y_output_node = y_orig_func_vals.argmax(axis=1)  # Predicted original classes

    # Column indices of features sorted by descending order of significance (most sig. timesteps)
    sorted_idx = np.argsort(X_sig_scores_sum, axis=1)[:, ::-1]

    remove_ratio = np.arange(0, 1.0001, 0.1)
    occluded_avg_func_vals = []
    occluded_metrics_list = []

    for r in remove_ratio:
        num_timesteps_removed = int(num_timesteps * r)
        remove_idx = sorted_idx[:, :num_timesteps_removed]

        X_occluded = np.array(X, copy=True)

        for i in range(num_samples):  # Destroy the full timestep in most significant r% timesteps
            X_occluded[i, remove_idx[i], :] = fill_value

        y_occluded_prob = model.predict(X_occluded)  # This has multiple probs for the multiclass case
        # Pick the new prediction (probability) for original classes
        y_occluded_prob_output_node = y_occluded_prob[np.arange(num_samples), y_output_node].reshape(-1, )

        avg_y_occluded = np.mean(y_occluded_prob_output_node)
        occluded_avg_func_vals.append(avg_y_occluded)

        y_occluded_classes = model.predict(X_occluded)
        occluded_mse = mean_squared_error(y, y_occluded_classes)

        occluded_metrics_list.append(occluded_mse)

    return remove_ratio, np.array(occluded_avg_func_vals), np.array(occluded_metrics_list)
