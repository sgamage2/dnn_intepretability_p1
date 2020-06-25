import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error

# ----------------------------------------
# Interaface


def get_occlusion_metric_tabular(model, X, y, X_sig_scores, metric='accuracy', fill_value=0):
    occluder = CommonOccluder()
    return _get_occlusion_metric(model, X, y, X_sig_scores, metric, occluder, fill_value)


def get_occlusion_metric_seq_individual_features(model, X, y, X_sig_scores, fill_value=0):
    num_samples, num_timesteps, num_features = X.shape[0], X.shape[1], X.shape[2],
    rnn_model = RNNModel(model, num_timesteps, num_features)

    X_new = X.reshape((-1, num_timesteps * num_features))
    X_sig_scores_new = X_sig_scores.reshape((-1, num_timesteps * num_features))

    occluder = CommonOccluder()
    return _get_occlusion_metric(rnn_model, X_new, y, X_sig_scores_new, 'mse', occluder, fill_value)


def get_occlusion_metric_seq_mask_full_timesteps(model, X, y, X_sig_scores, fill_value=0):
    X_sig_scores_sum = np.sum(X_sig_scores, axis=2)      # Sum across all features in a timestep
    occluder = CommonOccluder()
    return _get_occlusion_metric(model, X, y, X_sig_scores_sum, 'mse', occluder, fill_value)


def get_occlusion_metric_flat_images(model, X, y, X_sig_scores, img_width, mask_size=8, fill_value=0):
    assert X.ndim == 2   # gray-scale flattened (for now)
    # num_imgs, height, width = X.shape

    occluder = FlatImageOccluder(img_width, mask_size)
    return _get_occlusion_metric(model, X, y, X_sig_scores, 'accuracy', occluder, fill_value)
    # X_sig_scores = X_sig_scores.reshape(num_imgs, -1)

# ----------------------------------------
# Implementation

# Thin wrapper class to hold a keras RNN model and do the reshaping before predict calls
class RNNModel:
    def __init__(self, keras_model, num_timsteps, num_features):
        self.keras_model = keras_model
        self.num_timsteps = num_timsteps
        self.num_features = num_features

    def predict(self, X):
        X = X.reshape((-1, self.num_timsteps, self.num_features))
        return self.keras_model.predict(X)


# Occlusion for tabular and seq
class CommonOccluder:
    def occlude(self, X_orig, remove_idx, fill_value):
        X_occluded = np.array(X_orig, copy=True)

        for i in range(len(X_occluded)):  # Destroy the most significant r% elements (features/ timesteps)
            X_occluded[i, remove_idx[i], ...] = fill_value

        return X_occluded


# Occlusion for flat images
class FlatImageOccluder():
    def __init__(self, img_width, mask_size):
        self.img_width = img_width
        self.mask_size = mask_size
        assert mask_size % 2 == 0

    def occlude(self, X_orig, remove_idx, fill_value):
        # Elements in remove_idx are in the range (0, height*width)
        s = self.mask_size // 2
        w = self.img_width
        rows = remove_idx // w
        cols = remove_idx % w

        X_occluded = np.array(X_orig, copy=True)
        X_occluded = X_occluded.reshape((-1, w, w))     # Conver to imgs of shape (w, w)

        for i in range(len(X_occluded)):  # Destroy the most significant r% elements (features/ timesteps)
            for row, col in zip(rows[i], cols[i]):
                X_occluded[i, row-s: row+s, col-s: col+s] = fill_value

        X_occluded = X_occluded.reshape((len(X_occluded), -1))    # Convert back to 1-D feature vector

        return X_occluded


def _get_occlusion_metric(model, X, y, X_sig_scores, metric, occluder, fill_value=0):
    num_samples = X.shape[0]
    # For tabular and seq_individual_features, elements are features. For seq_mask_full_timesteps, it is timesteps
    num_elements = X.shape[1]

    y_orig_func_vals = model.predict(X)
    y_output_node = y_orig_func_vals.argmax(axis=1)    # Predicted original classes

    # Column indices of features sorted by descending order of significance
    sorted_idx = np.argsort(X_sig_scores, axis=1)[:, ::-1]

    remove_ratio = np.arange(0, 1.0001, 0.1)
    occluded_avg_func_vals = []
    occluded_metrics_list = []

    for r in remove_ratio:
        num_elements_removed = int(num_elements * r)
        # num_features_kept = num_features - num_features_removed
        remove_idx = sorted_idx[:, :num_elements_removed]

        X_occluded = occluder.occlude(X, remove_idx, fill_value)

        y_occluded_prob = model.predict(X_occluded)  # This has multiple probs for the multiclass case
        # Pick the new prediction (probability) for original classes
        y_occluded_prob_output_node = y_occluded_prob[np.arange(num_samples), y_output_node].reshape(-1,)

        avg_y_occluded = np.mean(y_occluded_prob_output_node)
        occluded_avg_func_vals.append(avg_y_occluded)

        occluded_metric = -1
        assert metric in ['accuracy', 'mse']
        if metric == 'accuracy':
            y_occluded_classes = model.predict_classes(X_occluded)
            occluded_metric = accuracy_score(y, y_occluded_classes)
        elif metric == 'mse':
            y_occluded_classes = model.predict(X_occluded)
            occluded_metric = mean_squared_error(y, y_occluded_classes)

        occluded_metrics_list.append(occluded_metric)

    return remove_ratio, np.array(occluded_avg_func_vals), np.array(occluded_metrics_list)

