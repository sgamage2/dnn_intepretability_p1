import numpy as np
import logging
from sklearn.metrics import accuracy_score, mean_squared_error

# ----------------------------------------
# Interaface


def get_occlusion_metric_tabular(model, X, y, X_sig_scores, metric='accuracy', fill_value=0):
    occluder = CommonOccluder(X.shape[1:])
    output_nodes = np.array(y, dtype=int)
    return _get_occlusion_metric(model, X, y, X_sig_scores, output_nodes, metric, occluder, fill_value)


def get_occlusion_metric_seq_individual_features(model, X, y, X_sig_scores, fill_value=0):
    num_samples, num_timesteps, num_features = X.shape[0], X.shape[1], X.shape[2],
    # rnn_model = RNNModel(model, num_timesteps, num_features)

    X_new = X.reshape((-1, num_timesteps * num_features))
    X_sig_scores_new = X_sig_scores.reshape((-1, num_timesteps * num_features))

    occluder = CommonOccluder(X.shape[1:])
    output_nodes = 0
    return _get_occlusion_metric(model, X_new, y, X_sig_scores_new, output_nodes, 'mse', occluder, fill_value)


def get_occlusion_metric_seq_mask_full_timesteps(model, X, y, X_sig_scores, fill_value=0):
    X_sig_scores_sum = np.sum(X_sig_scores, axis=2)      # Sum across all features in a timestep
    occluder = CommonOccluder(X.shape[1:])
    output_nodes = 0
    return _get_occlusion_metric(model, X, y, X_sig_scores_sum, output_nodes, 'mse', occluder, fill_value)


def get_occlusion_metric_flat_images(model, X, y, X_sig_scores, img_width, mask_size=2, fill_value=0):
    assert X.ndim == 2   # gray-scale flattened (for now)
    # num_imgs, height, width = X.shape

    occluder = FlatImageOccluder(img_width, mask_size)
    output_nodes = 0
    return _get_occlusion_metric(model, X, y, X_sig_scores, output_nodes, 'accuracy', occluder, fill_value)
    # X_sig_scores = X_sig_scores.reshape(num_imgs, -1)


def get_occlusion_metric_video(model, X, y, X_sig_scores, mask_size=2, fill_value=0):
    assert X.ndim == 5   # seqs of images
    num_samples, num_timesteps, width, height, channels = X.shape
    assert width == height

    X_sig_scores_sum = np.sum(X_sig_scores, axis=4)      # Sum across channels of a image

    X_new = X.reshape((-1, num_timesteps * width * height * channels))
    # We are considering significance of individual pixels --> flatten all dims of a video
    X_sig_scores_new = X_sig_scores_sum.reshape((-1, num_timesteps * width * height))

    occluder = VideoOccluder(X.shape[1:], mask_size)
    output_nodes = y.argmax(axis=1)
    return _get_occlusion_metric(model, X_new, y, X_sig_scores_new, output_nodes, 'mse', occluder, fill_value)


# ----------------------------------------
# Implementation


# Occlusion for tabular and seq
class CommonOccluder:
    def __init__(self, shape):
        self.shape = shape  # The remaining part of the shape shape that X_occluded should take --> X_occluded: (-1, *shape)

    def occlude(self, X_orig, remove_idx, fill_value):
        X_occluded = np.array(X_orig, copy=True)

        for i in range(len(X_occluded)):  # Destroy the most significant r% elements (features/ timesteps)
            X_occluded[i, remove_idx[i], ...] = fill_value

        X_occluded = X_occluded.reshape((-1, *self.shape))
        return X_occluded


# Occlusion for videos (sequences of video)
class VideoOccluder():
    def __init__(self, dimensions, mask_size):
        self.dimensions = dimensions
        self.mask_size = mask_size
        assert mask_size % 2 == 0

    def occlude(self, X_orig, remove_idx, fill_value):
        timesteps, w, h, channels = self.dimensions
        s = int(self.mask_size // 2)

        remove_idx = np.array(remove_idx) # Elements in remove_idx are in the range (0, timesteps * height * width)
        frames = np.array(remove_idx // (w * h))
        remainder = np.array(remove_idx % (w * h))
        rows = np.array(remainder // w)
        cols = np.array(remainder % w)

        X_occluded = np.array(X_orig, copy=True)
        X_occluded = X_occluded.reshape((-1, *self.dimensions))     # Conver to videos of shape (B, T, w, w, c)

        # This kind of vectorization is not possible
        # X_occluded[np.arange(len(X_occluded)), frames, rows-s: rows+s, cols-s: cols+s, :] = fill_value

        for i in range(len(X_occluded)):  # Each sample
            # It maybe possible to vectorize the inner loop
            # f = frames[i]
            # r = rows[i]
            # c = cols[i]
            # print(f.shape, r.shape, c.shape)
            # print(f.dtype, r.dtype, c.dtype)
            # X_occluded[i, f, r-s: r+s, c-s: c+s, :] = fill_value

            for frame, row, col in zip(frames[i], rows[i], cols[i]):
                X_occluded[i, frame, row-s: row+s, col-s: col+s, :] = fill_value

        return X_occluded


# Occlusion for flat images
class FlatImageOccluder():
    def __init__(self, img_width, mask_size):
        self.img_width = img_width
        self.mask_size = mask_size
        assert mask_size % 2 == 0

    def occlude(self, X_orig, remove_idx, fill_value):
        # Elements in remove_idx are in the range (0, height * width)
        s = self.mask_size // 2
        w = self.img_width
        rows = remove_idx // w
        cols = remove_idx % w

        X_occluded = np.array(X_orig, copy=True)
        X_occluded = X_occluded.reshape((-1, w, w))     # Conver to imgs of shape (w, w)

        for i in range(len(X_occluded)):  # Each sample
            for row, col in zip(rows[i], cols[i]):
                X_occluded[i, row-s: row+s, col-s: col+s] = fill_value

        X_occluded = X_occluded.reshape((len(X_occluded), -1))    # Convert back to 1-D feature vector

        return X_occluded


def _get_occlusion_metric(model, X, y, X_sig_scores, output_nodes, metric, occluder, fill_value=0):
    num_samples = X.shape[0]
    # For tabular and seq_individual_features, elements are features. For seq_mask_full_timesteps, it is timesteps
    num_elements = X.shape[1]

    # Column indices of features sorted by descending order of significance
    sorted_idx = np.argsort(X_sig_scores, axis=1)[:, ::-1]

    remove_ratio = np.arange(0, 1.0001, 0.1)
    occluded_avg_func_vals = []
    occluded_metrics_list = []

    logging.info('Calculating occlusion metric')
    for r in remove_ratio:
        logging.info('remove_ratio = {:.2f}'.format(r))
        num_elements_removed = int(num_elements * r)
        # num_features_kept = num_features - num_features_removed
        remove_idx = sorted_idx[:, :num_elements_removed]

        X_occluded = occluder.occlude(X, remove_idx, fill_value)

        y_occluded_prob = model.predict(X_occluded)  # This has multiple probs for the multiclass case
        # Pick the new prediction (probability) for original classes
        y_occluded_prob_output_node = y_occluded_prob[np.arange(num_samples), output_nodes].reshape(-1,)

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

