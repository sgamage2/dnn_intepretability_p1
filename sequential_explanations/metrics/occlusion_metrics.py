import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score


def get_occlusion_accuracies(model, X, y, X_sig_scores, fill_value=0):
    num_samples, num_features = X.shape[0], X.shape[1]

    y_orig_func_vals = model.predict(X)
    y_output_node = y_orig_func_vals.argmax(axis=1)    # Predicted original classes

    # Column indices of features sorted by descending order of significance
    sorted_idx = np.argsort(X_sig_scores, axis=1)[:, ::-1]

    remove_ratio = np.arange(0, 1.0001, 0.1)
    occluded_avg_func_vals = []
    occluded_avg_accuracy = []

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

        # y_occluded_classes = y_occluded_prob.argmax(axis=1)
        y_occluded_classes = model.predict_classes(X_occluded)

        avg_y_occluded = np.mean(y_occluded_prob_output_node)
        occluded_avg_func_vals.append(avg_y_occluded)

        occluded_accuracy = accuracy_score(y, y_occluded_classes)
        occluded_avg_accuracy.append(occluded_accuracy)

    return remove_ratio, np.array(occluded_avg_func_vals), np.array(occluded_avg_accuracy)

