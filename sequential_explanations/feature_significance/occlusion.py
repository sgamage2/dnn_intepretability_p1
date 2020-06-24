import tensorflow as tf
import numpy as np


def get_occlusion_scores(model, X, output_layer_idx, output_node=0, mask_size=1, stride=1, fill_value=0,):
    assert stride <= mask_size  # Ensure we don't miss any feature for occlusion
    X_sig_scores = np.zeros(X.shape)
    num_features = X.shape[1]

    temp_output_layer = model.layers[output_layer_idx]
    temp_model = tf.keras.models.Model(inputs=model.input, outputs=temp_output_layer.output)

    y_original = temp_model.predict(X)

    for j in range(0, num_features - mask_size + 1, stride):
        # print('Masking features in range: [{}, {}]'.format(j, j + mask_size - 1))
        X_occluded = np.array(X, copy=True)
        X_occluded[:, j: j + mask_size] = fill_value
        y_occluded = temp_model.predict(X_occluded)

        y_diff = y_original - y_occluded
        y_diff = y_diff[:, output_node].reshape(-1, 1)   # For multiclass classification
        X_sig_scores[:, j: j + mask_size] += y_diff

    return X_sig_scores


def get_occlusion_scores_lstm(model, X, output_layer_idx, mask_size=1, stride=1, mask_full_timestep=False, fill_value=0):
    assert stride <= mask_size  # Ensure we don't miss any feature for occlusion
    X_sig_scores = np.zeros(X.shape)
    num_timesteps = X.shape[1]
    num_features = X.shape[2]

    temp_output_layer = model.layers[output_layer_idx]
    temp_model = tf.keras.models.Model(inputs=model.input, outputs=temp_output_layer.output)

    y_original = temp_model.predict(X)

    if mask_full_timestep:
        mask_size = num_features
        stride = 1

    for t in range(num_timesteps):
        # X_occluded = np.array(X, copy=True)

        for j in range(0, num_features - mask_size + 1, stride):
            X_occluded = np.array(X, copy=True)

            # print('Masking features in range: timestep: {}, features: [{}, {}]'.format(t, j, j + mask_size - 1))
            X_occluded[:, t, j: j + mask_size] = fill_value
            y_occluded = temp_model.predict(X_occluded)

            y_diff = y_original - y_occluded
            X_sig_scores[:, t, j: j + mask_size] += y_diff

    return X_sig_scores


def get_occlusion_scores_lrcn(model, X, output_layer_idx, output_node, mask_window_size=32, stride=16, mask_full_timestep=False, fill_value=0):
    assert stride <= mask_window_size  # Ensure we don't miss any feature for occlusion
    X_sig_scores = np.zeros(X.shape)
    num_samples, num_timesteps, height, width, channels = X.shape[0], X.shape[1], X.shape[2], X.shape[3], X.shape[4]

    # temp_output_layer = model.layers[output_layer_idx]
    # temp_model = tf.keras.models.Model(inputs=model.input, outputs=temp_output_layer.output)
    assert output_layer_idx == -1   # Graph disconnects when building a temp model from arbitrary output layer --> Fix
    temp_model = model

    y_original = temp_model.predict(X)

    if mask_full_timestep:
        mask_window_size = width
        stride = 1

    m = mask_window_size

    time_stride = 10

    for t in range(0, num_timesteps, time_stride):  # Each frame separately
        # X_occluded = np.array(X, copy=True)

        print('t = {}'.format(t))

        # Mask 2-D patches
        for row in range(0, height - m + 1, stride):
            for col in range(0, width - m + 1, stride):
                # print('row = {}, col = {}'.format(row, col))
                X_occluded = np.array(X, copy=True)

                # print('Masking features in range: timestep: {}, features: [{}, {}]'.format(t, j, j + mask_size - 1))
                X_occluded[:, t:t+time_stride, row:row+m, col:col+m, :] = fill_value
                y_occluded = temp_model.predict(X_occluded)

                y_diff = y_original - y_occluded
                y_diff = y_diff[:, output_node].reshape(-1, 1)  # For multiclass classification
                X_sig_scores[:, t:t+time_stride, row:row+m, col:col+m, :] += y_diff

    return X_sig_scores
