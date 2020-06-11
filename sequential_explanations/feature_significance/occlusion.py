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
        X_occluded = np.array(X, copy=True)

        for j in range(0, num_features - mask_size + 1, stride):
            # print('Masking features in range: timestep: {}, features: [{}, {}]'.format(t, j, j + mask_size - 1))
            X_occluded[:, t, j: j + mask_size] = fill_value
            y_occluded = temp_model.predict(X_occluded)

            y_diff = y_original - y_occluded
            X_sig_scores[:, t, j: j + mask_size] += y_diff

    return X_sig_scores


def get_occlusion_scores_lrcn(model, X, output_layer_idx, mask_size=16, stride=1, mask_full_timestep=False, fill_value=0):
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
        X_occluded = np.array(X, copy=True)

        for j in range(0, num_features - mask_size + 1, stride):

            assert False    # Mask 2-D patches of a single frame

            # print('Masking features in range: timestep: {}, features: [{}, {}]'.format(t, j, j + mask_size - 1))
            X_occluded[:, t, j: j + mask_size] = fill_value
            y_occluded = temp_model.predict(X_occluded)

            y_diff = y_original - y_occluded
            X_sig_scores[:, t, j: j + mask_size] += y_diff

    return X_sig_scores