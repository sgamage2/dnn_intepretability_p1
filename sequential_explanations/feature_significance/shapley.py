# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import logging, time
import models.ann
import utility
import matplotlib.pyplot as plt
import shap

exp_params = {}
exp_params['results_dir'] = 'output'
exp_params['exp_id'] = 'ann_toy'
exp_params['num_train_samples'] = 5000
exp_params['num_test_samples'] = 2500
exp_params['num_features'] = 16

exp_params['ann_input_nodes'] = exp_params['num_features']
exp_params['output_nodes'] = 1
exp_params['ann_layer_units'] = [10, 5]
exp_params['ann_layer_activations'] = [None, None]
exp_params['ann_layer_dropout_rates'] = [0.2, 0.2]
exp_params['ann_batch_size'] = 32
exp_params['ann_epochs'] = 25
exp_params['ann_early_stop_patience'] = -1  # Disabled
exp_params['ann_class_weights'] = 0


# Synthetic task described in A Benchmark for Interpretability Methods in Deep Neural Networks, Hooker et al
# Can only call this once per datset (random no.s in this are generated one per dataset)
def generate_datset(num_samples, num_features=16, num_relevant_features=4):
    a_vec = np.zeros(num_features)
    a_vec[:num_relevant_features] = np.random.normal(size=num_relevant_features)
    a_vec = np.broadcast_to(a_vec, shape=(num_samples, num_features))

    d_vec = np.random.normal(size=num_features)
    d_vec = np.broadcast_to(d_vec, shape=(num_samples, num_features))

    # No need to broadcast the following (automatically done in eqn for X)
    eta = np.random.normal(size=(num_samples, 1))
    eps = np.random.normal(size=(num_samples, 1))
    z = np.random.normal(size=(num_samples, 1))

    y = (z > 0).astype(int)
    y = y.reshape(num_samples)
    logging.info('Count of Class = 1: {} / {}'.format(np.sum(y), num_samples))

    X = a_vec * z / 10 + d_vec * eta + eps / 10
    assert X.shape == (num_samples, num_features)
    assert y.shape == (num_samples,)

    return X, y


def create_and_train_ann(X_train, y_train, X_val, y_val, params):
    model = models.ann.ANN()
    model.initialize(params)

    logging.info('Training model')
    t0 = time.time()

    history = model.fit(X_train, y_train, X_val, y_val)

    time_to_train = time.time() - t0
    logging.info('Training complete. time_to_train = {:.2f} sec, {:.2f} min'.format(time_to_train, time_to_train / 60))

    return model, history


def evaluate_model(model, X, y_true, dataset_name):
    logging.info('{}: Predicting and evaluating model'.format(dataset_name))

    y_pred = model.predict_classes(X)  # Integer labels
    utility.print_evaluation_report(y_true, y_pred, dataset_name)



utility.initialize(exp_params)

total_samples = exp_params['num_train_samples'] + 2 * exp_params['num_test_samples']
X, y = generate_datset(total_samples, exp_params['num_features'])

train_ratio = exp_params['num_train_samples'] / total_samples
test_ratio = exp_params['num_test_samples'] / total_samples

datset = utility.split_dataset(X, y, (train_ratio, test_ratio, test_ratio))
(X_train, y_train), (X_val, y_val), (X_test, y_test) = datset

model, history = create_and_train_ann(X_train, y_train, X_val, y_val, exp_params)
utility.plot_training_history(history)

model_filename = exp_params['results_dir'] + '/ann_toy_model.pickle'
# utility.save_obj_to_disk(model, model_filename)

evaluate_model(model, X_train, y_train, "Train set")
evaluate_model(model, X_test, y_test, "Test set")

# Save test dataset (to use for predictions and feature significance)
np.save(exp_params['results_dir'] + '/X_test.npy', X_test)
np.save(exp_params['results_dir'] + '/y_test.npy', y_test)

utility.save_all_figures(exp_params['results_dir'])

explainer = shap.DeepExplainer(model.ann, data=X_train)

shap_values = explainer.shap_values(X_test)
shap_values = shap_values[0]
print(pd.DataFrame(shap_values).head())

# explaining individual predictions
print('Expected Value:', explainer.expected_value[0])
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)

feature_names = pd.DataFrame(X_train).columns[0:].values
# '1' in explainer.expected_value[1]: we consider the feature effects of P(Y=1)
# shap_values[1][i]: extract the shape value of the ith sample that correponds to P(Y=1)
# X_train[i]: the ith sample in training set

# i = 1
# shap_values = explainer.shap_values(X_train)
# shap.force_plot(base_value=explainer.expected_value[0], shap_values=shap_values[0][1],
#                features=X_train[i], feature_names=list(feature_names))