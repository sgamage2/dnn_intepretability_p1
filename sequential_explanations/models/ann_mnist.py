import numpy as np
import logging, time, os
import models.ann
import utility
import matplotlib.pyplot as plt
import tensorflow as tf


exp_params = {}
exp_params['results_dir'] = 'output'
exp_params['exp_id'] = 'ann_mnist'
exp_params['img_width'] = 28
exp_params['num_classes'] = 10

exp_params['ann_input_nodes'] = exp_params['img_width'] * exp_params['img_width']
exp_params['output_nodes'] = exp_params['num_classes']
exp_params['ann_layer_units'] = [128, 64]
exp_params['ann_layer_activations'] = ['relu', 'relu']
exp_params['ann_layer_dropout_rates'] = [0.2, 0.2]
exp_params['ann_batch_size'] = 256
exp_params['ann_epochs'] = 10
exp_params['ann_early_stop_patience'] = -1  # Disabled
exp_params['ann_class_weights'] = 0


def get_mnist_dataset():
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    # flatten 2-D images to 1-D
    X_train = train_images.reshape(train_images.shape[0], -1)
    X_test = test_images.reshape(test_images.shape[0], -1)


    # Scale to [0, 1]
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    y_train = tf.keras.utils.to_categorical(train_labels, exp_params['num_classes'])
    y_test = tf.keras.utils.to_categorical(test_labels, exp_params['num_classes'])

    dataset = (X_train, y_train), (None, None), (X_test, y_test)
    return dataset


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
    y_true = y_true.argmax(axis=1)  # Integer labels
    utility.print_evaluation_report(y_true, y_pred, dataset_name)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    utility.initialize(exp_params)

    dataset = get_mnist_dataset()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = dataset

    model, history = create_and_train_ann(X_train, y_train, X_val, y_val, exp_params)
    utility.plot_training_history(history)

    utility.save_model(model, exp_params['results_dir'])

    # model = utility.load_model(exp_params['results_dir'])   # For testing

    evaluate_model(model, X_train, y_train, "Train set")
    evaluate_model(model, X_test, y_test, "Test set")

    # # Save test dataset (to use for predictions and feature significance)
    # np.save(exp_params['results_dir'] + '/X_test.npy', X_test)
    # np.save(exp_params['results_dir'] + '/y_test.npy', y_test)
    # np.save(exp_params['results_dir'] + '/X_train.npy', X_train)
    # np.save(exp_params['results_dir'] + '/y_train.npy', y_train)

    utility.save_all_figures(exp_params['results_dir'])


if __name__ == '__main__':
    main()

