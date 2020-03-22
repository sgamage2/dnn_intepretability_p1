import logging, time, os
import numpy as np
import keras
from keras.datasets import mnist, cifar10
import cnn
import utility


def get_exp_params():
    params = dict()
    # params['dataset'] = 'mnist' # Options: mnist, cifar-10
    params['dataset'] = 'cifar-10'
    params['dataset_portion'] = 1   # To reduce dataset for quick testing
    params['num_classes'] = 10
    params['batch_size'] = 128
    params['epochs'] = 10
    params['early_stop_patience'] = -1  # Disabled
    params['tensorboard_log_dir'] = 'output'
    params['results_dir'] = 'output'

    params['channels'] = 1
    if params['dataset'] == 'cifar-10':
        params['channels'] = 3

    layer_defs = list()
    params['layer_defs'] = layer_defs

    # Following CNN architecture works well for MNIST (with epochs = 10, batch_size = 128)
    # layer_defs.append({'type': 'Conv', 'filters': 32, 'kernel_size': (5, 5), 'strides': (1, 1), 'activation': 'relu'})
    # layer_defs.append({'type': 'MaxPool', 'pool_size': (2, 2)})
    # layer_defs.append({'type': 'Conv', 'filters': 64, 'kernel_size': (5, 5), 'strides': (1, 1), 'activation': 'relu'})
    # layer_defs.append({'type': 'MaxPool', 'pool_size': (2, 2)})
    # layer_defs.append({'type': 'Dense', 'units': 128, 'activation': 'relu'})
    # layer_defs.append({'type': 'Dropout', 'dropout_rate': 0.2})
    # layer_defs.append({'type': 'Dense', 'units': params['num_classes'], 'activation': 'softmax'}) # Output layer

    # Following CNN architecture works well for CIFAR-10 (with epochs = 10, batch_size = 128)
    layer_defs.append({'type': 'Conv', 'filters': 32, 'kernel_size': (3, 3), 'strides': (1, 1), 'activation': 'relu'})
    layer_defs.append({'type': 'MaxPool', 'pool_size': (2, 2)})
    layer_defs.append({'type': 'Conv', 'filters': 64, 'kernel_size': (3, 3), 'strides': (1, 1), 'activation': 'relu'})
    layer_defs.append({'type': 'MaxPool', 'pool_size': (2, 2)})
    layer_defs.append({'type': 'Dense', 'units': 64, 'activation': 'relu'})
    layer_defs.append({'type': 'Dropout', 'dropout_rate': 0.2})
    layer_defs.append({'type': 'Dense', 'units': params['num_classes'], 'activation': 'softmax'})  # Output layer

    return params


def setup_logging():
    # noinspection PyArgumentList
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
        level=logging.INFO
    )


def get_dataset(dataset_name):
    if dataset_name == 'mnist':
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # MNIST
    elif dataset_name == 'cifar-10':
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'] # CIFAR-10
    else:
        assert False

    dataset = (X_train, y_train), (X_test, y_test)

    return dataset, labels


def create_and_train_model(params, dataset):
    logging.info('Initializing cnn_model')
    cnn_model = cnn.CNN()
    cnn_model.initialize(params)

    (X_train, y_train), (X_val, y_val) = dataset

    logging.info('Training cnn_model')
    t0 = time.time()

    history = cnn_model.fit(X_train, y_train, X_val, y_val)

    time_to_train = time.time() - t0
    logging.info('Training complete. time_to_train = {:.2f} sec, {:.2f} min'.format(time_to_train, time_to_train / 60))

    return cnn_model, history


def preprocess(params, dataset):
    (X_train, y_train), (X_val, y_val) = dataset

    portion = params['dataset_portion']
    if portion < 1.0:
        (X_train, y_train), (_, _) = utility.split_dataset(X_train, y_train, [portion, 1-portion], random_seed=1)
        (X_val, y_val), (_, _) = utility.split_dataset(X_val, y_val, [portion, 1-portion], random_seed=1)

    X_train = X_train.astype('float64')
    X_val = X_val.astype('float64')
    X_train /= 255.0
    X_val /= 255.0

    X_train = X_train.reshape(X_train.shape[0], params['img_shape_x'], params['img_shape_y'], params['channels'])
    X_val = X_val.reshape(X_val.shape[0], params['img_shape_x'], params['img_shape_y'], params['channels'])

    num_classes = params['num_classes']

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)

    logging.info('X_train.shape = {}'.format(X_train.shape))
    logging.info('X_val.shape = {}'.format(X_val.shape))

    dataset = (X_train, y_train), (X_val, y_val)

    return dataset


def set_extra_params(exp_params, dataset):
    (X_train, y_train), (X_val, y_val) = dataset
    exp_params['img_shape_x'] = X_train.shape[1]
    exp_params['img_shape_y'] = X_train.shape[2]


def evaluate_model(params, model, dataset):
    (X_train, y_train), (X_test, y_test) = dataset
    y_test = y_test.argmax(axis=1)  # Integer labels

    y_test_pred = model.predict_classes(X_test)   # Integer labels

    utility.print_evaluation_report(y_test_pred, y_test, "Test set")


def plot_images(dataset, num_imgs, labels):
    (X_train, y_train), (X_test, y_test) = dataset
    rand_indices = np.random.choice(X_train.shape[0], num_imgs)

    X = X_train[rand_indices]
    y = y_train[rand_indices]

    utility.plot_images(X, y, labels)


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # This line disables GPU

    setup_logging()

    logging.info('Started experiment')

    exp_params = get_exp_params()

    dataset, labels = get_dataset(exp_params['dataset'])

    set_extra_params(exp_params, dataset)

    dataset = preprocess(exp_params, dataset)

    # plot_images(dataset, 25, labels)

    model, history = create_and_train_model(exp_params, dataset)

    utility.save_training_history(history, exp_params['results_dir'])
    utility.plot_training_history(history, exp_params['results_dir'])

    evaluate_model(exp_params, model, dataset)

    logging.info('Finished experiment')


