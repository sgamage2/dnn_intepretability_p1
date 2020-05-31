import numpy as np
import logging, time, os
from sklearn.metrics import mean_squared_error
import models.lstm
import utility
import matplotlib.pyplot as plt


exp_params = {}
exp_params['results_dir'] = 'output'
exp_params['exp_id'] = 'lstm_adding_1'
exp_params['num_train_seqs'] = 5000
exp_params['num_test_seqs'] = 2500
exp_params['seq_length'] = 12

exp_params['lstm_time_steps'] = exp_params['seq_length']
exp_params['lstm_input_nodes'] = 2
exp_params['output_nodes'] = 1
exp_params['lstm_layer_units'] = [10, 5]
exp_params['lstm_layer_activations'] = ['tanh', 'tanh']
exp_params['lstm_layer_dropout_rates'] = [0.2, 0.2]
exp_params['lstm_epochs'] = 25
exp_params['lstm_early_stop_patience'] = -1
exp_params['lstm_batch_size'] = 16


# Taken from: https://minpy.readthedocs.io/en/latest/tutorial/rnn_tutorial/rnn_tutorial.html
def adding_problem_generator(N, seq_len=6, high=1):
    """ A data generator for adding problem.

    The data definition strictly follows Quoc V. Le, Navdeep Jaitly, Geoffrey E.
    Hintan's paper, A Simple Way to Initialize Recurrent Networks of Rectified
    Linear Units.

    The single datum entry is a 2D vector with two rows with same length.
    The first row is a list of random data; the second row is a list of binary
    mask with all ones, except two positions sampled by uniform distribution.
    The corresponding label entry is the sum of the masked data. For
    example:

     input          label
     -----          -----
    1 4 5 3  ----->   9 (4 + 5)
    0 1 1 0

    :param N: the number of the entries.
    :param seq_len: the length of a single sequence.
    :param p: the probability of 1 in generated mask
    :param high: the random data is sampled from a [0, high] uniform distribution.
    :return: (X, Y), X the data, Y the label.
    """
    X_num = np.random.uniform(low=0, high=high, size=(N, seq_len, 1))
    X_mask = np.zeros((N, seq_len, 1))
    Y = np.ones((N, 1))
    for i in range(N):
        # Default uniform distribution on position sampling
        positions = np.random.choice(seq_len, size=2, replace=False)
        X_mask[i, positions] = 1
        Y[i, 0] = np.sum(X_num[i, positions])
    X = np.append(X_num, X_mask, axis=2)
    return X, Y


def create_and_train_lstm(X_train, y_train, X_val, y_val, params):
    model = models.lstm.LSTMRegressor()
    model.initialize(params)

    logging.info('Training model')
    t0 = time.time()

    history = model.fit(X_train, y_train, X_val, y_val)

    time_to_train = time.time() - t0
    logging.info('Training complete. time_to_train = {:.2f} sec, {:.2f} min'.format(time_to_train, time_to_train / 60))

    return model, history


def evaluate_model(model, X, y_true, dataset_name):
    logging.info('{}: Predicting and evaluating model'.format(dataset_name))

    y_pred = model.predict(X)
    mse = mean_squared_error(y_true, y_pred)

    logging.info('{}: MSE = {:.4f}'.format(dataset_name, mse))

    return mse


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    utility.initialize(exp_params)

    X_train, y_train = adding_problem_generator(N=exp_params['num_train_seqs'], seq_len=exp_params['seq_length'])
    X_val, y_val = adding_problem_generator(N=exp_params['num_test_seqs'], seq_len=exp_params['seq_length'])
    X_test, y_test = adding_problem_generator(N=exp_params['num_test_seqs'], seq_len=exp_params['seq_length'])

    model, history = create_and_train_lstm(X_train, y_train, X_val, y_val, exp_params)
    utility.plot_training_history(history)

    utility.save_model(model, exp_params['results_dir'])

    # model = utility.load_model(exp_params['results_dir'])  # For testing

    evaluate_model(model, X_train, y_train, "Train set")
    evaluate_model(model, X_test, y_test, "Test set")

    # Save test dataset (to use for predictions and feature significance)
    np.save(exp_params['results_dir'] + '/X_train.npy', X_train)
    np.save(exp_params['results_dir'] + '/y_train.npy', y_train)
    np.save(exp_params['results_dir'] + '/X_test.npy', X_test)
    np.save(exp_params['results_dir'] + '/y_test.npy', y_test)

    utility.save_all_figures(exp_params['results_dir'])


if __name__ == '__main__':
    main()

