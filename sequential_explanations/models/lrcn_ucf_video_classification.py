import logging, time, os
import models.lrcn
import utility
from video_utility.data import DataSet
import numpy as np

exp_params = {}
exp_params['data_base_path'] = '/home/jiazhi/videoucftest/data'
exp_params['sequences_path'] = '/home/jiazhi/videoucftest/data/sequences111'
exp_params['results_dir'] = 'output'
exp_params['exp_id'] = 'lrcn_1'
exp_params['image_shape'] = (224, 224, 3)
exp_params['latent_dim'] = 2048  # No. of features of LSTM input
exp_params['lstm_layer_units'] = [256, 256]
exp_params['lstm_layer_dropout_rates'] = [0.2, 0.2]
exp_params['output_nodes'] = 70  # No. of classes
exp_params['lstm_time_steps'] = 40   # No. of frames to classify at one time (LSTM unrolling)
exp_params['lstm_batch_size'] = 64
exp_params['lstm_epochs'] = 10
exp_params['lstm_early_stop_patience'] = -1  # Disabled


def create_and_train_lrcn(X_train, y_train, X_test, y_test, params):
    model = models.lrcn.LRCNClassifier()
    model.initialize(params)

    logging.info('Training model')
    t0 = time.time()

    # history = None
    history = model.fit_lstm(X_train, y_train, X_test, y_test)

    time_to_train = time.time() - t0
    logging.info('Training complete. time_to_train = {:.2f} sec, {:.2f} min'.format(time_to_train, time_to_train / 60))

    return model, history


def evaluate_model(model, X, y_true, dataset_name):
    logging.info('{}: Predicting and evaluating model'.format(dataset_name))

    y_pred = model.predict_lrcn(X)

    # Integer labels
    y_pred = y_pred.argmax(axis=1)
    y_true = y_true.argmax(axis=1)

    utility.print_evaluation_report(y_true, y_pred, dataset_name)


def main():
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    utility.initialize(exp_params)

    # ------------------------
    # Load sequences of features from CNN last layer

    dataset = DataSet(seq_length=exp_params['lstm_time_steps'],
                   class_limit=exp_params['output_nodes'],
                   image_shape=exp_params['image_shape'],
                   base_path=exp_params['data_base_path'],
                   sequences_path=exp_params['sequences_path'])

    X_train, y_train = dataset.get_all_sequences_in_memory('train', 'features')
    X_test, y_test = dataset.get_all_sequences_in_memory('test', 'features')

    logging.info('X_train.shape = {}, y_train.shape = {}'.format(X_train.shape, y_train.shape))
    logging.info('X_test.shape = {}, y_test.shape = {}'.format(X_test.shape, y_test.shape))

    # ------------------------
    # Train model
    model, history = create_and_train_lrcn(X_train, y_train, X_test, y_test, exp_params)

    # dummy_videos = np.random.randn(5, 40, 229, 229, 3)
    # preds = model.predict_lrcn(dummy_videos)
    # print('preds.shape = {}'.format(preds.shape))

    utility.save_model(model, exp_params['results_dir'])

    # model = utility.load_model(exp_params['results_dir'])  # For testing

    # ------------------------
    # Evaluate model on test set videos (img seqs)

    X_video_test, y_video_test = dataset.get_frames_for_sample_set('test', num_samples=8)
    logging.info('X_test.shape = {}, y_test.shape = {}'.format(X_video_test.shape, y_video_test.shape))

    evaluate_model(model, X_video_test, y_video_test, "Test set")


if __name__ == "__main__":
    main()
