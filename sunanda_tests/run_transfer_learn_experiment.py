import logging, time, os
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import vgg16
from keras.applications import resnet50
import cnn
import utility


def get_exp_params():
    params = dict()
    params['base_model'] = 'resnet-50'  # Options resnet-50, vgg16
    params['expected_img_size'] = (224, 224)
    params['dataset'] = 'tiny_imagenet'   # Options: tiny_imagenet
    # Options: train_new_classifier, train_all_from_scratch, train_new_classifier_and_fine_tune
    # See the transfer_learn() function for descriptions of strategies
    params['tl_strategy'] = 'train_new_classifier'
    params['dataset_portion'] = 1   # To reduce dataset for quick testing
    params['num_classes'] = 1000
    params['batch_size'] = 128
    params['epochs'] = 10
    params['early_stop_patience'] = -1  # Disabled
    params['tensorboard_log_dir'] = 'output'
    params['results_dir'] = 'output'

    params['channels'] = 3

    layer_defs = list()
    params['layer_defs'] = layer_defs

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


def get_dataset(params, preproc_func):
    dataset_name = params['dataset']
    batch_size = params['batch_size']
    target_size = params['expected_img_size']

    if dataset_name == 'tiny_imagenet':
        logging.info("Creating tiny_imagenet data generators")
        datagen = ImageDataGenerator(preprocessing_function=preproc_func)
        train_it = datagen.flow_from_directory('../../Datasets/tiny-imagenet-200-truncated/train/', batch_size=batch_size, target_size=target_size)
        # val_it = datagen.flow_from_directory('../../Datasets/tiny-imagenet-200/train/', batch_size=batch_size, target_size=target_size)
        # test_it = datagen.flow_from_directory('../../Datasets/tiny-imagenet-200/train/', batch_size=batch_size, target_size=target_size)
        val_it = None
        test_it = None
    else:
        assert False

    dataset = (train_it, val_it, test_it)

    return dataset


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


def evaluate_model(params, model, dataset):
    (train_it, val_it, test_it) = dataset

    # y_test = y_test.argmax(axis=1)  # Integer labels

    X_batch, y_batch = train_it.next()

    y_pred = model.predict(X_batch)   # Integer labels

    a = 5

    # utility.print_evaluation_report(y_test_pred, y_test, "Test set")


def plot_random_images(dataset, num_imgs):
    (train_it, val_it, test_it) = dataset

    X_batch, y_batch = train_it.next()

    assert X_batch.shape[0] >= num_imgs

    rand_indices = np.random.choice(X_batch.shape[0], num_imgs)

    X = X_batch[rand_indices]
    y = y_batch[rand_indices]

    labels = list(train_it.class_indices.keys())
    y_ints = y.argmax(axis=1)  # Integer labels
    y_labels = [labels[y_int] for y_int in y_ints]

    utility.plot_images(X, y_labels)


def get_base_model(params):
    model_name = exp_params['base_model']

    if model_name == 'vgg16':
        base_model = vgg16.VGG16(weights='imagenet')
        preproc_func = vgg16.preprocess_input
        decode_preds_func = vgg16.decode_predictions
    elif model_name == 'resnet-50':
        base_model = resnet50.ResNet50(weights='imagenet')
        preproc_func = resnet50.preprocess_input
        decode_preds_func = resnet50.decode_predictions
    else:
        assert False

    base_model.summary()
    return base_model, preproc_func, decode_preds_func


def train_new_classifier(base_model, dataset, params):
    pass
    # (train_it, val_it, test_it) = dataset
    #
    # i = 0
    # for inputs_batch, labels_batch in train_it:
    #     features_batch = conv_base.predict(inputs_batch)
    #     features[i * batch_size: (i + 1) * batch_size] = features_batch
    #     labels[i * batch_size: (i + 1) * batch_size] = labels_batch
    #     i += 1
    #     if i * batch_size >= sample_count:
    #         break


def transfer_learn(base_model, dataset, params):
    tl_strategy = params['tl_strategy']

    if tl_strategy == 'train_new_classifier':
        # Remove the fully connected layers and use output from the convolutional base as extracted features
        # Train a new classifier on this data
        # Good for when new dataset is similar to previous, but small
        new_model, history = train_new_classifier(base_model, dataset, params)
    elif tl_strategy == 'train_all_from_scratch':
        # Do not use pre-trained weights in the base model (reset them to random)
        # Train the full model from scratch (with a suitable output layer matching the no. of classes in the current problem)
        # Good for when new dataset is different to previous, but large
        assert False
    elif tl_strategy == 'train_new_classifier_and_fine_tune':
        # First, freeze the conv base and train the fully connected layers for a few epochs
        # (with a suitable output layer matching the no. of classes in the current problem)
        # Then unfreeze the last N conv base layers and do a few more training epochs
        # (fine-tuning the unfrozen conv base + fully connected layers)
        # Choose N based on similarity between current dataset and the one that the base model was trained on
        # If very similar, N can be small (higher conv base layers are valid for the current problem: transferable)
        # If very different, N should be large (only a few lower conv base layers have low-level transferable features)
        assert False
    else:
        assert False

    return new_model


def predict_on_random_batch(dataset, base_model, decode_preds_func):
    logging.info('Predicting on a random batch and plotting images and predicted labels')
    (train_it, val_it, test_it) = dataset
    X_batch, y_batch = train_it.next()

    preds = base_model.predict(X_batch)
    decoded_preds = decode_preds_func(preds, top=3)  # Tuples of (class, description, probability) for top 3 classes

    logging.info('Predictions of 25 images')
    print(*decoded_preds[0:25], sep="\n")    # First 25 of the batch

    labels = [pred[0][1] for pred in decoded_preds[0:25]]
    utility.plot_images(X_batch[0:25], labels)


def predict_on_training_data(dataset, base_model, decode_preds_func, params):
    logging.info('Predicting on the full training set and evaluating')
    (train_it, val_it, test_it) = dataset
    n_samples = train_it.samples
    num_classes = params['num_classes']
    batch_size = params['batch_size']

    y_true = np.zeros(shape=(n_samples))
    y_preds = np.zeros(shape=(n_samples, num_classes))

    num_batches = n_samples // batch_size
    next_print_progress = 0
    stop_progress = 100 #Percentage

    i = 0
    for X_batch, y_batch in train_it:
        progress = i * 100 / num_batches
        if progress >= next_print_progress:
            print('Progress = {:.0f}%'.format(progress))
            next_print_progress += 10

        if progress >= stop_progress:
            print('Progress = {:.0f}%. Stopping predictions'.format(progress))
            predicted_batches = i
            break

        preds = base_model.predict(X_batch) # Need to be decoded later
        y_preds[i * batch_size: (i + 1) * batch_size] = preds
        y_true[i * batch_size: (i + 1) * batch_size] = y_batch.argmax(axis=1)  # Integer labels (of loaded dataset)

        i += 1
        if i * batch_size >= n_samples:
            predicted_batches = i
            break

    logging.info('Decoding predictions')
    decoded_preds = decode_preds_func(y_preds, top=3)  # Tuples of (class, description, probability) for top 3 classes

    # logging.info('Predictions of 25 images')
    # print(*decoded_preds[0:25], sep="\n")  # First 25 of the batch

    ints_to_labels_map = {idx: label for label, idx in train_it.class_indices.items()}

    y_true_labels = [ints_to_labels_map[y] for y in y_true[0: batch_size * predicted_batches]]
    y_pred_labels = [pred[0][0] for pred in decoded_preds[0: batch_size * predicted_batches]]

    logging.info('Evaluating predictions')
    utility.print_evaluation_report(y_pred_labels, y_true_labels, "Training set")


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # This line disables GPU
    # print('Current dir = {}'.format(os.getcwd()))

    setup_logging()

    logging.info('Started experiment')

    exp_params = get_exp_params()

    base_model, preproc_func, decode_preds_func = get_base_model(exp_params)

    base_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

    dataset = get_dataset(exp_params, preproc_func)

    # plot_random_images(dataset, 25)

    # predict_on_random_batch(dataset, base_model, decode_preds_func)
    predict_on_training_data(dataset, base_model, decode_preds_func, exp_params)

    # new_model, history = transfer_learn(base_model, dataset, exp_params)

    # utility.save_training_history(history, exp_params['results_dir'])
    # utility.plot_training_history(history)

    # evaluate_model(exp_params, model, dataset)

    logging.info('Finished experiment')


