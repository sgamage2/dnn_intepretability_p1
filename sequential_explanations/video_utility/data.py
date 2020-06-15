import csv
import numpy as np
import glob
import os.path
from tensorflow.keras.utils import to_categorical
import random
from tensorflow.keras.preprocessing import image as Img
from tensorflow.keras.applications.inception_v3 import preprocess_input

class DataSet:
    def __init__(self, seq_length=40, class_limit=None, image_shape=(224, 224, 3), base_path=None, sequences_path=None):
        self.seq_length = seq_length
        self.class_limit = class_limit
        assert base_path is not None
        assert sequences_path is not None
        self.base_path = base_path
        self.sequences_path = sequences_path
        self.max_frames = 300  # max number of frames a video can have for us to use it
        self.data = self.get_data()
        self.classes = self.get_classes()
        self.data = self.clean_data()
        self.image_shape = image_shape


    def get_data(self):
        with open(os.path.join(self.base_path, 'data_file.csv'), 'r') as fin:
            reader = csv.reader(fin)
            data = list(reader)
        return data

    def clean_data(self):
        data_clean = []
        for item in self.data:
            if int(item[3]) >= self.seq_length and int(item[3]) <= self.max_frames \
                    and item[1] in self.classes:
                data_clean.append(item)

        return data_clean

    def get_classes(self):
        classes = []
        for item in self.data:
            if item[1] not in classes:
                classes.append(item[1])
        classes = sorted(classes)
        if self.class_limit is not None:
            return classes[:self.class_limit]
        else:
            return classes

    def get_class_one_hot(self, class_str):
        # Encode it first.
        label_encoded = self.classes.index(class_str)
        # Now one-hot it.
        label_hot = to_categorical(label_encoded, len(self.classes))
        assert len(label_hot) == len(self.classes)
        return label_hot

    def split_train_test(self):
        train = []
        test = []
        for item in self.data:
            if item[0] == 'train':
                train.append(item)
            else:
                test.append(item)
        return train, test

    def get_all_sequences_in_memory(self, train_test, data_type):
        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        print("Loading %d samples into memory for %sing." % (len(data), train_test))

        X, y = [], []
        for row in data:
            sequence = self.get_extracted_sequence(data_type, row)
            if sequence is None:
                print("Can't find sequence. Did you generate them?")
                raise
            X.append(sequence)
            y.append(self.get_class_one_hot(row[1]))
        return np.array(X), np.array(y)

    def get_extracted_sequence(self, data_type, sample):
        filename = sample[2]
        path = os.path.join(self.sequences_path, filename + '-' + str(self.seq_length) + '-' + data_type + '.npy')
        # print(path)
        if os.path.isfile(path):
            return np.load(path)
        else:
            return None

    def get_frames_by_filename(self, filename, data_type):
        sample = None
        for row in self.data:
            if row[2] == filename:
                sample = row
                break
        if sample is None:
            raise ValueError("Couldn't find sample: %s" % filename)
        sequence = self.get_extracted_sequence(data_type, sample)
        if sequence is None:
            raise ValueError("Can't find sequence. Did you generate them?")
        return sequence

    def get_frames_for_sample(self, sample):
        """Given a sample row from the data file, get all the corresponding frame
        filenames."""
        path = os.path.join(self.base_path, sample[0], sample[1])
        filename = sample[2]
        images = sorted(glob.glob(os.path.join(path, filename + '*jpg')))
        return images

    def get_frames_for_sample_set(self, train_test, num_samples, random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)

        train, test = self.split_train_test()
        data = train if train_test == 'train' else test

        if num_samples >= len(data):
            rand_videos = data
        else:
            rand_videos = random.sample(data, num_samples)

        width, height, channels = self.image_shape
        X_out = np.zeros(shape=(num_samples, self.seq_length, width, height, channels))
        y_out = []

        for n, video_row in enumerate(rand_videos):
            frames = self.get_frames_for_sample(video_row)  # Get the frames for this video_row.
            frames = self.rescale_list(frames, self.seq_length)  # Now downsample to just the ones we need.
            for t, image in enumerate(frames):
                img = Img.load_img(image, target_size=(width, height))
                x = Img.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                X_out[n, t, :] = x[0]

            y_out.append(self.get_class_one_hot(video_row[1]))

        return X_out, np.array(y_out)


    @staticmethod
    def rescale_list(input_list, size):
        assert len(input_list) >= size
        skip = len(input_list) // size
        output = [input_list[i] for i in range(0, len(input_list), skip)]
        return output[:size]