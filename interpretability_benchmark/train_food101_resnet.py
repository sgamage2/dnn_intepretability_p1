# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
r"""Testing script for data iterator and training for implementing ROAR.

"""

import os
from absl import flags
from absl.testing import absltest
import tensorflow.compat.v1 as tf
from interpretability_benchmark import data_input
# from interpretability_benchmark.train_resnet import food_101_params
from interpretability_benchmark.train_resnet import resnet_model_fn

import matplotlib.pyplot as plt
from PIL import Image
import io


# data_dir = '/home/sunanda/research/Datasets/food-101-small/prepared/'
data_dir = '../../Datasets/food-101-small/prepared/'
dest_dir = '../../Datasets/food-101-small/saliency_output/'
flags.DEFINE_string('dest_dir', dest_dir, 'Pathway to directory where output is saved.')
flags.DEFINE_integer('steps_per_loss_print', 10, 'How often to print out loss during training')

# model params
FLAGS = flags.FLAGS
FLAGS.steps_per_checkpoint = 200


food_101_params = {
    # Base params
    'train_batch_size': 256,
    'num_train_images': 3000,   # Used to compute epoch number in train_resnet.resnet_model_fn()
    'num_eval_images': 630, # Used to compute #steps to evaluate in train_resnet.main()
    'num_label_classes': 3,
    'num_train_steps': 10000,
    'steps_per_eval': 10,
    'base_learning_rate': 0.7,
    'weight_decay': 0.0001,
    'eval_batch_size': 256,
    'mean_rgb': [0.561, 0.440, 0.312],
    'stddev_rgb': [0.252, 0.256, 0.259],

    # Other params
    'output_dir':  dest_dir,
    'eval_steps':  5,   # won't be used
    'batch_size': 64,
    'threshold':  80,
    'data_format':  'channels_last',
    'momentum':  0.9,
    'lr_schedule': [(1.0, 5), (0.1, 30), (0.01, 60), (0.001, 80)],  # (multiplier, epoch to start) tuples
}


class TrainSaliencyTest(absltest.TestCase):

    def testDatasetLoading(self):
        return

        image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.io.FixedLenFeature([], tf.string),
            'image/channels': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'prediction_class': tf.io.FixedLenFeature([], tf.int64),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'raw_image': tf.io.FixedLenFeature([], tf.string)}

        def _parse_image_function(example_proto):
            # Parse the input tf.Example proto using the dictionary above.
            return tf.io.parse_single_example(example_proto, image_feature_description)

        filename = os.path.join(data_dir, 'train-00000-of-00002')
        raw_dataset = tf.data.TFRecordDataset([filename])
        parsed_image_dataset = raw_dataset.map(_parse_image_function)
        print(parsed_image_dataset)

        for record in parsed_image_dataset.take(2):
            print('height = {}'.format(record['height']))
            print('width = {}'.format(record['width']))
            print('label = {}'.format(record['label']))
            print('prediction_class = {}'.format(record['prediction_class']))

            image_bytes = record['raw_image'].numpy() # Returns bytes object

            image = Image.open(io.BytesIO(image_bytes))
            # image.show()


    def testEndToEnd(self):
        params = food_101_params

        data_files_pattern = os.path.join(data_dir, 'train*')

        dataset_ = data_input.DataIterator(
            mode=FLAGS.mode,
            data_directory=data_files_pattern,
            saliency_method='ig_smooth_2',
            transformation='modified_image',
            # transformation='raw_image',
            threshold=params['threshold'],
            keep_information=False,
            use_squared_value=True,
            mean_stats=params['mean_rgb'],
            std_stats=params['mean_rgb'],
            test_small_sample=False,
            num_cores=FLAGS.num_cores)

        images, labels = dataset_.input_fn(params)
        self.assertEqual(images.shape.as_list(), [params['batch_size'], 224, 224, 3])
        self.assertEqual(labels.shape.as_list(), [params['batch_size'],])

        run_config = tf.estimator.RunConfig(
            model_dir=FLAGS.dest_dir,
            save_checkpoints_steps=FLAGS.steps_per_checkpoint,
            log_step_count_steps=FLAGS.steps_per_loss_print
        )

        classifier = tf.estimator.Estimator(
            model_fn=resnet_model_fn,
            model_dir=FLAGS.dest_dir,
            params=params,
            config=run_config)
        classifier.train(input_fn=dataset_.input_fn, max_steps=params['num_train_steps'])
        tf.logging.info('finished training.')


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # This line disables GPU
    absltest.main()
