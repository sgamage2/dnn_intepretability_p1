import os, logging
import keras
from keras.applications import inception_v3 as inc_net
from keras.preprocessing import image
from keras.applications.imagenet_utils import decode_predictions
from skimage.io import imread
import matplotlib.pyplot as plt
# %matplotlib inline
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries


data_dir = '../../Datasets/images/'
results_dir = 'output'
num_local_samples = 1000  # no. of samples (synthetically generated) used to train local model
num_regions = 5   # no. of regions (aka superpixels) used when extracting an explanation


def setup_logging():
    # noinspection PyArgumentList
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
        level=logging.INFO
    )


def transform_img_fn(path_list):
    out = []
    for img_path in path_list:
        img = image.load_img(img_path, target_size=(299, 299))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = inc_net.preprocess_input(x)
        out.append(x)
    return np.vstack(out)


def plot_image(img, filename):
    # I'm dividing by 2 and adding 0.5 because of how this Inception represents images
    plt.imshow(img / 2 + 0.5)
    filename = os.path.join(results_dir, filename)
    plt.savefig(filename, bbox_inches='tight')


if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # This line disables GPU

    setup_logging()

    # --------------------------------------------------
    # Load Inception and predict on an image
    logging.info('Loading inception model')
    inet_model = inc_net.InceptionV3()

    file = os.path.join(data_dir, 'cat_mouse_1.jpg')

    images = transform_img_fn([file])
    # plot_image(images[0], 'cat_mouse_1.png')

    logging.info('Predicting on image with inception model')
    preds = inet_model.predict(images)
    decoded_preds = decode_predictions(preds)[0]
    print(*decoded_preds, sep="\n")

    # --------------------------------------------------
    # LIME explanations for prediction

    explainer = lime_image.LimeImageExplainer()

    logging.info('Generating explanation for image (will train local model)')
    explanation = explainer.explain_instance(images[0], inet_model.predict, top_labels=5, hide_color=0, num_samples=num_local_samples)
    # The explanation object now contains the local model and the explanation that can be extracted from the local model
    # Explanations are extracted by calling the get_image_and_mask with different params
    logging.info('Explanation generated')

    # logging.info('---------- 1')
    # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=num_regions, hide_rest=True)
    # plt.figure()
    # plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    #
    # logging.info('---------- 2')
    # plt.figure()
    # temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=num_regions, hide_rest=False)
    # plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

    # logging.info('---------- 3')

    for i in range(4):  # For top 3 labels
        temp, mask = explanation.get_image_and_mask(explanation.top_labels[i], positive_only=False,
                                                    num_features=num_regions, hide_rest=False)
        plt.figure()
        plt.title('Explanation for label: {}'.format(decoded_preds[i][1]))
        plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))

    plt.show()
