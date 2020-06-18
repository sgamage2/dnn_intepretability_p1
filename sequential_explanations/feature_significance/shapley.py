# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import logging, time
import models.ann
import utility
import matplotlib.pyplot as plt
import shap

from keras.applications.inception_v3 import preprocess_input
from skimage.segmentation import slic
import matplotlib.pylab as pl



def get_shapley_feature_sig_scores(model,X_train,X_test):
    explainer = shap.DeepExplainer(model, data=X_train)

    shap_values = explainer.shap_values(X_test)
    shap_values = shap_values[0]
    print(pd.DataFrame(shap_values).head())

    # explaining individual predictions
    print('Expected Value:', explainer.expected_value[0])

    # These 2 plots will be plotted by the caller
    # shap.summary_plot(shap_values, X_test, plot_type="bar")
    # shap.summary_plot(shap_values, X_test)

    feature_names = pd.DataFrame(X_train).columns[0:].values
    return shap_values
# '1' in explainer.expected_value[1]: we consider the feature effects of P(Y=1)
# shap_values[1][i]: extract the shape value of the ith sample that correponds to P(Y=1)
# X_train[i]: the ith sample in training set

# i = 1
# shap_values = explainer.shap_values(X_train)
# shap.force_plot(base_value=explainer.expected_value[0], shap_values=shap_values[0][1],
#                features=X_train[i], feature_names=list(feature_names))

def get_shapley_video(model, X_test):
    X_test = X_test[0]
    img_orig = X_test[0]
    # print(img_orig.shape)
    # segment the image so with don't have to explain every pixel
    segments_slic = slic(img_orig, n_segments=50, compactness=30, sigma=3)

    # define a function that depends on a binary mask representing if an image region is hidden
    def mask_image(zs, segmentation, image, background=None):
        if background is None:
            background = image.mean((0, 1))
        out = np.zeros((zs.shape[0], image.shape[0], image.shape[1], image.shape[2]))
        for i in range(zs.shape[0]):
            out[i, :, :, :] = image
            for j in range(zs.shape[1]):
                if zs[i, j] == 0:
                    out[i][segmentation == j, :] = background
        return out

    def f(z):
        return model.predict(preprocess_input(mask_image(z, segments_slic, img_orig, 255)))

    # use Kernel SHAP to explain the network's predictions
    explainer = shap.KernelExplainer(f, np.zeros((1, 50)))
    shap_values = explainer.shap_values(np.ones((1, 50)), nsamples=1000)  # runs inc 300 times

    # get the top predictions from the model
    preds = model.predict(preprocess_input(np.expand_dims(img_orig.copy(), axis=0)))
    top_preds = np.argsort(-preds)

    # make a color map
    from matplotlib.colors import LinearSegmentedColormap
    colors = []
    for l in np.linspace(1, 0, 100):
        colors.append((245 / 255, 39 / 255, 87 / 255, l))
    for l in np.linspace(0, 1, 100):
        colors.append((24 / 255, 196 / 255, 93 / 255, l))
    cm = LinearSegmentedColormap.from_list("shap", colors)

    def fill_segmentation(values, segmentation):
        out = np.zeros(segmentation.shape)
        for i in range(len(values)):
            out[segmentation == i] = values[i]
        return out

    # plot our explanations
    fig, axes = pl.subplots(nrows=1, ncols=4, figsize=(12, 4))
    inds = top_preds[0]
    axes[0].imshow(img_orig)
    axes[0].axis('off')
    max_val = np.max([np.max(np.abs(shap_values[i][:, :-1])) for i in range(len(shap_values))])
    for i in range(3):
        m = fill_segmentation(shap_values[inds[i]][0], segments_slic)
        # axes[i+1].set_title(feature_names[str(inds[i])][1])
        axes[i + 1].imshow(img_orig, alpha=0.15)
        im = axes[i + 1].imshow(m, cmap=cm, vmin=-max_val, vmax=max_val)
        axes[i + 1].axis('off')
    cb = fig.colorbar(im, ax=axes.ravel().tolist(), label="SHAP value", orientation="horizontal", aspect=60)
    cb.outline.set_visible(False)
    pl.show()