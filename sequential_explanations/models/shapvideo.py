import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import inception_v3

from tensorflow.keras.models import Model
import shap
import pandas as pd

c = np.load("/home/jiazhi/sequential_explanations/v_ApplyEyeMakeup_g01_c01-40-features.npy")

cnn_base = inception_v3.InceptionV3(weights='imagenet', pooling='avg', include_top=False)
        # self.cnn_base_with_top = resnet.ResNet152(weights='imagenet', include_top=True)   # For comparison
for layer in cnn_base.layers:
    layer.trainable = False
cnn_out = cnn_base.output
cnn_model = Model(inputs=cnn_base.input, outputs=cnn_out)

#background = c[np.random.choice(c.shape[0], 5, replace=False)]
x = c[0][np.newaxis, :]
e = shap.DeepExplainer(cnn_model, x)

shap_values = e.shap_values(x)
shap.image_plot(shap_values, x)
