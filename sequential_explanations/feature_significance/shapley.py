# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import logging, time
import models.ann
import utility
import matplotlib.pyplot as plt
import shap

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