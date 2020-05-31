import numpy as np

import lime
import lime.lime_tabular

def get_lime_feature_sig_scores(model, X_train, X_test, feature_names):
    #If the feature is numerical, compute the mean and std, and discretize it into quartiles.
    explainer = lime.lime_tabular.LimeTabularExplainer(X_train,
                                                   feature_names = feature_names,
                                                   class_names = [0, 1],
                                                   discretize_continuous=True)

    exp = explainer.explain_instance(X_test[0], model.predict_proba, num_features=16, labels=(0,))
    exp_map = exp.as_map()
    #print(exp.as_map())
    convert_array =np.array(exp_map[0])
    list_weights = [i[1] for i in convert_array]
    oned_array_weights =np.array(list_weights)
    X_sig_scores = oned_array_weights[:, np.newaxis].T


    #print(X_sig_scores)
    for j in range(1,10):
        exp1 = explainer.explain_instance(X_test[j], model.predict_proba, num_features=16, labels=(0,))
        exp_map1 = exp1.as_map()
        convert_array1 = np.array(exp_map1[0])
        list_weights1 = [i[1] for i in convert_array1]
        oned_array_weights1 = np.array(list_weights1)
        X_sig_scores1 = oned_array_weights1[:, np.newaxis].T
        X_sig_scores = np.concatenate((X_sig_scores, X_sig_scores1), axis=0)

    return X_sig_scores

def get_lime_feature_sig_scores_lstm(model, X_train, X_test, y_train, feature_names):
    #If the feature is numerical, compute the mean and std, and discretize it into quartiles.

    explainer = lime.lime_tabular.RecurrentTabularExplainer(X_train,
                                                   training_labels=y_train,
                                                   feature_names = feature_names,
                                                   class_names = [0, 1],
                                                   discretize_continuous=True,
                                                   discretizer='decile')

    exp = explainer.explain_instance(X_test[0], model.predict, num_features=12, labels=(0,))
    exp_map = exp.as_map()
    #print(exp.local_exp)
    #print(exp.as_map())

    convert_array =np.array(exp_map[0])
    list_weights = [i[1] for i in convert_array]
    oned_array_weights =np.array(list_weights)
    X_sig_scores = oned_array_weights[np.newaxis, :]


    for j in range(1,10):
        exp1 = explainer.explain_instance(X_test[j], model.predict_proba, num_features=12, labels=(0,))
        exp_map1 = exp1.as_map()
        convert_array1 = np.array(exp_map1[0])
        list_weights1 = [i[1] for i in convert_array1]
        oned_array_weights1 = np.array(list_weights1)
        X_sig_scores1 = oned_array_weights1[np.newaxis, :]
        X_sig_scores = np.concatenate((X_sig_scores, X_sig_scores1), axis=0)

    print(X_sig_scores)
    print(X_sig_scores.shape)
    print(X_sig_scores.ndim)

    return X_sig_scores