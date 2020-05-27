import numpy as np


def get_random_feature_sig_scores(X):
    """ Sets a random feature significance score (from standard normal distribution) for a given set of examples

    Useful as a baseline to evaluate other feature significance methods

    :param X: set of examples to compute feature significance scores on. Has shape: (num_samples, num_features)
    :return: random_sig: random significance of every feature of every example in X (array of the same shape as X)
    """

    random_sig = np.zeros(shape=X.shape)

    # Generate random sig for every example separately to ensure the random normal distribution applies to each example
    for i in range(random_sig.shape[0]):
        random_sig[i] = np.random.normal(size=random_sig.shape[1])

    return random_sig


def get_random_feature_sig_scores_lstm(X):

    random_sig = np.zeros(shape=X.shape)

    # Generate random sig for every example separately to ensure the random normal distribution applies to each example
    for i in range(random_sig.shape[0]):
        random_sig[i] = np.random.normal(size=(random_sig.shape[1], random_sig.shape[2]))

    return random_sig

if __name__ == '__main__':
    assert False    # Not meant to be run as a standalone script
