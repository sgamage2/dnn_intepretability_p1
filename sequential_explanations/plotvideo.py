import numpy as np
from matplotlib import pyplot as plt

x = np.load("D://first_DL//videoucftest//v_ApplyEyeMakeup_g01_c01-40-features.npy")
def plot_video(x):
    fig = plt.figure(figsize=(10, 5))
    axes = fig.subplots(5, 8)
    axes = axes.flatten()
    for i in range(0,x.shape[0]):
        axes[i].imshow(x[i])
    plt.show()
    return -1

plot_video(x)


