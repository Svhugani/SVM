from matplotlib import pyplot as plt
from matplotlib import cm as cm
import numpy as np

plt.style.use('dark_background')
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8

def plot_svm_classification(X, y, x_ticks, y_ticks, classification_grid=None, support_vectors=None):
    fig, ax = plt.subplots()
    cmap = cm.get_cmap('tab10')
    unique_labels = np.unique(y)
    labels_count = unique_labels.shape[0]
    for i, label in zip(range(labels_count), unique_labels):
        condition = (y == label)
        x_values = X[condition, 0]
        y_values = X[condition, 1]
        plt.scatter(
            x_values,
            y_values,
            color=cmap(float(i) / labels_count),
            label=f'Train data - Class {i}',
            alpha=0.65
        )

    if classification_grid is not None:
        contour = ax.contour(
            x_ticks,
            y_ticks,
            classification_grid,
            (-1, 0, 1),
            linewidths=(1, 2, 1),
            linestyles=("--", "-", "--"),
            colors=['pink', 'pink', 'pink']
        )
        ax.clabel(contour, inline=True, fontsize=7)
    if support_vectors is not None:
        ax.scatter(
            X[support_vectors, 0],
            X[support_vectors, 1],
            marker='o',
            facecolors='none',
            edgecolors='pink',
            label='Support vectors',
        )
    plt.title("SVM CLASSIFICATION")
    plt.xlabel('Feature x')
    plt.ylabel('Feature y')
    plt.grid(True, linestyle='--', alpha=.3)
    plt.legend()
    return fig, ax


def plot_learning_curve(losses):
    fig, ax = plt.subplots()
    ax.plot(losses)
    plt.title("LOSS VALUE EVOLUTION")
    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')
    plt.grid(True, linestyle='--', alpha=.3)
    return fig, ax