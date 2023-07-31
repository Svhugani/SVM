from sklearn.utils import shuffle
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np

def get_blobs(n_samples=200, centers=2, cluster_std=3, skew_factor=0):
    X, y = make_blobs(n_samples=n_samples, centers=centers, n_features=2, cluster_std=cluster_std)
    if skew_factor != 0:
        X = skew(X, skew_factor)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)
    return X_train, X_test, y_train, y_test

def skew(X, s=.2):
    rot = np.array([[1 + s, s], [0, 1]])
    return np.dot(X, rot)

if __name__ == '__main__':
    plt.style.use('dark_background')
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    X_train, X_test, y_train, y_test = get_blobs(skew_factor=.2)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='tab10', label='Train data')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test,  cmap='Pastel1', label='Test data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, linestyle='--', alpha=.3)
    plt.show()