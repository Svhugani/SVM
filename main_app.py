import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn import datasets
from sklearn.model_selection import train_test_split
import svm
import datasets


def make_plot(X_train, X_test, y_train, y_test):
    plt.style.use('dark_background')
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='tab10', label='Train data')
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test,  cmap='Pastel1', label='Test data')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.grid(True, linestyle='--', alpha=.3)
    plt.show()

X_train, X_test, y_train, y_test = datasets.get_blobs(skew_factor=.2)
kernel = svm.Kernel(kernel_type="rbf")
classifier = svm.SVM(kernel=kernel, accuracy=1)
classifier.fit(data=X_train, targets=y_train)
res_train = classifier.calculate_score(X_train, y_train)
res_test = classifier.calculate_score(X_test, y_test)

print(f"Train results: {res_train}")
print(f"Test results: {res_test}")