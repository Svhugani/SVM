import numpy as np
from matplotlib import pyplot as plt

class Kernel:
    def __init__(self, kernel_type="linear"):
        self.sigmoid_gamma = .1
        self.sigmoid_bias = 1
        self.rbf_gamma = None
        self.linear_bias = 0

        if kernel_type == "linear":
            self.kernel = self.__linear_kernel
        elif kernel_type == "rbf":
            self.kernel = self.__rbf_kernel
        elif kernel_type == "sigmoid":
            self.kernel = self.__sigmoid_kernel

    def __linear_kernel(self, samples_1, samples_2):
        return np.matmul(np.transpose(samples_1), samples_2) + self.linear_bias

    def __rbf_kernel(self, samples_1, samples_2):
        if self.rbf_gamma is None:
            gamma = 1.0 / samples_1.shape[1]
        return np.exp(-gamma * np.linalg.norm(samples_1 - samples_2) ** 2)

    def __sigmoid_kernel(self, samples_1, samples_2):
        return np.tanh(self.sigmoid_gamma * np.matmul(np.transpose(samples_1), samples_2) + self.sigmoid_bias)

    def set_custom_parameters(self, sigmoid_gamma = .1, sigmoid_bias = 1, rbf_gamma = None, linear_bias = 0):
        self.sigmoid_gamma = sigmoid_gamma
        self.sigmoid_bias = sigmoid_bias
        self.rbf_gamma = rbf_gamma
        self.linear_bias = linear_bias

    def evaluate(self, samples_1, samples_2):
        return self.kernel(samples_1, samples_2)


class SVM:
    def __init__(self, kernel, accuracy=1.0):

        self.accuracy = accuracy
        self.kernel = kernel
        self.u_weights = None
        self.bias = None
        self.n_samples = None
        self.n_features = None
        self.kernel_value = None
        self.data = None
        self.targets = None
        self.losses = None
        self.margins = None
        self.svcs = None

    def __loss_function(self):

        return 0.5 * self.u_weights.dot(self.kernel_value * self.u_weights) \
            + self.accuracy * np.sum(np.maximum(0, 1 - self.margins))

    def fit(self, data, targets, learning_rate=1e-4, n_iterations=300):

        self.n_samples, self.n_features = data.shape
        self.u_weights = np.random.randn(self.n_samples)
        self.bias = 0

        self.data = data
        self.targets = targets
        self.kernel_value = self.kernel.evaluate(data, data)

        self.losses = list()

        for i in range(n_iterations):
            self.margins = self.targets * (np.dot(self.u_weights, self.kernel_value) + self.bias)
            self.losses.append(self.__loss_function())

            impact_subset = np.where(self.margins < 1)
            u_weights_grad = np.dot(self.kernel_value, self.u_weights) \
                - self.accuracy * np.sum((self.targets * self.kernel_value)[impact_subset])
            bias_grad = - self.accuracy * np.sum(self.targets[impact_subset])
            self.u_weights -= learning_rate * u_weights_grad
            self.bias -= learning_rate * bias_grad

        self.svcs = np.where(self.margins <= 1)[0]

    def __prediction_function(self, data):

        return self.u_weights.dot(self.kernel.evaluate(data, data)) + self.bias

    def __predict_single_point(self, point):

        return self.u_weights.dot(self.kernel.evaluate(point, point)) + self.bias
    def predict(self, data):

        return np.sign(self.__prediction_function(data))

    def calculate_score(self, data, targets):

        return np.mean(self.predict(data) == targets)

    def classification_bound_2d(self, data=None, resolution=150):

        if data is None:
            data = self.data
        x_ticks = np.linspace(data[:, 0].min(), data[:, 0].max(), resolution)
        y_ticks = np.linspace(data[:, 1].min(), data[:, 1].max(), resolution)
        grid = np.array(np.meshgrid(x_ticks, y_ticks))
        classification = np.zeros_like(grid[0])
        for i, j in zip(range(resolution), range(resolution)):
            classification[i, j] = self.__predict_single_point(np.array([grid[0, i, j], grid[1, i, j]]))
        return grid, classification
