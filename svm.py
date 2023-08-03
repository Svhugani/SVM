import numpy as np

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
        else:
            self.kernel = self.__linear_kernel

    def __linear_kernel(self, samples_1, samples_2):
        #return np.matmul(samples_1, np.transpose(samples_2)) + self.linear_bias
        return samples_1.dot(samples_2.T) + self.linear_bias

    def __rbf_kernel(self, samples_1, samples_2):
        if self.rbf_gamma is None:
            gamma = 1.0 / samples_1.shape[1]
        #return np.exp(-gamma * np.linalg.norm(samples_1 - samples_2) ** 2)
        return np.exp(-gamma * np.linalg.norm(samples_1[:, np.newaxis] - samples_2[np.newaxis, :], axis=2) ** 2)

    def __sigmoid_kernel(self, samples_1, samples_2):
        return np.tanh(self.sigmoid_gamma * np.matmul(samples_1, np.transpose(samples_2)) + self.sigmoid_bias)

    def set_custom_parameters(self, sigmoid_gamma=.1, sigmoid_bias=1, rbf_gamma=None, linear_bias=0):
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
        self.data = None
        self.targets = None
        self.losses = None
        self.svcs_idx = None

    def __loss_function(self, kernel_value, margins):
        return 0.5 * self.u_weights.dot(kernel_value.dot(self.u_weights)) \
            + self.accuracy * np.maximum(0, 1 - margins).sum()

    def fit(self, data, targets, learning_rate=1e-5, n_iterations=500):

        self.n_samples, self.n_features = data.shape
        self.u_weights = np.random.randn(self.n_samples)
        self.bias = 0
        self.losses = list()
        self.data = data
        self.targets = targets
        kernel_value = self.kernel.evaluate(data, data)

        for i in range(n_iterations):
            margins = self.targets * (self.u_weights.dot(kernel_value) + self.bias)
            self.losses.append(self.__loss_function(kernel_value, margins))

            impact_subset = np.where(margins < 1)[0]

            u_weights_grad = kernel_value.dot(self.u_weights) \
                - self.accuracy * self.targets[impact_subset].dot(kernel_value[impact_subset])
            bias_grad = - self.accuracy * np.sum(self.targets[impact_subset])
            self.u_weights -= learning_rate * u_weights_grad
            self.bias -= learning_rate * bias_grad

        self.svcs_idx = np.where((self.targets * (self.u_weights.dot(kernel_value) + self.bias)) <= 1)[0]

    def __prediction_function(self, data):
        return self.u_weights.dot(self.kernel.evaluate(self.data, data)) + self.bias

    def predict(self, data):
        return np.sign(self.__prediction_function(data))

    def calculate_score(self, data, targets):
        prediction = self.predict(data)
        return np.mean(prediction == targets)

    def classification_bound_2d(self, data=None, resolution=50, bound_mul=1.5):
        if data is None:
            data = self.data
        x_ticks = np.linspace(data[:, 0].min() * bound_mul, data[:, 0].max() * bound_mul, resolution)
        y_ticks = np.linspace(data[:, 1].min() * bound_mul, data[:, 1].max() * bound_mul, resolution)
        grid = np.array([[x, y] for x in x_ticks for y in y_ticks])
        classification = self.__prediction_function(grid)
        classification = np.array(classification).reshape(resolution, resolution)

        return x_ticks, y_ticks, classification
