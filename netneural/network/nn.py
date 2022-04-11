import os
from math import ceil
from random import shuffle

import numpy as np
import pandas as pd
from typing import List
import mnist
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

import netneural.network.feature_scaling as fs
from .one_hot_encoder import OneHotEncoder
from .plot_lib import plot_line
from ..optimizers.adam_optimizer import AdamOptimizer
from .metrics import get_f1_score, get_accuracy, accuracy_softmax


class ActivationFunctions:
    @staticmethod
    def sigmoid(z, derivation=False):
        if derivation:
            output = ActivationFunctions.sigmoid(z)  # first computes the sigmoid
            return output * (1 - output)
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def tanh(z, derivation=False):
        if derivation:
            return 1 - np.tanh(z) ** 2
        return np.tanh(z)

    @staticmethod
    def relu(z, derivation=False):
        if derivation:
            return (z > 0).astype(float)
        return np.maximum(0, z)


class NeuralNetwork:
    def __init__(self, layers_nn, activation_function='sigmoid', weight_matrices: List[np.ndarray] = None,
                 encoder: OneHotEncoder = None, scaler=None, scaler_name='standard', regression=False):
        """
        Initialize the neural network by configuring the input dimension, ...
        The list of hidden layers contains the output layer, by default it contains a layer with 3 nodes and
        an output layer with dimension 1
        :param layers_nn: number of nodes for hidden layers / shape of the network
        :param activation_function: activation function is sigmoid by default
        :param weight_matrices: preset weights, to recreate the same network (will be initialized randomly if None)
        :param scaler: default is 'standard' for Standard Scaler, can also choose 'normal' for Normal Scaler
        :param regression: set to True if neural network is needed for regression
        """
        self.optimizer = None
        self.layer_count = len(layers_nn) - 1
        self.shape = layers_nn

        self._layer_input = []
        self._layer_output = []

        self.activation_function = getattr(ActivationFunctions, activation_function)
        if scaler is None:
            self.scaler_name = scaler_name
            if scaler_name == 'standard':
                self.scaler = fs.StandardScaler()
            elif scaler_name == 'normal':
                self.scaler = fs.NormalScaler()
        else:
            self.scaler = scaler

        self.encoder = encoder
        self.regression = regression

        if weight_matrices is None:
            self.weights = []
            for (l1, l2) in zip(layers_nn[:-1], layers_nn[1:]):  # instantiate random weights
                self.weights.append(np.random.uniform(low=-1.0, high=1.0, size=(l2, l1 + 1)))
        else:
            self.weights = weight_matrices
        self.init_weights = []
        for weight in self.weights:
            self.init_weights.append(weight.copy())
        print('Neural Network created')

    @staticmethod
    def add_bias_column(x):
        return np.c_[np.ones(x.shape[0]), x]

    @staticmethod
    def add_bias_row(x):
        return np.r_[np.ones(shape=(1, x.shape[1])), x]

    def forward_pass(self, input_data: np.ndarray):
        """
        Updates all layers prediction and activated predictions.
        :param input_data: input values
        :return: Predicted Values for every input data point
        """
        if len(input_data.shape) == 1:  # only one input data point
            input_data = input_data.reshape(1, len(input_data))  # matrix with one row
        # initialize or cleaned
        self._layer_input = []
        self._layer_output = []
        for index in range(self.layer_count):
            if index == 0:  # first layer
                self._layer_input.append(self.weights[index] @ self.add_bias_column(input_data).T)
            else:  # every other layer
                output_with_bias = self.add_bias_row(self._layer_output[-1])  # add bias
                self._layer_input.append(self.weights[index] @ output_with_bias)  # compute not activated output
            if index == self.layer_count - 1 and self.shape[-1] != 1:  # last layer and not binary classification
                if self.shape[-1] > 1:  # multiclass
                    self._layer_output.append(self.softmax(self._layer_input[-1]))
                elif self.regression:  # regression
                    self._layer_output.append(self.linear_activation(self._layer_input[-1]))  # linear 'activation'
            else:
                self._layer_output.append(self.activation_function(self._layer_input[-1]))  # activation
        return self._layer_output[-1]

    def get_error(self, predictions: np.ndarray, observed_vals):
        """
        Compute error for every made prediction
        :param predictions: made predictions
        :param observed_vals: observed y values
        :return: error for every data point
        """
        if self.shape[-1] > 1:  # multiclass classification
            m = observed_vals.shape[0]
            return (1 / m) * self.categorical_cross_entropy(predictions, observed_vals)

        if self.regression:  # mean squared error
            return self.mse(predictions, observed_vals)

        predictions = np.clip(predictions, 1e-9, 1 - 1e-9)
        errors = self.cross_entropy(predictions, observed_vals)
        return errors

    @staticmethod
    def mse(h, y, derivation=False):
        if derivation:
            return h-y
        return np.mean((h - y) ** 2) / 2

    @staticmethod
    def linear_activation(z, derivation=False):
        if derivation:
            return 1
        return z

    @staticmethod
    def cross_entropy(o, y):
        return -y * np.log(o) - (1 - y) * np.log(1 - o)

    @staticmethod
    def categorical_cross_entropy(h, y):
        h = np.clip(h, a_min=1e-9, a_max=None)
        return -np.sum(y * np.log(h), axis=1)

    @staticmethod
    def softmax(x):
        x = x - np.max(x)
        e_x = np.e ** x
        return e_x / ((np.sum(e_x, axis=0, keepdims=True)) + 1e-9)

    @staticmethod
    def softmax_der(x):
        pass

    def train(self, training_data, target_vals, iterations: int, learning_rate: float, batch_size=None,
              test_data=None, test_target=None, optimizer: str = None, plots: bool = False, encode=False):
        """
        Train the neural network
        :param batch_size: number of samples trained at the same time in one epoch
        :param plots: if set to True metrics will be visualized after training
        :param optimizer: choice between default gradient descent or 'adam'
        :param training_data: input data, the columns should represent features
        :param target_vals: target values, the rows should represent different classes
        :param iterations: number of iterations the network is going to be trained for
        :param learning_rate: learning rate for the network
        :param test_data: test data not used for training, but only for plotting
        :param test_target: target values for test data
        :return: the training history for f1 score, accuracy, error history on the training set if available, otherwise
                    on the training set
        """
        print('Training the Network')
        error_history = list()
        f1_score_history = list()
        f1_score_per_class_history = list()
        accuracy_per_class_history = list()
        test_acc_history = list()
        test_f1_history = list()
        test_error_history = list()
        accuracy_history = list()
        multiclass = False
        if optimizer == "adam":
            print('Using Adam Optimizer ')
            self.optimizer = AdamOptimizer(initial_thetas=self.weights, learning_rate=learning_rate)
        if batch_size is not None:
            print('Batchsize:', batch_size)

        if self.shape[-1] > 1:  # multiclass classification
            multiclass = True
            if encode:
                if self.encoder is None:  # encoder was not used yet for encoding
                    self.encoder = OneHotEncoder()
                target_vals = self.encoder.encode(target_vals).T  # encode
                if test_target is not None:
                    test_target = self.encoder.encode(test_target).T

        if len(target_vals.shape) == 1:
            target_vals = target_vals.reshape(1, len(target_vals))

        t = tqdm(range(iterations))
        for i in t:
            errors, predictions = self.train_epoch(training_data, ground_truth=target_vals, alpha=learning_rate,
                                                   batch_size=batch_size)
            test_f1 = None
            if self.shape[-1] > 1:  # multiclass classification
                accuracy_per_class, accuracy = get_accuracy(predictions.T, target_vals.T)
                f1_score_per_class, f1_score = get_f1_score(predictions.T, target_vals.T)
                f1_score_per_class_history.append(f1_score_per_class)
                # accuracy_per_class_history.append(accuracy_per_class)
            elif not self.regression:  # binary classification
                accuracy = get_accuracy(predictions, target_vals)
                f1_score = get_f1_score(predictions, target_vals)

            # get values if test data available
            if test_data is not None and test_target is not None:
                test_predictions = self.forward_pass(test_data).T
                if not self.regression:
                    test_acc, test_f1 = self.compute_metrics(test_predictions, test_target, multiclass)
                    test_acc_history.append(test_acc)
                    test_f1_history.append(test_f1)
                # add metrics for test data to history
                errors_test = self.get_error(test_predictions.T, test_target)
                error_test = errors_test if self.regression else errors_test.mean()
                test_error_history.append(error_test)

            # add metrics from training data to history
            error = errors if self.regression else errors.mean()
            error_history.append(error)
            if not self.regression:
                f1_score_history.append(f1_score)
                accuracy_history.append(accuracy)

            # visualization with tqdm in command line
            if test_data is not None and test_target is not None:
                if self.regression:
                    t.set_postfix({'MSE': error_test}, refresh=True)
                else:
                    t.set_postfix({'F1 Test': test_f1, 'F1 Training': f1_score}, refresh=True)
            else:
                if self.regression:
                    t.set_postfix({'MSE': error}, refresh=True)
                else:
                    t.set_postfix({'F1 Score': f1_score}, refresh=True)

        if plots and not self.regression:
            plot_line(error_history, "iteration", "Error", "Error on Training Data")
            plot_line(accuracy_history, "iteration", "Accuracy", "Accuracy on Training Data")
            plot_line(f1_score_history, "iteration", "F1 score", "F1 Score on Training Data")
            if test_data is not None and test_target is not None:
                plot_line(test_acc_history, "iteration", "Accuracy", "Accuracy on Test Data")
                plot_line(test_f1_history, "iteration", "F1 Score", "F1 Score on Test Data")
                plot_line(test_error_history, "iteration", "Error", "Error on Test Data")
        elif plots:
            plot_line(error_history, "iteration", "Error", "Error on Training Data")
            if test_data is not None and test_target is not None:
                plot_line(test_error_history, "iteration", "Error", "Error on Test Data")

        print('Training of the Neural Network Done')
        if not self.regression:
            print(f'F1 Score (Training Set): {f1_score_history[-1]}')
        else:
            print(f'MSE (Training Set): {error_history[-1]}')
        if test_data is not None and test_target is not None:
            if not self.regression:
                print(f'F1 Score (Test Set): {test_f1_history[-1]}')
                return test_f1_history, test_acc_history, test_error_history
            else:
                print(f'MSE (Test Set): {test_error_history[-1]}')
                return test_error_history
        else:
            if not self.regression:
                return f1_score_history, accuracy_history, error_history
            else:
                return error_history

    def compute_metrics(self, h, y, multiclass):
        if multiclass:
            test_acc = accuracy_softmax(h, y.T)[1]
            test_f1 = get_f1_score(h, y.T, multiclass)[1]
        else:
            test_acc = get_accuracy(h.T, y)
            test_f1 = get_f1_score(h.T, y)
        return test_acc, test_f1

    def predict(self, input_data, with_confidence=False):
        """
        Predicts an output using the input data.
        :param input_data: input for the model, one row equals one data point
        :param with_confidence: if set to true will return confidence of max predicted label
        :return: predictions made based on the input data
        """
        predictions = self.forward_pass(input_data)
        one_ex = False if len(input_data.shape) == 2 else True

        if self.shape[-1] > 1:  # multiclass
            result = self.encoder.decode(predictions.T)
            if with_confidence:
                arg_max = np.argmax(predictions.T, axis=1).astype(int)
                confidence = np.choose(arg_max, predictions)
                result = (np.c_[result, confidence])
            return result[0] if one_ex else result
        elif self.regression:
            return predictions[0, 0] if one_ex else predictions
        elif self.activation_function == ActivationFunctions.sigmoid:
            result = predictions.round()
            return result[0, 0] if one_ex else result
        elif self.activation_function == ActivationFunctions.tanh:
            predictions[predictions < 0] = 0
        predictions[predictions > 0] = 1
        return predictions[0, 0] if one_ex else predictions

    @staticmethod
    def accuracy(predictions, target_vals):
        h = predictions
        accuracy = (np.rint(h) == target_vals).mean()
        return accuracy

    @staticmethod
    def accuracy_tanh(predictions, target_vals):
        h = predictions
        h[h > 0] = 1
        h[h < 0] = 0
        accuracy_tanh = (h == target_vals).astype(int).mean()
        return accuracy_tanh

    @staticmethod
    def accuracy_softmax(predictions, target_vals):
        labels_validation = np.argmax(target_vals, axis=0)
        h = np.argmax(predictions, axis=0)
        confusion_matrix = np.zeros((10, 10))
        for i in range(len(predictions)):
            confusion_matrix[labels_validation[i] - 1][h[i] - 1] += 1
        per_class_accuracy = np.diag(confusion_matrix) / (confusion_matrix.sum(axis=1) + 1e-9)
        acc = (per_class_accuracy * 100).round(2)
        total_acc = (per_class_accuracy.mean() * 100).round(2)
        return acc, total_acc

    def train_epoch(self, training_data: np.ndarray, ground_truth: np.ndarray, alpha, batch_size):
        """
        One training epoch of the model. Computes the predicted values and uses them to compute delta values, which are
        equal to the product of the node error up until the specific node. The deltas are used to update the weights
        afterwards.
        :param batch_size: size of the batch
        :param training_data: input of the neural net
        :param ground_truth: training validation values
        :param alpha: learning rate
        :return: error after this iteration
        """
        if batch_size is None:
            batch_size = training_data.shape[0]
        batches = ceil(training_data.shape[0] / batch_size)
        shuffled_indices = list(range(training_data.shape[0]))
        shuffle(shuffled_indices)
        prev_batch = 0
        for batch in range(1, batches+1):
            next_batch = batch_size * batch
            indices_to_use = shuffled_indices[prev_batch:next_batch]
            batch_training_data = training_data[indices_to_use]
            predictions = self.forward_pass(batch_training_data)
            batch_ground_truth = ground_truth[:, indices_to_use]
            deltas = self.compute_deltas(predictions, batch_ground_truth)
            self.update_weights(batch_training_data, deltas, alpha)
            prev_batch = next_batch
        predictions_after = self.forward_pass(training_data)
        errors = self.get_error(predictions_after, ground_truth)
        return errors, predictions_after

    def compute_deltas(self, predictions: np.ndarray, ground_truth: np.ndarray):
        """
        Compute the error of each node and multiply it with the previous node errors. These values are stored for
        the current node.
        :param predictions: output of the neural net
        :param ground_truth: training validation values
        :return: the computed delta values as a list
        """
        deltas = []
        for index in reversed(range(self.layer_count)):  # start from output layer
            if index == self.layer_count - 1:  # last layer
                error_der = self.get_error_derivative(predictions, ground_truth)
                deltas.append(error_der.T)
            else:
                output_delta = deltas[-1] @ self.weights[index + 1][:, 1:]  # dz_da
                activation_derivation = self.activation_function(self._layer_input[index], derivation=True).T
                deltas.append(output_delta * activation_derivation)  # compute neuron error for every neuron in layer
        deltas.reverse()  # because they were added starting from the output layer
        return deltas

    def get_error_derivative(self, predictions: np.ndarray, ground_truth: np.ndarray):
        """
        Compute the partial derivative of the error function for the output value and the partial
        derivative of the output value to the unactivated value in the output node multiplied.
        :param predictions: output of the neural net
        :param ground_truth: training validation values
        :return: the error of the output node
        """
        if self.shape[-1] > 1 or self.regression:
            return predictions - ground_truth  # multiclass

        predictions[predictions == 0] = 0.000001
        predictions[predictions == 1] = 0.999999
        dJ_do = (-(ground_truth / predictions)) + ((1 - ground_truth) / (1 - predictions))  # deriv. of cost function
        do_dz = self.activation_function(self._layer_input[-1], derivation=True)  # derivative of activation function
        return dJ_do * do_dz

    def update_weights(self, training_data: np.ndarray, deltas, alpha):
        """
        Update the weights according to the computed delta values.
        :param training_data: input values of the neural net
        :param deltas: error value for every node, computed until this node
        :param alpha: learning rate
        :return:
        """
        # delta.shape = (number of entries, number of neurons)
        for index in range(self.layer_count):
            if index == 0:
                layer_output = self.add_bias_column(training_data)
            else:
                layer_output = self.add_bias_row(self._layer_output[index - 1]).T
            # transposed deltas contain one row for each neuron (excluding bias) and one col for each sample
            # layer_output contain one row for each sample
            # and one col for each neuron of previous layer (including bias)
            # normalize by number of samples
            weights_delta = (deltas[index].T @ layer_output) / layer_output.shape[0]
            if self.optimizer is not None:
                self.weights[index] = self.optimizer.get_updated_weights(weights_delta, index)
            else:
                self.weights[index] -= alpha * weights_delta


def xor_main():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0.05, 0.95, 0.95, 0.05])
    nn = NeuralNetwork(layers_nn=(2, 2, 1))
    for i in range(10000):
        nn.train_epoch(X, y)
    print(nn.forward_pass(X))
    print(nn.weights)


def regression():
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([0.05, 0.95, 0.95, 0.05])
    nn = NeuralNetwork(layers_nn=(2, 2, 1), activation_function='relu', regression=True)
    nn.train(X, y, 10, 0.1)
    print(nn.predict(X[0]))


def pizza_main():
    # load the data from the Logistic Regression module
    delivery_data = pd.read_csv("../neural_net/delivery_data.csv")
    X_delivery = delivery_data[["motivation", "distance"]].to_numpy()
    y_delivery = delivery_data["delivery?"].to_numpy()
    scaler = fs.StandardScaler()
    scaler.fit(X_delivery)
    X = scaler.transform(X_delivery)
    print(X.shape)
    # nn = NeuralNetwork((2, 3, 2, 1), activation_function='relu')
    # nn = NeuralNetwork((2, 2, 2, 1), activation_function='tanh',
    #                   weight_matrices=[np.array([[2, -1, 0.67], [-3, 1, -0.67]]),
    #                                    np.array([[1, 1, 1], [-4, -0.33, 0.67]]),
    #                                    np.array([[0.5, 0.67, -1.3]])])
    nn = NeuralNetwork((2, 3, 3, 1), activation_function='sigmoid')
    nn.train(X, y_delivery, 10000, 0.001)

    h = nn.forward_pass(X)
    h[h > 0] = 1
    h[h < 0] = 0

    print(h)
    print(nn.weights)
    accuracy = (np.rint(h) == y_delivery).mean()
    # accuracy_tanh = (h == y_delivery).astype(int).mean()
    print(accuracy)
    # print(accuracy_tanh)


def mnist_main():
    mnist_folder_exists = os.path.isdir('mnist')
    if not mnist_folder_exists:
        import mnist_downloader
        download_folder = "./mnist/"
        mnist_downloader.download_and_unzip(download_folder)
    mnist_data = mnist.MNIST('mnist', return_type="numpy")
    images_train, labels_train = mnist_data.load_training()
    images_validation, labels_validation = mnist_data.load_testing()
    encoder = OneHotEncoder()
    y = encoder.encode(labels_train).T
    neural_network = NeuralNetwork((784, 20, 20, 10), encoder=encoder)
    neural_network.scaler.fit(images_train)
    X = neural_network.scaler.transform(images_train)

    neural_network.train(X, y, iterations=2, learning_rate=3)

    X_validation = neural_network.scaler.transform(images_validation)
    # predictions = np.argmax(neural_network.forward_pass(X_validation), axis=0)
    predictions = neural_network.predict(X_validation, True)
    # Usage of confidences
    confidences = predictions[:, 1]
    print(confidences)
    predictions = predictions[:, 0].astype(int)
    print(predictions)

    confusion_matrix = np.zeros((10, 10))
    for i in range(len(predictions)):
        confusion_matrix[labels_validation[i] - 1][predictions[i] - 1] += 1
    sns.heatmap(confusion_matrix)
    plt.show()
    per_class_accuracy = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)
    print((per_class_accuracy * 100).round(2))
    print((per_class_accuracy.mean() * 100).round(2))


if __name__ == '__main__':
    regression()
    # mnist_main()
    # mnist_main()
    # pizza_main()
    # data
    # X = np.array([[1, -2], [1, -2], [1, -2]])
    # Y = np.array([0, 0, 0])
    # # weights
    # thetas_B = np.array([[2, -1, 0.67], [-3, 1, -0.67]])
    # thetas_C = np.array([[1, 1, 1], [-4, -0.33, 0.67]])
    # thetas_D = np.array([[0.5, 0.67, -1.3]])
    # weight_matrices = [thetas_B, thetas_C, thetas_D]
    #
    # nn = NeuralNetwork(layers_nn=(2, 2, 2, 1), weight_matrices=weight_matrices)
    # # nn.forward_pass(X)
    # nn.train_epoch(X, Y)
    # a = 1
    # nn.train(X, Y, 10, 0.1)
    #

    # d_tX10 = deltaZB1 * 1
    # d_tX11 = deltaZB1 * x1
    # d_tX12 = deltaZB1 * x2
    # zB1 = 1 * thetaX10 + x1 * thetaX11 + x2 * thetaX12
