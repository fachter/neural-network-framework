from typing import List

import numpy as np
from abc import ABC, abstractmethod


class Optimizer(ABC):
    @abstractmethod
    def get_updated_weights(self, gradients, index):
        pass


class AdamOptimizer(Optimizer):
    # check https://ruder.io/optimizing-gradient-descent/index.html#adam for formulas
    def __init__(self, initial_thetas: List[np.ndarray], learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m = []
        self.v = []
        self.thetas = []
        for theta in initial_thetas:
            self.m.append([0] * theta)
            self.v.append([0] * theta)
            self.thetas.append(theta.copy())
        self.epsilon = epsilon

        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2

    def get_updated_weights(self, gradients, index):
        m = self.beta1 * self.m[index] + (1 - self.beta1) * gradients
        v = self.beta2 * self.v[index] + (1 - self.beta2) * gradients ** 2
        m_corrected = m / (1 - self.beta1)
        v_corrected = v / (1 - self.beta2)

        self.thetas[index] = self.thetas[index] - (self.learning_rate * m_corrected) / (np.sqrt(v_corrected) + self.epsilon)
        return self.thetas[index]
