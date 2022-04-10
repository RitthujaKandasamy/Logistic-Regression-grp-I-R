import numpy as np


class LogisticRegression():
    def __init__(self, X, learning_rate = 0.1, num_iters = 10000):
        self.lr = learning_rate
        self.num_iters = num_iters

     # m for training examples, n for features
        self.m, self.n = X.shape

    def train(self, X, y):
        