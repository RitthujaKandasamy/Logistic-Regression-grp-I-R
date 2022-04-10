import numpy as np


class LogisticRegression():
    def __init__(self, X, learning_rate = 0.1, num_iters = 10000):
        self.lr = learning_rate
        self.num_iters = num_iters

     # m for training examples, n for features
        self.m, self.n = X.shape

    def train(self, X, y):
        self.weights = np.zeros((self.n, 1))
        self.bias = 0

        for i in range(self.num_iters+1):
            # calculate hypothesis
            y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)

            # calculate cost
            cost = -1/self.m * np.sum(y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict))

            # backprop
            dw = 1/self.m * np.dot(X.T, (y_predict - y))
            db = 1/self.m * np.sum(y_predict - y)

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

            if i % 1000 == 0:
                print(f'Cost after iteration {i}: {cost}')
              
        return self.weights, self.bias

    
    def predict