import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

class LogisticRegression():
    def __init__(self, X, learning_rate = 0.1, num_iters = 10000):
        self.lr = learning_rate
        self.num_iters = num_iters

     # m for training examples, n for features
        self.m, self.n = X.shape

    def train(self, X, y):
        self.weights = np.zeros((self.n, 768))
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

    
    def predict(self, X):
        y_predict = self.sigmoid(np.dot(X, self.weights) + self.bias)
        y_predict_labels = y_predict > 0.5

        return y_predict_labels

    def accuracy(self, y, y_hat):
        accuracy = np.sum(y == y_hat) / X.shape[0]

        return accuracy


    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


if __name__ == '__main__':
    # np.random.seed(1)
    # X, y = make_blobs(n_samples=1000, centers=2)
    # y = y[:, np.newaxis]

    data = pd.read_csv("C:\\Users\\ritth\\code\\Strive\\Logistic-Regression-grp-I-R\\diabetes.csv")
    X = data.iloc[:, [0, 1, 2, 3, 4, 5, 6, 7]].values  
    y = data.iloc[:, 8].values  

    logreg = LogisticRegression(X)
    w, b = logreg.train(X, y)
    y_predict = logreg.predict(X)
    acc = logreg.accuracy(y, y_predict)
    

    print(f'Accuracy: {acc}')