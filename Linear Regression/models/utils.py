import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def loadData():
    x = pd.read_csv("../data/linearX.csv")
    y = pd.read_csv("../data/linearY.csv")
    return x, y


class MinMaxScaler:
    def __init__(self):
        pass

    def fitTransform(self, x):
        xMin, xMax = np.min(x), np.max(x)
        xNorm = (x - xMin) / (xMax - xMin)
        return xNorm


class LinearRegression:
    def __init__(self):
        self.w = self.b = 0
        self.costs = []

    def modelFunc(self, x):
        yHat = np.dot(self.w, x) + self.b
        return yHat

    def lossFunc(self, x, y):
        yHat = self.modelFunc(x)
        l = np.square(np.subtract(yHat, y))
        return l

    def costFunc(self, x, y):
        l = self.lossFunc(x, y)
        J_wb = (1 / 2) * np.mean(l)
        return J_wb

    def error(self, x, y):
        yHat = self.modelFunc(x)
        e = np.subtract(yHat, y)
        return e

    def gradient(self, x, y):
        e = self.error(x, y)
        dJ_dw = np.mean(e * x)
        dJ_db = np.mean(e)
        return dJ_dw, dJ_db

    def fit(self, x, y, epochs, learning_rate, shuffle=False, batch_size=None):
        x = x.to_numpy()
        y = y.to_numpy()
        a, bat = learning_rate, batch_size
        for _ in range(epochs):
            if shuffle:
                rndIds = np.random.permutation(x.shape[0])
                x, y = x[rndIds], y[rndIds]
            m = x.shape[0]
            for i in range(0, m, m or bat):
                if batch_size is not None and shuffle:
                    xNew, yNew = x[i:min(i + bat, m)], y[i:min(i + bat, m)]
                elif shuffle:
                    xNew, yNew = x[i:i + 1], y[i:i + 1]
                else:
                    xNew, yNew = x.copy(), y.copy()
                dJ_dw, dJ_db = self.gradient(xNew, yNew)
                self.w -= a * dJ_dw
                self.b -= a * dJ_db
            currCost = self.costFunc(x, y)
            self.costs.append(currCost)

    def fit(self, x, y, epochs, learning_rate, min_delta=0.0001, max_iter=100, shuffle=False, batch_size=None, patience=5):
        x = x.to_numpy()
        y = y.to_numpy()
        a, bat = learning_rate, batch_size
        bestCost = math.inf
        patienceCnt = 0
        for _ in range(epochs):
            if shuffle:
                rndIds = np.random.permutation(x.shape[0])
                x, y = x[rndIds], y[rndIds]
            iterCnt = 0
            m = x.shape[0]
            for i in range(0, m, m or bat):
                if shuffle:
                    x, y = x[i:i + 1], y[i:i + 1]
                elif batch_size is not None:
                    x, y = x[i:i + bat], y[i:i + bat]
                if max_iter is not None and iterCnt >= max_iter:
                    break
                dJ_dw, dJ_db = self.gradient(x, y)
                self.w -= a * dJ_dw
                self.b -= a * dJ_db
                iterCnt += 1
            currCost = self.costFunc(x, y)
            self.costs.append(currCost)
            if patience is not None:
                if currCost < bestCost - min_delta:
                    bestCost = currCost
                    patienceCnt = 0
                else:
                    patienceCnt += 1
                if patienceCnt >= patience:
                    break
            if max_iter is not None and iterCnt >= max_iter:
                break

    def predict(self, x_i):
        return self.modelFunc(x_i)

    def plotCost(self, subtitle):
        fig, ax = plt.subplots()
        ax.set_title("Cost vs Epoch")
        fig.suptitle(subtitle)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cost")
        ax.plot(list(range(1, len(self.costs) + 1)), self.costs)
        plt.show()

    def scatterPlot(self, x, y, subtitle):
        fig, ax = plt.subplots()
        ax.set_title("Linear Regression Fit on Dataset")
        fig.suptitle(subtitle)
        ax.set_xlabel("Input Features (x)")
        ax.set_ylabel("Output Labels (y)")
        ax.scatter(x, y, label="Data points")
        ax.plot(x, self.predict(x), color="red", label="Best Fit line")
        ax.legend()
        plt.show()
