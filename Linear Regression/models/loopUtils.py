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

    def modelFunc(self, x_i):
        yHat_i = self.w * x_i + self.b
        return yHat_i

    def lossFunc(self, x_i, y_i):
        yHat_i = self.modelFunc(x_i)
        l_i = (yHat_i - y_i) ** 2
        return l_i

    def costFunc(self, x, y):
        m = x.shape[0]
        J_wb = 0
        for i in range(m):
            l_i = self.lossFunc(x[i], y[i])
            J_wb += l_i
        J_wb *= 1 / (2 * m)
        return J_wb

    def error(self, x_i, y_i):
        yHat_i = self.modelFunc(x_i)
        e_i = yHat_i - y_i
        return e_i

    def gradient(self, x, y):
        m = x.shape[0]
        e, ex = 0, 0
        for i in range(m):
            e_i = self.error(x[i], y[i])
            e += e_i
            e_i_x_i = e_i * x[i]
            ex += e_i_x_i
        dJ_dw = (1 / m) * ex
        dJ_db = (1 / m) * e
        return dJ_dw, dJ_db

    def fit(self, x, y, epochs, learning_rate):
        x = x.to_numpy()
        y = y.to_numpy()
        a = learning_rate
        for _ in range(epochs):
            dJ_dw, dJ_db = self.gradient(x, y)
            self.w -= a * dJ_dw
            self.b -= a * dJ_db
            self.costs.append(self.costFunc(x, y))

    def predict(self, x_i):
        return self.modelFunc(x_i)

    def sgdFit(self, x, y, epochs, learning_rate):
        x = x.to_numpy()
        y = y.to_numpy()
        a = learning_rate
        m = x.shape[0]
        for _ in range(epochs):
            rndIds = np.random.permutation(m)
            x, y = x[rndIds], y[rndIds]
            for i in range(m):
                x_i, y_i = x[i:i + 1], y[i:i + 1]
                dJ_dw_i, dJ_db_i = self.gradient(x_i, y_i)
                self.w -= a * dJ_dw_i
                self.b -= a * dJ_db_i
            self.costs.append(self.costFunc(x, y))

    def mbgdFit(self, x, y, epochs, batch_size, learning_rate):
        x = x.to_numpy()
        y = y.to_numpy()
        a, bat = learning_rate, batch_size
        m = x.shape[0]
        for _ in range(epochs):
            rndIds = np.random.permutation(m)
            x, y = x[rndIds], y[rndIds]
            for i in range(0, m, bat):
                xBat, yBat = x[i:i + bat], y[i:i + bat]
                dJ_dw, dJ_db = self.gradient(xBat, yBat)
                self.w -= a * dJ_dw
                self.b -= a * dJ_db
            self.costs.append(self.costFunc(x, y))

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
