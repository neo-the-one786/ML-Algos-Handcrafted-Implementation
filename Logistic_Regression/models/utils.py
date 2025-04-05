import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def loadData():
    x = pd.read_csv("../data/logisticX.csv")
    y = pd.read_csv("../data/logisticY.csv")
    return x, y


class MinMaxScaler:
    def __init__(self):
        pass

    def fitTransform(self, x):
        xMin, xMax = np.min(x, axis=0), np.max(x, axis=0)
        xNorm = (x - xMin) / (xMax - xMin)
        return xNorm


class LogisticRegression:
    def __init__(self):
        self.w = np.zeros((2, 1))
        self.b = 0
        self.costs = []

    def linActFunc(self, x):
        z = np.dot(x, self.w) + self.b
        return z

    def modelFunc(self, x):
        z = self.linActFunc(x)
        yHat = 1 / (1 + np.exp(-z))
        return yHat

    def lossFunc(self, x, y):
        yHat = self.modelFunc(x)
        yHat = np.clip(yHat, 1e-10, 1 - 1e-10)
        l = -(y * np.log2(yHat) + (1 - y) * np.log2(1 - yHat))
        return l

    def costFunc(self, x, y):
        m = x.shape[0]
        l = self.lossFunc(x, y)
        J_wb = 1 / m * np.sum(l)
        return J_wb

    def error(self, x, y):
        yHat = self.modelFunc(x)
        e = yHat - y
        return e

    def gradient(self, x, y):
        m = x.shape[0]
        e = self.error(x, y)
        dJ_dw = (1 / m) * np.dot(x.T, e)
        dJ_db = (1 / m) * np.sum(e)
        return dJ_dw, dJ_db

    def fit(self, x, y, epochs, learning_rate, min_delta=0.0001, patience=5):
        x = x.to_numpy()
        y = y.to_numpy()
        a = learning_rate
        bestCost = float("inf")
        patienceCnt = 0
        for _ in range(epochs):
            dJ_dw, dJ_db = self.gradient(x, y)
            self.w -= a * dJ_dw
            self.b -= a * dJ_db
            currCost = self.costFunc(x, y)
            self.costs.append(currCost)
            if currCost < bestCost - min_delta:
                bestCost = currCost
                patienceCnt = 0
            else:
                patienceCnt += 1
            if patienceCnt >= patience:
                break
        return self.costFunc(x, y)

    def predict(self, x, threshold=0.5):
        return np.where(self.modelFunc(x) >= threshold, 1, 0)

    def plotCost(self, subtitle):
        fig, ax = plt.subplots()
        ax.set_title("Cost vs Epoch")
        fig.suptitle(subtitle)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cost")
        ax.plot(list(range(1, len(self.costs) + 1)), self.costs)
        ax.grid(True)
        plt.show()

    def decisionBoundary(self, x, y, subtitle):
        x = x.to_numpy()
        y = y.to_numpy()
        x1Max = x[:, 0].max() + 1
        x1Min = x[:, 0].min() - 1
        x2Max = x[:, 1].max() + 1
        x2Min = x[:, 1].min() - 1
        x1, x2 = np.meshgrid(np.linspace(x1Min, x1Max, 100), np.linspace(x2Min, x2Max, 100))
        gridPts = np.c_[x1.ravel(), x2.ravel()]
        yHat = self.modelFunc(gridPts)
        yHat = yHat.reshape(x1.shape)
        fig, ax = plt.subplots()
        ax.contour(x1, x2, yHat, levels=[0.5], linewidths=2, colors="black")
        ax.plot(x[y.flatten() == 0, 0], x[y.flatten() == 0, 1], "ro", label="Class 0")
        ax.plot(x[y.flatten() == 1, 0], x[y.flatten() == 1, 1], "bo", label="Class 1")
        # ax.scatter(x[y == 0, 0], x[y == 0, 1], c='red', marker='o', label="Class 0")
        # ax.scatter(x[y == 1, 0], x[y == 1, 1], c='blue', marker='o', label="Class 1")
        ax.set_title("Decision Boundary with Data Points")
        fig.suptitle(subtitle)
        ax.set_xlabel("Feature x_1")
        ax.set_ylabel("Feature x_2")
        ax.legend()
        ax.grid(True)
        plt.show()

    def confusionMatrix(self, y, yHat):
        y = y.to_numpy()
        TP = np.sum((y == 1) & (yHat == 1))
        TN = np.sum((y == 0) & (yHat == 0))
        FP = np.sum((y == 0) & (yHat == 1))
        FN = np.sum((y == 1) & (yHat == 0))
        cm = np.array([[TP, FP], [FN, TN]])
        return cm

    def classificationReport(self, y, yHat):
        cm = self.confusionMatrix(y, yHat)
        TP, TN = cm[0]
        FP, FN = cm[1]
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        f1Score = 2 * precision * recall / (precision + recall)
        return accuracy, precision, recall, f1Score


def compareCosts(costs1, costs2, epochs, subtitle):
    fig, ax = plt.subplots()
    ax.plot(list(range(1, epochs + 1)), costs1, label='lr = 0.1', color='blue')
    ax.plot(list(range(1, epochs + 1)), costs2, label='lr = 5', color='red')
    ax.set_title('Cost Function vs Iterations')
    fig.suptitle(subtitle)
    ax.set_xlabel('Iterations')
    ax.set_ylabel('Cost')
    ax.legend()
    ax.grid(True)
    plt.show()
