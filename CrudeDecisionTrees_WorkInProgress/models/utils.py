from collections import Counter

import pandas as pd
import numpy as np


def loadData(path):
    return pd.read_csv(path)


def cleanData(datSet):
    datSet = datSet.drop(columns=["day"])
    for col in datSet.columns:
        datSet[col] = datSet[col].astype("category").cat.codes
    return datSet


def getFeatLabel(datSet):
    X = datSet.iloc[:, :-1]
    y = datSet.iloc[:, -1]
    return X, y


def entropy(y):
    classFreq = Counter(y)
    h = 0
    for freq in classFreq.values():
        p_i = freq / len(y)
        h -= p_i * np.log2(p_i)
    return h


def infoGain(X, y, featIdx):
    h_parent = entropy(y)
    uniqVal = X.iloc[:, featIdx].unique()
    s = len(y)
    h_child = 0
    for val in uniqVal:
        s_v = y[X.iloc[:, featIdx] == val]
        h_child += (len(s_v) / s) * entropy(s_v)
    return h_parent - h_child


def bestSplitFeat(X, y):
    bestFeat = None
    bestI = float("-inf")
    for featIdx in range(X.shape[1]):
        i = infoGain(X, y, featIdx)
        if isinstance(i, (int, float)):
            if i > bestI:
                bestI = i
                bestFeat = featIdx
    return bestFeat


import numpy as np
import pandas as pd

import numpy as np


def splitDatSet(X, y, bestFeat):
    uniqVal = np.unique(X.iloc[:, bestFeat])
    subsets = []
    for val in uniqVal:
        subset_X = X[X.iloc[:, bestFeat] == val].drop(columns=X.columns[bestFeat])
        subset_y = y[X.iloc[:, bestFeat] == val]
        subset_X = subset_X.reset_index(drop=True)
        subset_y = subset_y.reset_index(drop=True)
        subsets.append((subset_X, subset_y))
    return subsets


class Node:
    def __init__(self, feat=None, val=None, children=None, label=None):
        self.feat = feat
        self.val = val
        self.children = children
        self.label = label


def buildTree(X, y, depth=0, max_depth=3):
    if len(set(y)) == 1:
        return Node(label=y.iloc[0] if isinstance(y, pd.Series) else y[0])
    if depth == max_depth:
        majorLabel = Counter(y).most_common(1)[0][0]
        return Node(label=majorLabel)

    bestFeat = bestSplitFeat(X, y)
    if bestFeat is None:
        majorLabel = Counter(y).most_common(1)[0][0]
        return Node(label=majorLabel)

    splits = splitDatSet(X, y, bestFeat)
    unique_values = np.unique(X.iloc[:, bestFeat])
    children = []

    for i, (X_subset, y_subset) in enumerate(splits):
        if len(y_subset) == 0:
            majorLabel = Counter(y).most_common(1)[0][0]
            children.append(Node(label=majorLabel, val=unique_values[i]))
        else:
            child = buildTree(X_subset, y_subset, depth + 1, max_depth)
            child.val = unique_values[i]  # Set the value for this branch
            children.append(child)

    return Node(feat=bestFeat, children=children)


def predict(tree, sample):
    if tree.label is not None:
        return tree.label
    if not hasattr(tree, 'children'):
        return None

    feat_value = sample.iloc[tree.feat] if isinstance(sample, pd.Series) else sample[tree.feat]

    for child in tree.children:
        if child.val == feat_value:
            return predict(child, sample)

    # If no matching branch found, return majority class
    if hasattr(tree, '_majority_class'):
        return tree._majority_class
    return None
