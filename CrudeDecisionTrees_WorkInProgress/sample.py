#!/usr/bin/env python
# coding: utf-8

# ### Library Imports
# - **random**: For generating random numbers during train-test split
# - **pprint**: For pretty-printing complex data structures
# - **pandas**: For data manipulation and analysis
# - **numpy**: For numerical operations
# - **matplotlib**: For data visualization
# - **seaborn**: For enhanced statistical visualizations

# In[302]:


import random
from pprint import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Data Loading & Initial Inspection
# - Load Iris dataset from CSV file located in './data/Iris.csv'
# - Display first 5 rows using `head()` to verify successful loading
# - Columns shown: Id, SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm, Species

# In[303]:


datSet = pd.read_csv("./data/Iris.csv")
datSet.head()


# ### Data Cleaning
# - Drop the 'Id' column as it's not useful for analysis
# - Verify removal by showing modified dataframe with new `head()` call

# In[304]:


datSet = datSet.drop(columns=["Id"])
datSet.head()


# ### Column Renaming
# - Rename 'Species' column to 'Label' for clarity
# - Makes distinction clearer between features (measurements) and target (label)

# In[305]:


datSet = datSet.rename(columns={"Species": "Label"})
datSet.head()


# ### Data Visualization
# - Create color mapping dictionary for different species
# - Define visualization function to show:
#   - Petal length vs petal width relationship
#   - Distinct clusters for different species
#   - Transparency (alpha=0.7) to handle overlapping points
# - X/Y labels and legend for interpretation
# - Title for context

# In[306]:


clrMap = {'Iris-setosa': 'r', 'Iris-versicolor': 'g', 'Iris-virginica': 'b'}


def plot_data(datSet):
    plt.figure(figsize=(8, 6))
    for species in datSet['Label'].unique():
        subset = datSet[datSet['Label'] == species]
        plt.scatter(subset['PetalLengthCm'], subset['PetalWidthCm'], c=clrMap[species], label=f"{species}", alpha=0.7)
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.legend()
    plt.title('Iris Dataset (Train & Test Split)')
    plt.show()


plot_data(datSet)


# ### Data Structure Inspection
# - Show dataset metadata using `info()`:
#   - 150 entries (samples)
#   - 4 numerical features (float64)
#   - 1 categorical target (object)
#   - No missing values
# - Memory usage (~6KB)

# In[307]:


datSet.info()


# In[308]:


datSet.index


# ### Custom Train-Test Split
# - Input validation: Convert float percentage to absolute number
# - Get all indices from dataframe
# - Randomly select test indices using `random.sample`
# - Create test set using `.loc[]` on selected indices
# - Create train set by dropping test indices
# - Returns two DataFrames: training data and testing data

# In[309]:


def testTrainSplit(datSet, testSize):
    if isinstance(testSize, float):
        testSize = round(testSize * len(datSet))
    allIdx = datSet.index.tolist()
    testIdx = random.sample(population=allIdx, k=testSize)
    testDat = datSet.loc[testIdx]
    trainDat = datSet.drop(testIdx)
    return trainDat, testDat


# ### Split Execution
# - Set random seed (0) for reproducibility
# - Create 20-sample test set (~13% of 150 total samples)
# - Show resulting shapes:
#   - Training data: 130 samples
#   - Testing data: 20 samples
# - Note: Non-stratified split (potential class imbalance risk)

# In[310]:


random.seed(0)
trainDat, testDat = testTrainSplit(datSet, testSize=20)
trainDat.shape, testDat.shape


# ### Training Data Preview
# - Display first 5 rows of training data
# - Verify all columns present except 'Id'
# - Note: Contains only setosa samples in shown rows (random sampling artifact)

# In[311]:


trainDat.head()


# ### Testing Data Preview
# - Display first 5 rows of testing data
# - Shows samples from all three species
# - Different index numbers confirm proper split

# In[312]:


testDat.head()


# ### Split Visualization
# - Differentiate train/test sets using:
#   - Filled circles for training data
#   - Hollow circles with black borders for test data
# - Maintains color coding by species
# - Alpha transparency helps visualize density
# - Visual verification of representative split

# In[313]:


clrMap = {'Iris-setosa': 'r', 'Iris-versicolor': 'g', 'Iris-virginica': 'b'}


def plotData(trainDat, testDat):
    plt.figure(figsize=(8, 6))
    for species in trainDat['Label'].unique():
        subset = trainDat[trainDat['Label'] == species]
        plt.scatter(subset['PetalLengthCm'], subset['PetalWidthCm'], c=clrMap[species], label=f"{species} (Train)", alpha=0.7)
    for species in testDat['Label'].unique():
        subset = testDat[testDat['Label'] == species]
        plt.scatter(subset['PetalLengthCm'], subset['PetalWidthCm'], edgecolors='k', facecolors='none', label=f"{species} (Test)", linewidth=1.2)
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Petal Width (cm)')
    plt.legend()
    plt.title('Iris Dataset (Train & Test Split)')
    plt.show()


plotData(trainDat, testDat)


# In[314]:


dat = trainDat.values
dat[:5]


# In[315]:


def isPure(dat):
    y = dat[:, -1]
    uniqClasses = np.unique(y)
    return True if len(uniqClasses) == 1 else False


# In[316]:


print(isPure(trainDat.values))
print(isPure(trainDat[trainDat["Label"] == "Iris-setosa"].values))
print(isPure(trainDat[trainDat.PetalWidthCm < 1.2].values))
print(isPure(trainDat[trainDat.PetalWidthCm < 0.8].values))


# In[317]:


def classifyDat(dat):
    y = dat[:, -1]
    uniqClasses, uniqClassesCounts = np.unique(y, return_counts=True)
    mostFreqClassIdx = uniqClassesCounts.argmax()
    clf = uniqClasses[mostFreqClassIdx]
    return clf


# In[318]:


print(classifyDat(trainDat[trainDat["Label"] == "Iris-setosa"].values))
print(classifyDat(trainDat[trainDat.PetalWidthCm < 1.2].values))
print(classifyDat(trainDat[trainDat.PetalWidthCm > 1.2].values))
print(classifyDat(trainDat[trainDat.PetalWidthCm > 0.8].values))
print(classifyDat(trainDat[trainDat.PetalWidthCm < 2].values))


# In[319]:


def getFeatType(datSet):
    featType = []
    uniqValLimit = 15
    for feat in datSet.columns:
        if feat != "Label":
            uniqVal = datSet[feat].unique()
            egVal = uniqVal[0]
            if isinstance(egVal, str) or len(uniqVal) <= uniqValLimit:
                featType.append("categorical")
            else:
                featType.append("continuous")
    return featType


# In[320]:


def getSplits(dat):
    m, n = dat.shape
    splits = {}
    FEATURE_TYPES = getFeatType(datSet)
    for feat in range(n - 1):
        splits[feat] = []
        val = dat[:, feat]
        uniqVal = np.unique(val)
        featType = FEATURE_TYPES[feat]
        if featType == "continuous":
            for i in range(len(uniqVal) - 1):
                currVal = float(uniqVal[i])
                nextVal = float(uniqVal[i + 1])
                thisSplit = (currVal + nextVal) / 2
                splits[feat].append(thisSplit)
        else:
            splits[feat] = uniqVal
    return splits


# In[321]:


splits = getSplits(dat)
splits


# In[322]:


sns.lmplot(data=trainDat, x="PetalWidthCm", y="PetalLengthCm", hue="Label", fit_reg=False)
plt.vlines(x=splits[3], ymin=0, ymax=max(trainDat.PetalLengthCm), linestyles='dashed')


# In[323]:


sns.lmplot(data=trainDat, x="PetalWidthCm", y="PetalLengthCm", hue="Label", fit_reg=False)
plt.hlines(y=splits[2], xmin=0, xmax=max(trainDat.PetalWidthCm), linestyles='dashed')


# In[324]:


splitFeat = 3
threshold = 0.8
featVal = dat[:, splitFeat]
featVal < threshold


# In[325]:


def split(dat, splitFeat, threshold):
    featVal = dat[:, splitFeat]
    type_of_feature = FEATURE_TYPES[splitFeat]
    if type_of_feature == "continuous":
        datBelow = dat[featVal <= threshold]
        datAbove = dat[featVal > threshold]
    else:
        datBelow = dat[featVal == threshold]
        datAbove = dat[featVal != threshold]
    return datBelow, datAbove


# In[326]:


datBelow, datAbove = split(dat, splitFeat, threshold)
plotDF = pd.DataFrame(dat, columns=datSet.columns)
sns.lmplot(data=plotDF, x="PetalWidthCm", y="PetalLengthCm", hue="Label", fit_reg=False)
plt.vlines(x=threshold, ymin=1, ymax=max(plotDF.PetalLengthCm), linestyles='dashed')


# In[327]:


def entropy(data):
    y = data[:, -1]
    uniqClasses, uniqClassesCounts = np.unique(y, return_counts=True)
    p = uniqClassesCounts / uniqClassesCounts.sum()
    h = sum(-p * np.log2(p))
    return h


# In[328]:


print(entropy(datBelow))
print(entropy(datAbove))


# In[329]:


datBelow, datAbove = split(dat, 3, 1.1)
print(entropy(datBelow))
print(entropy(datAbove))


# In[330]:


splits = getSplits(dat)
splits


# In[331]:


def overallEntropy(datBelow, datAbove):
    n = len(datBelow) + len(datAbove)
    wt_below = len(datBelow) / n
    wt_above = len(datAbove) / n
    fullEntropy = (wt_below * entropy(datBelow) + wt_above * entropy(datAbove))
    return fullEntropy


# In[332]:


print(overallEntropy(datBelow, datAbove))


# In[333]:


def bestSplit(dat, splits):
    bestH = float("inf")
    bestSplitFeat, bestThreshold = 0, 0
    for feat in splits:
        for val in splits[feat]:
            datBelow, datAbove = split(dat, splitFeat=feat, threshold=val)
            currH = overallEntropy(datBelow, datAbove)
            if currH <= bestH:
                bestH = currH
                bestSplitFeat = feat
                bestThreshold = val
    return bestSplitFeat, bestThreshold


# In[334]:


bestSplit(dat, splits)


# In[335]:


subTree = {"ques": ["yes_answer", "no_answer"]}


# In[336]:


egTree = {"petal_width <= 0.8": ["Iris-setosa", {"petal_width <= 1.65": [{"petal_length <= 4.9": ["Iris-versicolor", "Iris-virginica"]}, "Iris-virginica"]}]}


# In[337]:


FEATURE_TYPES = getFeatType(trainDat)
print(FEATURE_TYPES)


# In[338]:


datSet.columns


# In[339]:


def ID3(datSet, c=0, minSubset=2, maxDepth=5):
    if c == 0:
        global COLUMN_HEADERS, FEATURE_TYPES
        COLUMN_HEADERS = datSet.columns
        FEATURE_TYPES = getFeatType(datSet)
        dat = datSet.values
    else:
        dat = datSet
    if isPure(dat) or len(dat) < minSubset or c == maxDepth:
        return classifyDat(dat)
    else:
        c += 1
        splits = getSplits(dat)
        splitFeat, threshold = bestSplit(dat, splits)
        datBelow, datAbove = split(dat, splitFeat, threshold)
        if len(datBelow) == 0 or len(datAbove) == 0:
            return classifyDat(dat)
        featNam = COLUMN_HEADERS[splitFeat]
        featType = FEATURE_TYPES[splitFeat]
        if featType == "continuous":
            ques = f"{featNam} <= {threshold}"
        else:
            ques = f"{featNam} = {threshold}"
        subTree = {ques: []}
        yesAns = ID3(datBelow, c, minSubset, maxDepth)
        noAns = ID3(datAbove, c, minSubset, maxDepth)
        if yesAns == noAns:
            subTree = yesAns
        else:
            subTree[ques].append(yesAns)
            subTree[ques].append(noAns)
        return subTree


# In[340]:


tree = ID3(trainDat[trainDat.Label != "Iris-virginica"])
tree


# In[341]:


tree = ID3(trainDat)
tree


# In[342]:


pprint(tree)


# In[343]:


tree = ID3(trainDat, minSubset=60)
pprint(tree)


# In[344]:


tree = ID3(trainDat, maxDepth=1)
pprint(tree)


# In[345]:


tree = ID3(trainDat, maxDepth=3)
pprint(tree)


# In[346]:


tree.keys()


# In[347]:


list(tree.keys())[0]


# In[348]:


ques = list(tree.keys())[0]
ques.split()


# In[349]:


featNam, cmpOp, val = ques.split()
print(featNam, cmpOp, val)


# In[350]:


# print(eg["PetalWidthCm"] <= 0.8)


# In[351]:


def classifyEg(eg, tree):
    ques = list(tree.keys())[0]
    featNam, cmpOp, val = ques.split(" ")
    if cmpOp == "<=":
        if eg[featNam] <= float(val):
            ans = tree[ques][0]
        else:
            ans = tree[ques][1]
    else:
        if str(eg[featNam]) == val:
            ans = tree[ques][0]
        else:
            ans = tree[ques][1]
    if not isinstance(ans, dict):
        return ans
    else:
        remTree = ans
        return classifyEg(eg, remTree)


# In[352]:


eg = testDat.iloc[0]
eg


# In[353]:


classifyEg(eg, tree)


# In[354]:


eg = testDat.iloc[1]
eg


# In[355]:


classifyEg(eg, tree)


# In[356]:


eg = testDat.iloc[2]
eg


# In[357]:


classifyEg(eg, tree)


# In[358]:


def calcAccuracy(datSet, tree):
    datSet["classification"] = datSet.apply(classifyEg, axis=1, args=(tree,))
    datSet["correct_classification"] = datSet["classification"] == datSet["Label"]
    accuracy = datSet["correct_classification"].mean()
    return accuracy


# In[359]:


print(calcAccuracy(datSet, tree))


# In[360]:


testDat


# In[361]:


testDat.loc[77]


# In[362]:


def plot_node(text, x, y, node_type):
    bbox = dict(boxstyle="round,pad=0.3", fc="white" if node_type == "decision" else "lightgreen", ec="black")
    plt.text(x, y, text, ha="center", va="center", bbox=bbox, fontsize=9)


def plot_edge(x1, y1, x2, y2, text=None):
    plt.plot([x1, x2], [y1, y2], 'k-', lw=1)
    if text:
        plt.text((x1 + x2) / 2, (y1 + y2) / 2, text, ha="center", va="center", fontsize=8)


def plot_tree(tree, x=0, y=0, dx=2, dy=1, level=0, max_level=None):
    if max_level and level > max_level:
        return
    if isinstance(tree, dict):
        question = list(tree.keys())[0]
        yes_answer = tree[question][0]
        no_answer = tree[question][1]
        plot_node(question, x, y, "decision")
        x_yes = x - dx / (level + 1)
        x_no = x + dx / (level + 1)
        y_child = y - dy
        plot_edge(x, y, x_yes, y_child, "Yes")
        plot_tree(yes_answer, x_yes, y_child, dx, dy, level + 1, max_level)
        plot_edge(x, y, x_no, y_child, "No")
        plot_tree(no_answer, x_no, y_child, dx, dy, level + 1, max_level)
    else:
        plot_node(f"Class: {tree}", x, y, "leaf")


# In[363]:


plt.figure(figsize=(12, 8))
plot_tree(tree, max_level=3)
plt.axis('off')
plt.title("Decision Tree Visualization", fontsize=14)
plt.show()


# In[364]:


from matplotlib.colors import ListedColormap


def plot_decision_boundary(tree, data, feature1, feature2, title=None, step=0.02):
    X = data[[feature1, feature2]].values
    y = data['Label'].values
    classes = np.unique(y)
    colors = ['#FFAAAA', '#AAFFAA', '#AAAAFF'][:len(classes)]
    cmap_light = ListedColormap(colors)
    cmap_bold = ListedColormap([c.replace('AA', '44') for c in colors])
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    mesh_df = pd.DataFrame(mesh_points, columns=[feature1, feature2])
    for col in data.columns:
        if col not in [feature1, feature2, 'Label']:
            mesh_df[col] = data[col].median()
    Z = np.array([classifyEg(row, tree) for _, row in mesh_df.iterrows()])
    label_to_num = {label: i for i, label in enumerate(classes)}
    Z = np.array([label_to_num[label] for label in Z])
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)
    for i, label in enumerate(classes):
        idx = (y == label)
        plt.scatter(X[idx, 0], X[idx, 1], c=cmap_bold(i), label=label, edgecolor='k', s=50)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xlabel(feature1, fontsize=12)
    plt.ylabel(feature2, fontsize=12)
    if title:
        plt.title(title, fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


plot_decision_boundary(tree, trainDat, 'PetalLengthCm', 'PetalWidthCm', title="Decision Boundary (Petal Length vs Width)")


# In[365]:


pprint(tree)


# In[366]:


from graphviz import Digraph
import re


def sanitize_name(name):
    """Convert strings to valid DOT identifiers"""
    return re.sub(r'[^a-zA-Z0-9_]', '_', str(name))


def tree_to_graphviz(tree, feature_names, class_names=None):
    """
    Convert a decision tree to Graphviz format with proper escaping

    Args:
        tree: Your decision tree (nested dictionary structure)
        feature_names: List of feature names
        class_names: Optional mapping of class values to names
    """
    dot = Digraph(format='png')
    dot.attr('node', shape='box', style='rounded')
    _build_graph(tree, dot, feature_names, class_names)
    return dot


def _build_graph(tree, dot, feature_names, class_names=None, parent_node=None, edge_label=""):
    if isinstance(tree, dict):
        question = list(tree.keys())[0]

        # Create safe node ID and label
        node_id = f"node_{sanitize_name(question)}"
        dot.node(node_id, label=question)

        if parent_node:
            dot.edge(parent_node, node_id, label=edge_label)

        # Recursively build left and right branches
        _build_graph(tree[question][0], dot, feature_names, class_names, node_id, "Yes")
        _build_graph(tree[question][1], dot, feature_names, class_names, node_id, "No")
    else:
        # Handle leaf node
        leaf_id = f"leaf_{sanitize_name(tree)}"
        class_label = class_names[tree] if class_names and tree in class_names else str(tree)
        dot.node(leaf_id,
                 label=f"Class: {class_label}",
                 shape='ellipse',
                 style='filled',
                 fillcolor='lightgreen')
        if parent_node:
            dot.edge(parent_node, leaf_id, label=edge_label)


# Usage example
class_names = {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'}
dot = tree_to_graphviz(tree,
                       feature_names=datSet.columns[:-1],
                       class_names=class_names)

# Customize graph appearance
dot.attr(size='10,10', rankdir='TB')  # Top-to-bottom layout
dot.attr('edge', fontsize='10')

# Render and display
dot.render("iris_decision_tree", view=True, cleanup=True)
display(dot)


# In[367]:


from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns


def evaluate_model(tree, train_data, test_data, feature1, feature2):
    """
    Evaluate decision tree model with visualization and metrics

    Args:
        tree: Trained decision tree
        train_data: Training DataFrame
        test_data: Testing DataFrame
        feature1: First feature for visualization
        feature2: Second feature for visualization
    """
    # 1. Plot Decision Boundary
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)

    # Create mesh grid
    x_min, x_max = train_data[feature1].min() - 1, train_data[feature1].max() + 1
    y_min, y_max = train_data[feature2].min() - 1, train_data[feature2].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                         np.linspace(y_min, y_max, 100))

    # Prepare grid points with all features
    grid_points = pd.DataFrame({
        feature1: xx.ravel(),
        feature2: yy.ravel()
    })
    for col in train_data.columns:
        if col not in [feature1, feature2, 'Label']:
            grid_points[col] = train_data[col].median()

    # Predict classes
    Z = np.array([classifyEg(row, tree) for _, row in grid_points.iterrows()])
    Z = Z.reshape(xx.shape)
    # 2. Generate Classification Report and Confusion Matrix
    plt.subplot(1, 2, 1)

    # Predict on test data
    y_true = test_data['Label']
    y_pred = [classifyEg(row, tree) for _, row in test_data.iterrows()]

    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=np.unique(y_true),
                yticklabels=np.unique(y_true),
                cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title("Confusion Matrix")

    plt.tight_layout()
    plt.show()


# Usage
evaluate_model(tree, trainDat, testDat, 'PetalLengthCm', 'PetalWidthCm')


# In[368]:


# Find misclassified samples
testDat['Predicted'] = [classifyEg(row, tree) for _, row in testDat.iterrows()]
misclassified = testDat[testDat['Label'] != testDat['Predicted']]

# Plot with highlighting
plt.figure(figsize=(8, 6))
for species in testDat['Label'].unique():
    subset = testDat[testDat['Label'] == species]
    plt.scatter(subset['PetalLengthCm'], subset['PetalWidthCm'],
                label=species, s=80)

# Highlight errors
plt.scatter(misclassified['PetalLengthCm'], misclassified['PetalWidthCm'],
            s=200, facecolors='none', edgecolors='r', linewidths=2,
            label='Misclassified')
plt.legend()
plt.title('Misclassified Samples (Red Circles)')
plt.show()


# In[369]:


def compute_feature_importance(tree, feature_names):
    """Count how often each feature is used for splitting"""
    importance = {f: 0 for f in feature_names}

    def _count_splits(node):
        if isinstance(node, dict):
            question = list(node.keys())[0]
            feat = question.split('<=')[0].strip()
            importance[feat] += 1
            _count_splits(node[question][0])  # Yes branch
            _count_splits(node[question][1])  # No branch

    _count_splits(tree)
    return importance


features = trainDat.columns[:-1]
importance = compute_feature_importance(tree, features)
print("Feature Importance:", importance)


# In[370]:


# from sklearn.model_selection import KFold
#
#
# def cross_validate(data, n_splits=5):
#     kf = KFold(n_splits=n_splits)
#     accuracies = []
#     for train_idx, val_idx in kf.split(data):
#         train = data.iloc[train_idx]
#         val = data.iloc[val_idx]
#         fold_tree = ID3(train, maxDepth=3)
#         y_true = val['Label']
#         y_pred = [classifyEg(row, fold_tree) for _, row in val.iterrows()]
#         accuracy = np.mean(np.array(y_true) == np.array(y_pred))
#         accuracies.append(accuracy)
#
#     print(f"Cross-Validation Accuracy: {np.mean(accuracies):.2f} ± {np.std(accuracies):.2f}")
#
#
# cross_validate(datSet)  # Use full dataset


# In[371]:


datSet = pd.read_csv("./data/Titanic-Dataset.csv")
datSet["Label"] = datSet["Survived"]
datSet = datSet.drop(columns=["PassengerId", "Survived", "Name", "Ticket", "Cabin"])
datSet.head()


# In[372]:


datSet.info()


# In[373]:


datSet.isnull().sum()


# In[374]:


med_age = datSet.Age.median()
mod_emb = datSet.Embarked.mode()[0]
datSet = datSet.fillna({"Age": med_age, "Embarked": mod_emb})
datSet.head()


# In[375]:


trainDat, testDat = testTrainSplit(datSet, testSize=0.2)
trainDat.shape, testDat.shape


# In[376]:


trainDat.head()


# In[377]:


testDat.head()


# In[378]:


datSet.columns


# In[379]:


featType = getFeatType(datSet)
X = datSet.drop(columns=["Label"]).columns
i = 0
for feat in X:
    print(feat, ":", featType[i])
    i += 1


# In[380]:


datBelow, datAbove = split(datSet.values, splitFeat=1, threshold="male")
print(np.unique(datBelow[:, 1]))
print(np.unique(datAbove[:, 1]))


# In[381]:


dat = trainDat.values
getSplits(trainDat.values)


# In[382]:


tree = ID3(trainDat, maxDepth=3)
pprint(tree)


# In[383]:


tree = ID3(trainDat, maxDepth=10)
pprint(tree)


# In[384]:


eg = testDat.iloc[0]
eg


# In[385]:


classifyEg(eg, tree)


# In[386]:


print(calcAccuracy(testDat, tree))


# In[ ]:




