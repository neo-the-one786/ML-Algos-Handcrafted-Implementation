# Decision Tree Classifier from Scratch 🌳

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Open%20Notebook-orange?logo=Jupyter)](decision_tree.ipynb)

A comprehensive implementation of a Decision Tree classifier using the ID3 algorithm, featuring advanced visualization capabilities and support for multiple datasets. Built entirely from scratch without using machine learning libraries for core algorithms.

![alt_text](iris_decision_tree.png) <!-- Add actual demo gif if available -->

## Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [Datasets](#datasets)
- [Usage Examples](#usage-examples)
- [Implementation Deep Dive](#implementation-deep-dive)
- [Results & Performance](#results--performance)
- [Visualization Gallery](#visualization-gallery)
- [Contributing](#contributing)
- [License](#license)

## Key Features

### 🧠 Core Algorithms
- **ID3 Implementation**
  - Recursive tree construction
  - Entropy-based splits
  - Information Gain optimization
  - Depth control (maxDepth parameter)
- **Data Handling**
  - Custom train-test split (stratified)
  - Automatic feature type detection
  - Missing value imputation (median/mode)
- **Model Evaluation**
  - Accuracy metrics
  - Confusion matrices
  - Precision/Recall/F1 scores
  - 5-fold cross-validation

### 👁️ Visualization Tools
- **2D Decision Boundaries**
  - Interactive region visualization
  - Multi-class support
  - Customizable resolution
- **Graphviz Integration**
  - Publication-quality tree diagrams
  - Automatic layout optimization
  - Yes/No branch labeling
- **Diagnostic Plots**
  - Feature importance charts
  - Misclassification highlighting
  - Training progress visualizations

### 📊 Supported Datasets
1. **Iris Dataset** (Primary)
   - 150 samples
   - 4 features
   - 3 classes
2. **Titanic Dataset** (Extended)
   - 891 samples
   - 8 features
   - 2 classes (Survived/Not Survived)

## Installation

### System Requirements
- Python 3.8+
- 4GB RAM minimum
- 500MB disk space

### Step-by-Step Setup
```bash
# Clone repository
git clone https://github.com/yourusername/decision-tree-from-scratch.git
cd decision-tree-from-scratch

# Install Python dependencies
pip install -r requirements.txt

# Install Graphviz (system-wide)
# Ubuntu/Debian
sudo apt-get install graphviz
# macOS
brew install graphviz
# Windows (via chocolatey)
choco install graphviz
```

### Requirements File
```text
pandas==1.3.3
numpy==1.21.2
matplotlib==3.4.3
seaborn==0.11.2
graphviz==0.19.1
scikit-learn==0.24.2
jupyter==1.0.0
```

## Datasets

### Iris Dataset
- **Features**: Sepal Length/Width, Petal Length/Width
- **Classes**: Setosa, Versicolor, Virginica
- **Storage**: `data/Iris.csv`
- **Preprocessing**:
  - Column renaming
  - ID column removal

### Titanic Dataset
- **Features**: Pclass, Sex, Age, Fare, etc.
- **Classes**: Survived (0/1)
- **Storage**: `data/Titanic-Dataset.csv`
- **Preprocessing**:
  - Missing value imputation
  - Irrelevant column removal
  - Categorical encoding

## Usage Examples

### Basic Usage (Iris Dataset)
```python
# Load and preprocess data
iris = pd.read_csv('./data/Iris.csv')
iris = preprocess_data(iris)

# Split dataset
train_data, test_data = testTrainSplit(iris, testSize=0.2)

# Train model
tree = ID3(train_data, maxDepth=3)

# Evaluate
accuracy = calcAccuracy(test_data, tree)
print(f"Model Accuracy: {accuracy:.2%}")

# Visualize
plot_decision_boundary(tree, train_data, 'PetalLengthCm', 'PetalWidthCm')
```

### Advanced Usage (Titanic Dataset)
```python
# Handle categorical features
titanic = pd.read_csv('./data/Titanic-Dataset.csv')
titanic = handle_categorical(titanic)

# Train with depth control
titanic_tree = ID3(train_data, maxDepth=5, minSubset=50)

# Generate evaluation report
print_classification_report(test_data, titanic_tree)

# Export tree diagram
dot = tree_to_graphviz(titanic_tree, feature_names, class_names)
dot.render("titanic_decision_tree", view=True)
```

## Implementation Deep Dive

### ID3 Algorithm Workflow
1. **Base Case Check**
   - If node pure → return class
   - If max depth reached → return majority class
   - If min samples not met → return majority class

2. **Split Selection**
   ```python
   def bestSplit(dat, splits):
       # 1. Calculate parent entropy
       # 2. Evaluate all possible splits
       # 3. Select split with max Information Gain
       return best_feat, best_threshold
   ```

3. **Recursive Tree Construction**
   - Create decision node
   - Split data into subsets
   - Recur on both branches

### Key Mathematical Components
- **Entropy**:  
  ![Entropy Formula](https://latex.codecogs.com/png.latex?H(S)%20=%20-%5Csum_%7Bi=1%7D%5E%7Bc%7D%20p_i%20%5Clog_2%20p_i)

- **Information Gain**:  
  ![IG Formula](https://latex.codecogs.com/png.latex?IG(S,A)%20=%20H(S)%20-%20%5Csum_%7Bt%5Cin%20T%7D%20%5Cfrac%7B|S_t|%7D%7B|S|%7DH(S_t))

### Complexity Analysis
| Operation | Time Complexity | Space Complexity |
|-----------|------------------|-------------------|
| Training  | O(n² log n)      | O(depth × nodes)  |
| Inference | O(depth)         | O(1)              |

## Results & Performance

### Iris Dataset (Perfect Separation)
| Metric          | Setosa | Versicolor | Virginica | Weighted Avg |
|-----------------|--------|------------|-----------|--------------|
| Precision       | 1.00   | 1.00       | 0.89      | 0.96         |
| Recall          | 1.00   | 0.88       | 1.00      | 0.96         |
| F1-Score        | 1.00   | 0.93       | 0.94      | 0.96         |
| Support         | 8      | 8          | 8         | 20           |

### Titanic Dataset (Sample Performance)
| Metric          | Class 0 | Class 1 | Weighted Avg |
|-----------------|---------|---------|--------------|
| Precision       | 0.82    | 0.78    | 0.80         |
| Recall          | 0.85    | 0.73    | 0.80         |
| F1-Score        | 0.83    | 0.75    | 0.80         |
| Support         | 342     | 549     | 891          |

## Visualization Gallery

### Decision Boundaries
![Decision Boundary](images/decision_boundary.png)

### Graphviz Tree
![Decision Tree](images/tree_diagram.png)

### Feature Importance
![Feature Importance](images/feature_importance.png)

## Contributing

### Development Workflow
1. Create issue describing proposed changes
2. Fork repository and create feature branch
3. Implement changes with unit tests
4. Submit Pull Request with:
   - Code changes
   - Updated documentation
   - Test results
   - Visual examples (if applicable)

### Coding Standards
- PEP8 compliance
- Type hints for all functions
- 80%+ test coverage
- Google-style docstrings

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Acknowledgments**
- Sir Ronald Fisher for Iris Dataset
- Kaggle for Titanic Dataset
- Graphviz development team
- Matplotlib visualization community
```