import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

def train_decision_tree_model():
    raw = pd.read_csv("EffectiveMedication.csv")
    X = raw[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values

    le_sex = preprocessing.LabelEncoder()
    le_sex.fit(['F', 'M'])
    X[:, 1] = le_sex.transform(X[:, 1])

    le_BP = preprocessing.LabelEncoder()
    le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
    X[:, 2] = le_BP.transform(X[:, 2])

    le_Chol = preprocessing.LabelEncoder()
    le_Chol.fit(['NORMAL', 'HIGH'])
    X[:, 3] = le_Chol.transform(X[:, 3])

    y = raw["Drug"]

    X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

    drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
    drugTree.fit(X_trainset, y_trainset)
    predTree = drugTree.predict(X_testset)

    print("DecisionTree Accuracy: ", metrics.accuracy_score(y_testset, predTree))
    tree.plot_tree(drugTree)
    plt.show()
    return drugTree
