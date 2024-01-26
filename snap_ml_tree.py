from __future__ import print_function

import opendatasets as od
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
import time
import warnings


def train_snap_ml_tree():
    raw_data = pd.read_csv('creditcard.csv')
    print("There are " + str(len(raw_data)) + " observations in the credit card fraud dataset.")
    print("There are " + str(len(raw_data.columns)) + " variables in the dataset.")

    n_replicas = 10

    # inflate the original dataset
    big_raw_data = pd.DataFrame(np.repeat(raw_data.values, n_replicas, axis=0), columns=raw_data.columns)

    print("There are " + str(len(big_raw_data)) + " observations in the inflated credit card fraud dataset.")
    print("There are " + str(len(big_raw_data.columns)) + " variables in the dataset.")

    print("Minimum amount value is ", np.min(big_raw_data.Amount.values))
    print("Maximum amount value is ", np.max(big_raw_data.Amount.values))
    print("90% of the transactions have an amount less or equal than ", np.percentile(raw_data.Amount.values, 90))

    # data preprocessing such as scaling/normalization is typically useful for
    # linear models to accelerate the training convergence

    # standardize features by removing the mean and scaling to unit variance
    big_raw_data.iloc[:, 1:30] = StandardScaler().fit_transform(big_raw_data.iloc[:, 1:30])
    data_matrix = big_raw_data.values

    # X: feature matrix (for this analysis, we exclude the Time variable from the dataset)
    X = data_matrix[:, 1:30]

    # y: labels vector
    y = data_matrix[:, 30]

    # data normalization
    X = normalize(X, norm="l1")

    # print the shape of the features matrix and the labels vector
    print('X.shape=', X.shape, 'y.shape=', y.shape)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    print('X_train.shape=', X_train.shape, 'Y_train.shape=', y_train.shape)
    print('X_test.shape=', X_test.shape, 'Y_test.shape=', y_test.shape)

    # compute the sample weights to be used as input to the train routine so that
    # it takes into account the class imbalance present in this dataset
    w_train = compute_sample_weight('balanced', y_train)

    # for reproducible output across multiple function calls, set random_state to a given integer value
    sklearn_dt = DecisionTreeClassifier(max_depth=4, random_state=35)

    # train a Decision Tree Classifier using scikit-learn
    t0 = time.time()
    sklearn_dt.fit(X_train, y_train, sample_weight=w_train)
    sklearn_time = time.time() - t0
    print("[Scikit-Learn] Training time (s):  {0:.5f}".format(sklearn_time))

    # Snap ML offers multi-threaded CPU/GPU training of decision trees, unlike scikit-learn
    # to use the GPU, set the use_gpu parameter to True
    # snapml_dt = DecisionTreeClassifier(max_depth=4, random_state=45, use_gpu=True)

    # to set the number of CPU threads used at training time, set the n_jobs parameter
    # for reproducible output across multiple function calls, set random_state to a given integer value
    snapml_dt = DecisionTreeClassifier(max_depth=4, random_state=45, n_jobs=4)

    # train a Decision Tree Classifier model using Snap ML
    t0 = time.time()
    snapml_dt.fit(X_train, y_train, sample_weight=w_train)
    snapml_time = time.time() - t0
    print("[Snap ML] Training time (s):  {0:.5f}".format(snapml_time))
    # unfortunately we stop here as Snap ML isn't available for Apple Silicon
