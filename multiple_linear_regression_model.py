import array
import string
from typing import List

import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score


def train_multi_linear_reg_model(df: np.ndarray, x: List[string], y: string):
    """
    Trains a linear regression model against x and y for a given dataframe
    it also outputs validation data during the process
    :param df: the dataframe
    :param x: the feature
    :param y: the target
    :return: a linear regression model
    """
    # create a mask for the train/test set
    mask = np.random.rand(len(df)) < 0.8
    train = df[mask]
    test = df[~mask]

    # creates a linear reg model and assigns training data of x and y
    regr = linear_model.LinearRegression()
    train_x = np.asanyarray(train[x])
    train_y = np.asanyarray(train[[y]])
    regr.fit(train_x, train_y)

    # the regression function
    print('Coefficients: ', regr.coef_)
    print('Intercept: ', regr.intercept_)

    # tests the model
    test_x = np.asanyarray(test[x])
    test_y = np.asanyarray(test[[y]])
    test_y_ = regr.predict(test_x)

    # prints the testing results
    print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y, test_y_))
    print('============================================')
    print(' ')

    return regr