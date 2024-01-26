import json
import string

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score


def train_linear_reg_model(df: np.ndarray, x: string, y: string):
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
    train_x = np.asanyarray(train[[x]])
    train_y = np.asanyarray(train[[y]])
    regr.fit(train_x, train_y)

    # outputs the resulting trend line as a plot
    plt.scatter(train[x], train[y], color='blue')
    plt.plot(train_x, regr.coef_[0][0] * train_x + regr.intercept_[0], '-r')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()

    # the regression function
    print('Coefficients: ', regr.coef_)
    print('Intercept: ', regr.intercept_)

    # tests the model
    test_x = np.asanyarray(test[[x]])
    test_y = np.asanyarray(test[[y]])
    res_y = regr.predict(test_x)

    # prints the testing results
    print("Mean absolute error: %.2f" % np.mean(np.absolute(res_y - test_y)))
    print("Residual sum of squares (MSE): %.2f" % np.mean((res_y - test_y) ** 2))
    print("R2-score: %.2f" % r2_score(test_y, res_y))
    print('============================================')
    print(' ')

    return regr


def export_linear_regression_to_json(regrmodel, feature_column, target_column, json_filename):
    """
    Export linear regression model parameters to a JSON file.

    Parameters:
    :param regrmodel: The trained LinearRegression model.
    :param feature_column: Name of the feature column.
    :param target_column: Name of the target column.
    :param json_filename: The name of the JSON file to save the model parameters.
    """
    model_params = {
        'coefficients': regrmodel.coef_.tolist(),
        'intercept': regrmodel.intercept_.tolist(),
        'feature_column': feature_column,
        'target_column': target_column
    }

    with open(json_filename, 'w') as json_file:
        json.dump(model_params, json_file, indent=4)

