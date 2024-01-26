import numpy as np
import pandas as pd
from sklearn import preprocessing

from decision_tree_model import train_decision_tree_model
from knn import train_knn
from linear_regression_model import train_linear_reg_model, export_linear_regression_to_json
from multiple_linear_regression_model import train_multi_linear_reg_model
from snap_ml_tree import train_snap_ml_tree

# raw = pd.read_csv("FuelConsumptionCo2.csv")
# cdf = raw[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
# cdf.head(9)


# ==============================================================================================
# TRAINS AND OUTPUTS A LINEAR REGRESSION
# model = train_linear_reg_model(cdf, "ENGINESIZE", "CO2EMISSIONS")
# export_linear_regression_to_json(model, "ENGINESIZE", "CO2EMISSIONS", "model.json")


# ==============================================================================================
# TRAINS AND OUTPUTS A MULTIPLE LINEAR REGRESSION
# model = train_multi_linear_reg_model(cdf, ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB'], 'CO2EMISSIONS')
# export_linear_regression_to_json(model, ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB'], 'CO2EMISSIONS', 'model.json')


# ==============================================================================================
# TRAINS AND COMPARE K VALUES ON KNN
# raw = pd.read_csv("CustomerCategories.csv")
# X = raw[['region', 'tenure', 'age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']].values
# y = raw['custcat'].values

# train_knn(X, y)

# ==============================================================================================
# TRAINS AND ASSESSES PREDICTIONS FOR A#  DECISION TREE
# model = train_decision_tree_model()
#
# new_data = [
#     [25, 'F', 'HIGH', 'HIGH', 22.501],
#     [34, 'M', 'NORMAL', 'HIGH', 14.209],
#     [28, 'M', 'LOW', 'NORMAL', 10.803],
#     [39, 'F', 'LOW', 'HIGH', 8.423],
#     [55, 'M', 'LOW', 'HIGH', 16.872],
#     [36, 'F', 'HIGH', 'HIGH', 12.645],
#     [32, 'M', 'NORMAL', 'NORMAL', 9.814],
#     [45, 'F', 'LOW', 'HIGH', 14.927],
#     [52, 'M', 'NORMAL', 'NORMAL', 11.678],
#     [29, 'M', 'LOW', 'NORMAL', 18.765],
#     [42, 'F', 'HIGH', 'HIGH', 20.431],
#     [58, 'F', 'HIGH', 'NORMAL', 14.009],
#     [33, 'M', 'LOW', 'HIGH', 19.573],
#     [68, 'F', 'LOW', 'HIGH', 22.491],
#     [38, 'F', 'NORMAL', 'HIGH', 13.825],
#     [20, 'M', 'HIGH', 'HIGH', 16.237],
#     [47, 'F', 'LOW', 'NORMAL', 10.963],
#     [31, 'M', 'NORMAL', 'NORMAL', 11.642],
#     [56, 'M', 'HIGH', 'HIGH', 17.482],
#     [26, 'M', 'LOW', 'HIGH', 8.937],
#     [49, 'F', 'LOW', 'NORMAL', 26.743],
#     [62, 'M', 'HIGH', 'HIGH', 15.294],
#     [45, 'F', 'NORMAL', 'HIGH', 23.001],
#     [29, 'F', 'NORMAL', 'NORMAL', 11.497]
# ]
#
# new_labels = ['drugY', 'drugX', 'drugX', 'drugC', 'drugY', 'drugY', 'drugX', 'drugY', 'drugX', 'drugY', 'drugY', 'drugY', 'drugY', 'drugY', 'drugY', 'drugY', 'drugX', 'drugX', 'drugY', 'drugC', 'drugY', 'drugY', 'drugY', 'drugY', 'drugX']
#
# # Load the original data for label encoding
# original_data = pd.read_csv("EffectiveMedication.csv")
#
# # Preprocess the new data
# new_data_df = pd.DataFrame(new_data, columns=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'])
# new_data_df['Sex'] = preprocessing.LabelEncoder().fit(original_data['Sex']).transform(new_data_df['Sex'])
# new_data_df['BP'] = preprocessing.LabelEncoder().fit(original_data['BP']).transform(new_data_df['BP'])
# new_data_df['Cholesterol'] = preprocessing.LabelEncoder().fit(original_data['Cholesterol']).transform(new_data_df['Cholesterol'])
#
#
# # Predict on the new data
# predictions = model.predict(new_data_df.values)
#
# count = 0
# # Display the predictions
# for i in range(len(new_data)):
#     if predictions[i] == new_labels[i]:
#         count = count+1
#     print(f"Prediction for row {i + 1}: {predictions[i]}, the actual value is {new_labels[i]}")
#
# accuracy_percentage = (count / len(predictions)) * 100
# print(f"\nModel Accuracy: {accuracy_percentage:.2f}%")

# ==============================================================================================
# TRAINS AND ASSESSES PREDICTIONS FOR CARD FRAUD DECISION TREE
train_snap_ml_tree()

print('Exiting...')
