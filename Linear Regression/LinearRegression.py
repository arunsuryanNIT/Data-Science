# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 13:40:24 2020

@author: ArunSuryan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Read Data
dataset = pd.read_csv('headbrain.csv')
print(dataset.shape)
dataset.head()

#Relationship between size of heads and weight of brains
X = dataset['Head Size(cm^3)'].values
Y = dataset['Brain Weight(grams)'].values

#Mean of the values
mean_x = np.mean(X)
mean_y = np.mean(Y)

#Total number of values
n = len(X)

#Using the formula to calculate
numer = 0
denom = 0

for i in range(n):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2

m = numer / denom
c = mean_y - (m * mean_x)
#Plotting the values
max_x = np.max(X) + 100
min_x = np.min(X) - 100

#Calculating the line values
x = np.linspace(min_x, max_x, 237) #Creates an interval of 1000 values from main_x to max_x
y = c + m * x
plt.plot(x, y, color = '#58B970', label = 'Regression Line')
plt.scatter(X, Y, c = '#ef5423', label = 'Scatter Plot')
plt.xlabel('Header Size in cm3')
plt.ylabel('Brain Weight in grams')
plt.legend()
plt.show()

#Evaluating the model 
ss_t = 0 #Total sum of square
ss_r = 0 #Sum of residuals
for i in range(n):
    Ypred = c + m * X[i]
    ss_t += (Y[i] - mean_y) ** 2
    ss_r += (Y[i] - Ypred) ** 2

r2 = 1 - (ss_r/ss_t)
print(r2)

#Using ML
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

X = X.reshape(n, 1)

#Create the model
reg = LinearRegression()
#Fitting the model
reg = reg.fit(X, Y)
#Prediction
Y_pred = reg.predict(X)

#Calculate RMSE and R2 scores
mse = mean_squared_error(Y, Y_pred)
rmse = np.sqrt(mse)
r2_score = reg.score(X, Y)

print(r2_score)

