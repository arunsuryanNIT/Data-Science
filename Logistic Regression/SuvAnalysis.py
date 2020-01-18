# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 20:17:12 2020

@author: ArunSuryan
"""

import pandas as pd
import seaborn as sns

dataset = pd.read_csv('Social_Network_Ads.csv')
sns.countplot(x = "Purchased", data = dataset)
sns.countplot(x = "Purchased", hue = "EstimatedSalary", data = dataset)
sns.countplot(x = "Purchased", hue = "Age", data = dataset)
dataset.isnull().sum()

#Data Wrangling
dataset.drop("User ID", axis = 1, inplace = True)
dataset.drop("Gender", axis = 1, inplace = True)


#Load the Data
X = dataset.iloc[:, 0:2].values
Y = dataset["Purchased"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.25, random_state = 1)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

from sklearn.linear_model import LogisticRegression

lrmodel = LogisticRegression()
lrmodel.fit(X_train, y_train)
predictions = lrmodel.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)

from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)
