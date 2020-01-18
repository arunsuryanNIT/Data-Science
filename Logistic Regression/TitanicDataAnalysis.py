# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 15:45:12 2020

@author: ArunSuryan
"""

import pandas as pd
import seaborn as sns

dataset = pd.read_csv('Titanic.csv')
print(dataset.shape)
dataset.head(10)

sns.countplot(x = "Survived", data = dataset)
sns.countplot(x = "Survived", hue = "Sex", data = dataset)
sns.countplot(x = "Survived", hue = "Pclass", data = dataset)
dataset["Age"].plot.hist()
dataset["Fare"].plot.hist(bin = 20, figsize = (10, 5))
dataset.info()

#Data wrangling
dataset.isnull()
dataset.isnull().sum()
sns.boxplot(x = "Pclass", y = "Age", data = dataset)
dataset.drop("Cabin", axis = 1, inplace = True)
dataset.dropna(inplace = True)
sex = pd.get_dummies(dataset['Sex'], drop_first = True)
sex.head(5)
embark = pd.get_dummies(dataset["Embarked"], drop_first = True)
Pcl = pd.get_dummies(dataset["Pclass"], drop_first = True)
dataset = pd.concat([dataset, sex, embark, Pcl], axis = 1)
dataset.drop(["Sex", "Embarked", "PassengerId", "Name", "Ticket" , "Pclass"] , axis = 1, inplace = True)

#Train Data
X = dataset.drop("Survived", axis = 1)
Y = dataset["Survived"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

from sklearn.linear_model import LogisticRegression 
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report
classification_report(y_test, predictions)
from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, predictions)
from sklearn.metrics import accuracy_score
accuracy_score(y_test, predictions)