# -*- coding: utf-8 -*-


# ******     Data Preprocessing Template     ***** #

#import libraries
import numpy as np
import pandas as pd


name_csv = input('What is the name of the .csv file you want to import?\nNote: Do not include .csv in name\n')

#Import the dataset
dataset = pd.read_csv(name_csv + '.csv')    #Change name of file if needed


#Create ia matrix of independant variables
#default: all rows except the last row
X = dataset.iloc[:, :-1].values


#Create ia matrix of dependant variables
#default: the last row
y = dataset.iloc[:, -1].values


#Replacing missing data (if any) 
#Default strategy: mean
#exclude all categorical data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])


#Encoding Categorical Data in Independant Variable matrix
#Default: encodes the first column.
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
coltransform = ColumnTransformer(transformers = [('encoder', OneHotEncoder(), [0])],
                                 remainder = 'passthrough')
X = np.array(coltransform.fit_transform(X))


#Encoding Categorical Data in Dependant Variable matrix
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)


#Splitting to Training Set and Test Set
#default test size: 20%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)


#Feature Scaling using standardisation
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)