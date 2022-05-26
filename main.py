import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
#importing dataset & spliting it
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values
#encoding Male Female part of the dataset
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])
#encoding country part of the dataset
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))
#splitting dataset into test set and training set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
#Feature scalling !! Important
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)
#Building the ANN
ann = Sequential()
#First input layer & first hidden
ann.add(Dense(units=6, activation='relu'))
ann.add(Dense(units=6, activation='relu'))
#Output layer
ann.add(Dense(units=1, activation='sigmoid'))
#compiling the ann for training
ann.compile(optimizer= 'adam',loss='binary_crossentropy', metrics=['accuracy'])
#training ANN
ann.fit(x_train, y_train, batch_size=32, epochs= 100)
# ~ 86% accuracy
print(ann)
