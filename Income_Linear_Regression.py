# -*- coding: utf-8 -*-
"""
@author: vishn

"""
#IMPORT THE LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error,r2_score

#LOAD THE INCOME V EXPERIENCE DATASET
dataset=pd.read_csv("Income_Data.csv")

#OUR FEATURE IS "EXPERIENCE" AND LABEL IS "INCOME"
feature_X = dataset.iloc[:,0]
label_Y = dataset.iloc[:,-1]

#Split the dataset into TRAINING and TEST sets
#TRAINING dataset is used to train the data
#TEST is used to check if our model is working right and calculate the loss
X_train = feature_X[:-10].values.reshape(-1,1)
X_test = feature_X[-10:].values.reshape(-1,1)

Y_train = label_Y[:-10].values.reshape(-1,1)
Y_test = label_Y[-10:].values.reshape(-1,1)

#Create LINEAR REGRESSION object
regr = linear_model.LinearRegression()

#Train the model using TRAINING sets
regr.fit(X_train,Y_train)

#Make predictions using the TEST set
Y_pred = regr.predict(X_test)

#The Coeeficients
print("Coefficients: \n",regr.coef_)

#Find the loss using MEAN SQUARE ERROR
print("Mean squared error: %.2f"%mean_squared_error(Y_test,Y_pred))

#PLOT THE OUTPUTS
plt.scatter(X_test,Y_test,color='black')
plt.plot(X_test,Y_pred,color="red",linewidth=3)

plt.xticks(())
plt.yticks(())

#IF INPUT IS CUSTOM INPUT
my_exp=[]
my_exp.append((int(input())))#Enter your experience
reshaped_my_exp = np.reshape(my_exp,(-1,1))
custom_income_pred = regr.predict(reshaped_my_exp)
print("Income (in Rs.) is: ")
print(custom_income_pred)




