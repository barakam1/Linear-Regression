import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DS=pd.read_csv("Salary_Data.csv")
X=DS.iloc[:,:-1].values
Y=DS.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)

Y_pred=regressor.predict(X_test)