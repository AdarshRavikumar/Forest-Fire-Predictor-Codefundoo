#!/usr/bin/env python3

import cgitb
cgitb.enable()    
print("Content-Type: text/html;image/jpg\r\n\r\n")

import sys
import os

import numpy as np

import matplotlib

matplotlib.use('Agg')


import pandas as pd
from scipy.misc import imread
dataset=pd.read_csv("forest_fires.csv")
dataset=dataset.drop(["month","day"],axis=1)
X=dataset.iloc[:,2:10].values
y=dataset.iloc[:,10].values

#plotting the region
img=imread('sample.jpg')


from io import StringIO,BytesIO
import matplotlib.pyplot as plt
import base64
#html1 = """<img src="/sample.jpg" alt="FUCK" width="300" height="200"/> """
		
		
#print(html1)



# split into test and train
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
#print(len(x_train), len(y_train))

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 1)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)
explained_variance = pca.explained_variance_ratio_


from sklearn.ensemble import RandomForestRegressor
random_forest = RandomForestRegressor()
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)

x11=x_test[:]
z1=y_pred[:]

from sklearn.metrics import mean_squared_error
mse= mean_squared_error(y_test,y_pred)

def doit():
    
    #plt.gca().invert_yaxis()
        plt.scatter(x11,z1,color='Blue')
        plt.xticks(np.arange(0, 8, 1.0))
        plt.yticks(np.arange(0, 9, 1.0))
        
    #plt.gca().invert_yaxis() 
    
        plt.imshow(img, zorder=0, extent=[0.5, 10.0, 1.0, 7.0],origin='lower')
        
       
       
    
        

doit()


#print(y_pred)


#plt.gca().invert_yaxis()


   
