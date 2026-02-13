# -*- coding: utf-8 -*-
"""
Created on Fri Feb 13 14:21:38 2026

@author: tsow2
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 
from tensorflow.keras import datasets, layers , models 
import keras as ks 
import os 

os.chdir(r"C:\Users\tsow2\OneDrive\Bureau\cours2iemeannee\Projets\Projet personnel\Projet2_cnn")


# Chargement des données
(X_train , y_train) , (X_test,y_test)=datasets.cifar10.load_data()
X_train.shape  #(50000, 32, 32, 3)
X_test.shape #(10000, 32, 32, 3)
y_train.shape
y_train[:5]
y_train=y_train.reshape(-1,)
y_train[:5]



# voir une image 
plt.figure(figsize=(10,2))
plt.imshow(X_train[1])

classes=["airplane", "automobile" , "bird" , "cat" , "deer" , "dog", "frog" , "horse" , "ship" , "truck"]




def plot_sample(x,y, index): 
    plt.figure(figsize=(10 , 2))
    plt.imshow(X_train[index] )
    plt.xlabel(classes[y[index]])
plot_sample(X_train , y_train , 2)


# Normalisation de notre data
X_train = X_train / 255.0
X_test = X_test / 255.0



## utilisation d'un cnn 
from keras.optimizers import Adam , SGD


cnn=ks.Sequential([
    #cnn
    ks.layers.Conv2D(filters=32 ,kernel_size=(3,3) , activation="relu", input_shape=(32,32,3) 
                    ), 
    ks.layers.MaxPooling2D((2,2) ), 
    layers.Flatten() , 
    layers.Dense(64, activation="relu")
,
layers.Dense(10 , activation="softmax")



])
cnn.compile(optimizer="Adam" , loss="sparse_categorical_crossentropy", metrics=["accuracy"])
cnn.fit(X_train , y_train , epochs=10)


cnn.evaluate(X_test,y_test)



# chargement du modèle dans pickle


import pickle

pickle.dump(cnn , open("model_cnn.pkl" , "wb"))

#cnn.save("model_cnn.h5")
cnn.save("model_cnn.keras")


