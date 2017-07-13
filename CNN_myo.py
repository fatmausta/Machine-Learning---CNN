# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 20:40:28 2017

@author: fusta
"""
#sigmoid example

# Create your first MLP in Keras
from PatchPreperation_blocks_windows import PatchMaker 
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import scipy
import matplotlib.pyplot as plt
# Random Rotations
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy

K.set_image_dim_ordering('th')
patch_size = 3
window_size = 25
nclasses = 5
patchsize_sq = np.square(patch_size)
windowsize_sq = np.square(window_size)
numpy.random.seed(windowsize_sq-1)
PatchMaker(patch_size, window_size, nclasses)
dataset_training = np.loadtxt("training.csv", delimiter=',')
dataset_testing = np.loadtxt("testing.csv", delimiter=',')
# split into input (X) and output (Y) variables
X_training = dataset_training[:,0:windowsize_sq]
X_testing = dataset_testing[:,0:windowsize_sq]
Y_training = dataset_training[:,windowsize_sq]
Y_testing =dataset_testing[:,windowsize_sq]
#Reshape my dataset for my model       
X_training = X_training.reshape(X_training.shape[0], 1 , window_size, window_size)
X_testing = X_testing.reshape(X_testing.shape[0], 1, window_size, window_size)
X_training = X_training.astype('float32')
X_testing = X_testing.astype('float32')
X_training /= 255
X_testing /= 255

#visualize your test data before processing
Y_test_img = np.reshape(Y_testing, (int(205/patch_size), int(190/patch_size)))* (255/(nclasses-1))
plt.imshow(Y_test_img)
Y_training = np_utils.to_categorical(Y_training, nclasses)
Y_testing = np_utils.to_categorical(Y_testing, nclasses)
model = Sequential()
model.add(Convolution2D(32, 2, 2, activation='relu', input_shape=(1,window_size,window_size), dim_ordering='th'))
model.add(Convolution2D(32, 2, 2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(nclasses, activation='softmax'))
# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.fit(X_training, Y_training, batch_size=32, nb_epoch=15, verbose=1)

# evaluate the model
scores = model.evaluate(X_training, Y_training)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

y_pred = model.predict_classes(X_testing)
y_pred = y_pred.astype('float32')
y_pred = np_utils.to_categorical(y_pred, nclasses)

print(classification_report(Y_testing, y_pred))#, target_names = target_names))

y_pred = y_pred.argmax(1).astype('float32')
y_pred = np.reshape(y_pred, (int(205/patch_size), int(190/patch_size)))* (255/(nclasses-1))
plt.imshow(y_pred)

model.summary()
