# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:01:19 2017

@author: fusta
"""
# 3. Import libraries and modules
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist
from PatchPreperation_multiple_slices import PatchMaker 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import scipy
import matplotlib.pyplot as plt
# Random Rotations
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras import backend as K
from keras.callbacks import ModelCheckpoint

K.set_image_dim_ordering('th')

patch_size=5
nclasses = 5
patchsize_sq = np.square(patch_size)
np.random.seed(123)  # for reproducibility
#create dataset
#PatchMaker(patch_size,nclasses)
#load dataset
dataset_training = np.loadtxt("training.csv", delimiter=',')
dataset_testing = np.loadtxt("testing.csv", delimiter=',')

# split into input (X) and output (Y) variables
X_training = dataset_training[:,0:patchsize_sq]
X_testing = dataset_testing[:,0:patchsize_sq]

Y_training = dataset_training[:,patchsize_sq]
Y_testing =dataset_testing[:,patchsize_sq]

#PICK A subset of your data for augmentaion
X_training_scar=np.empty([0,patchsize_sq])
Y_training_scar=np.empty([0,])

#DATA PREP. for AUGMENTATION - the subset of dataset which has more scar than normal
#X_training_added_augmented = X_training
#Y_training_added_augmented = Y_training

X_training_added_augmented = np.empty([0,patchsize_sq])
Y_training_added_augmented = np.empty([0,])

#add 5 times augmented data
for a in range(0,10):
    for t in range(0,len(X_training)):
        if Y_training[t]>0:
            X_training_c = np.reshape(X_training[t], (1,patchsize_sq))        
            Y_training_c = np.reshape(Y_training[t], (1,))        
            X_training_added_augmented=np.concatenate((X_training_added_augmented, X_training_c),axis=0)
            Y_training_added_augmented=np.concatenate((Y_training_added_augmented, Y_training_c),axis=0)                
     
#Reshape my dataset for my model       
X_training = X_training.reshape(X_training.shape[0], 1 , patch_size, patch_size)
X_training_added_augmented = X_training_added_augmented.reshape(X_training_added_augmented.shape[0], 1 , patch_size, patch_size)
X_testing = X_testing.reshape(X_testing.shape[0], 1, patch_size, patch_size)
X_training = X_training.astype('float32')
X_training_added_augmented = X_training_added_augmented.astype('float32')
X_testing = X_testing.astype('float32')
X_training /= 255
X_training_added_augmented /= 255
X_testing /= 255

nscar_trainig = np.count_nonzero(Y_training)
nscar_testing = np.count_nonzero(Y_testing)
print(nscar_trainig)
print(nscar_testing)

#visualize your test data before processing
Y_test_img = np.reshape(Y_testing, (int(187/patch_size), int(200/patch_size)))* (255/(nclasses-1))
plt.imshow(Y_test_img)

Y_training = np_utils.to_categorical(Y_training, nclasses)
Y_training_added_augmented = np_utils.to_categorical(Y_training_added_augmented, nclasses)
Y_testing = np_utils.to_categorical(Y_testing, nclasses)

#Data Augmentation Step
datagen = ImageDataGenerator(rotation_range=90)


# fit parameters from data
datagen.fit(X_training_added_augmented)

# 7. Define model architecture
model = Sequential()
model.add(Convolution2D(32, 2, 2, activation='relu', input_shape=(1,patch_size,patch_size), dim_ordering='th'))
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

#model.fit(X_training, Y_training, batch_size=32, nb_epoch=10, verbose=1)
checkpoint = ModelCheckpoint('weights.best.hdf5', monitor='val_acc', save_best_only=True, mode='max')
callbacks_list = [checkpoint]
    
#combine the augmented dataset with the original dataset
X_training_added_augmented=np.concatenate((X_training_added_augmented, X_training),axis=0)
Y_training_added_augmented=np.concatenate((Y_training_added_augmented, Y_training),axis=0)

history = model.fit_generator(datagen.flow(X_training_added_augmented, Y_training_added_augmented, batch_size=32),
                    steps_per_epoch=len(X_training_added_augmented) / 32, epochs=1, verbose=1, callbacks=callbacks_list)

#keras.fit_generator(datagen, samples_per_epoch=len(X_training), epochs=2)
#model.fit(X_training, Y_training, 
#batch_size=32, nb_epoch=10, verbose=1)
## 9. Fit model on training data
#history = model.fit(X_training, Y_training, 
#          batch_size=32, nb_epoch=2, verbose=1)
 
# 10. Evaluate model on test data
score = model.evaluate(X_testing, Y_testing, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
x_blindtest = X_testing
y_blindtest = Y_testing

x_blindtest = x_blindtest.astype('float32')
x_blindtest /= 255

# Get segmentation predictions from model, convert to categorical
y_pred = model.predict_classes(x_blindtest)
y_pred = y_pred.astype('float32')
y_pred = np_utils.to_categorical(y_pred, nclasses)

# Calculate metrics
##correct it later
#target_names = ['class 0', 'class 1', 'background']
print(classification_report(y_blindtest, y_pred))#, target_names = target_names))

#.argmax(1) converts from categorical back to scalar of classes
#print(confusion_matrix(y_blindtest.argmax(1), y_pred.argmax(1)))

print('\nEvaluation:')
score2 = model.evaluate(x_blindtest, y_blindtest, verbose=1)
print('\nTest Score: ', score2[0])
print('Test Accuracy:', score2[1])

##decode GT  <3<3<3
#decode your segmentation <3<3<3
y_pred = y_pred.argmax(1).astype('float32')
y_pred = np.reshape(y_pred, (int(185/patch_size), int(200/patch_size)))* (255/(nclasses-1))
plt.imshow(y_pred)
#

## list all data in history
#history = model.fit(...)
print(history.history.keys())

Y_training_added_augmented = Y_training_added_augmented.argmax(1).astype('float32')
n_Y_training= np.count_nonzero(Y_training_added_augmented)


model.layers
model.inputs
model.outputs