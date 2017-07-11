# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:01:19 2017

@author: fusta
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 14:49:05 2017

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
from PatchPreperation_function import PatchMaker 
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import scipy

patch_size=5
nclasses = 5
input_dim = patch_size
seed = input_dim - 1
np.random.seed(seed)
np.random.seed(123)  # for reproducibility

#patch size as pixel and number of classes
PatchMaker(patch_size,nclasses)

# 4. Load pre-shuffled MNIST data into train and test sets
#(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

dataset = np.loadtxt("test.csv", delimiter=',')

# load pima indians dataset
# split into input (X) and output (Y) variables
X = dataset[:,0:25]
Y = dataset[:,25]

#number of patches as number of images for trainign  and testing
npatches = X.shape[0]
#number of training data
ra= 0.2 #ratio of testing data respect to training data
#up to which patches number we will consider trainig
line_sprt=int(np.multiply(npatches,ra))
X_testing = X[:line_sprt,0:25]
X_training = X[line_sprt:,0:25]

Y_testing = Y[:line_sprt,]
Y_training= Y[line_sprt:,]
#number of testing data

X_training = X_training.reshape(X_training.shape[0], 1 , 5, 5)
X_testing = X_testing.reshape(X_testing.shape[0], 1, 5, 5)
X_training = X_training.astype('float32')
X_testing = X_testing.astype('float32')
X_training /= 255
X_testing /= 255

## 5. Preprocess input data
#X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
#X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
#X_train /= 255
#X_test /= 255
 
# 6. Preprocess class labels - need to think about it later!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Y_training = np_utils.to_categorical(Y_training, 6)
Y_testing = np_utils.to_categorical(Y_testing, 6)
 
# 7. Define model architecture
model = Sequential()
model.add(Convolution2D(32, 2, 2, activation='relu', input_shape=(1,5,5), dim_ordering='th'))
model.add(Convolution2D(32, 2, 2, activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))

# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
# 9. Fit model on training data
model.fit(X_training, Y_training, 
          batch_size=32, nb_epoch=10, verbose=1)
 
# 10. Evaluate model on test data
score = model.evaluate(X_testing, Y_testing, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

x_blindtest = X_testing
y_blindtest = Y_testing

# Use model to predict patch classes

#img = 'Image Crops\S03-3178 D3.tif'
#class_mask = 'Image Crops\S03-3178 D3_class.png'
#bg_mask = 'Image Crops/S03-3178 D3_bg.png'
#(x_blindtest, y_blindtest, grid_w, grid_h) = make_patches(img, class_mask, bg_mask, patch_size)

x_blindtest = x_blindtest.astype('float32')
x_blindtest /= 255

#y_blindtest = keras.utils.to_categorical(y_blindtest, 6)


# Get segmentation predictions from model, convert to categorical

y_pred = model.predict_classes(x_blindtest)
y_pred = y_pred.astype('float32')
y_pred = np_utils.to_categorical(y_pred, 6)

# Calculate metrics
##correct it later
#target_names = ['class 0', 'class 1', 'background']
#print(classification_report(y_blindtest, y_pred, target_names = target_names))

#.argmax(1) converts from categorical back to scalar of classes
print(confusion_matrix(y_blindtest.argmax(1), y_pred.argmax(1)))

print('\nEvaluation:')
score2 = model.evaluate(x_blindtest, y_blindtest, verbose=1)
print('\nTest Score: ', score2[0])
print('Test Accuracy:', score2[1])

#
##correct it later
## Display segmentation visually
#
Y_pred = y_pred.argmax(1).astype('float32')

# Reshape to grid of patches (patch representation of original image)
Y_pred = np.reshape(Y_pred, (np.multiply(Y_pred.shape[0],5), np.multiply(Y_pred.shape[0],5))) * (255/5)

scipy.misc.imresize(Y_pred,500)
scipy.misc.imresize(Y_pred,5)
Y_pred = Y_pred.reshape(Y_pred.shape[0],5,5) 

Image.fromarray(Y_pred).show()
