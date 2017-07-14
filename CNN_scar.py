# -*- coding: utf-8 -*-
"""
Created on Fri Jul 14 00:14:02 2017

@author: fusta
"""
# Create your first MLP in Keras
from PatchPreperation_blocks_windows import PatchMaker 
import numpy as np
import keras
import SimpleITK
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
import SimpleITK as sitk

K.set_image_dim_ordering('th')
patch_size = 3
window_size = 25
nclasses = 5
patchsize_sq = np.square(patch_size)
windowsize_sq = np.square(window_size)
numpy.random.seed(windowsize_sq-1)
(training,testing)=PatchMaker(patch_size, window_size, nclasses)
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
#Y_test_img = np.reshape(Y_testing, (int(205/patch_size), int(190/patch_size)))* (255/(nclasses-1))
#plt.imshow(Y_test_img)
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
model.fit(X_training, Y_training, batch_size=32, nb_epoch=2, verbose=1)

# evaluate the model
scores = model.evaluate(X_training, Y_training)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

y_pred = model.predict_classes(X_testing)
y_pred = y_pred.astype('float32')
y_pred = np_utils.to_categorical(y_pred, nclasses)

print(classification_report(Y_testing, y_pred))#, target_names = target_names))

y_pred = y_pred.argmax(1).astype('float32')
y_pred_multi = np.reshape(y_pred, (6,int(205/patch_size), int(190/patch_size)))* (255/(nclasses-1))
for t in range(0,y_pred_multi.shape[0]):
    plt.figure()
    plt.imshow(y_pred_multi[t])

#I need them from Patch maker
LGE
h_pad
w_pad
patch_size =3

#take some slices of my GT
y_pred_empty= np.empty((d_LGE,h_LGE,w_LGE))

scarGT = SimpleITK.ReadImage("0485-scar-cropped.mhd")

y_pred_multi = SimpleITK.GetArrayFromImage(scarGT)

y_pred_empty[range(39,45),:,:]=y_pred_multi[range(39,45),:,:]
scarGTsome = sitk.GetImageFromArray(y_pred_empty)

spacing = np.array(list(LGE.GetSpacing()))
origin = np.array(list(LGE.GetOrigin()))
direction = np.array(list(LGE.GetDirection()))
SimpleITK.Image.SetOrigin(scarGTsome,origin)
SimpleITK.Image.SetSpacing(scarGTsome,spacing)
SimpleITK.Image.SetDirection(scarGTsome,direction)
SimpleITK.WriteImage(scarGTsome, '0485-scar.mhd')





#read the pre-saved small scale image
scarCNN = SimpleITK.ReadImage("0485-scar-CNN.mhd")
y_pred_multi = SimpleITK.GetArrayFromImage(scarCNN)

h_LGE = y_pred_multi.shape[1]*patch_size - h_pad
w_LGE = y_pred_multi.shape[2]*patch_size - w_pad

#y_pred_scaled= np.empty((y_pred_multi.shape[0],y_pred_multi.shape[1]*patch_size,y_pred_multi.shape[2]*patch_size))
y_pred_scaled= np.empty((d_LGE,y_pred_multi.shape[1]*patch_size,y_pred_multi.shape[2]*patch_size))

#scaling the result of CNN algorithm to make it the full size image
#for s in range(0, y_pred_multi.shape[0]):
  
    
some_LGE = LGE_3D[range(39,45),:,:]

for s in range(39, 45):
    #scale your predicted iamge
    y_pred_scaled[s] =  y_pred_multi[s-39].repeat(patch_size, axis =0).repeat(patch_size, axis =1)

#cropping the initial pads before making the patches
y_pred_scaled_cropped= y_pred_scaled[:,:-h_pad,:-w_pad]
scarCNNo = sitk.GetImageFromArray(y_pred_scaled_cropped)

spacing = np.array(list(LGE.GetSpacing()))
origin = np.array(list(LGE.GetOrigin()))
direction = np.array(list(LGE.GetDirection()))


SimpleITK.Image.SetOrigin(scarCNNo,origin)
SimpleITK.Image.SetSpacing(scarCNNo,spacing)
SimpleITK.Image.SetDirection(scarCNNo,direction)
SimpleITK.WriteImage(scarCNNo, '0485-scar-CNN-full-size.mhd')

SimpleITK.Image.SetOrigin(scarCNN,origin)
SimpleITK.Image.SetSpacing(scarCNN,spacing)
SimpleITK.Image.SetDirection(scarCNN,direction)
SimpleITK.WriteImage(scarCNN, '0485-scar-CNN.mhd')


BW1 = np.array([[1,1],[1,1]])
BW2 = np.array([[1,1],[1,1]])
#Dice Calculation
def DiceIndex(BW1, BW2):
    BW1 = BW1.astype('float32')
    BW2 = BW2.astype('float32')
    #elementwise multiplication
    t= (np.multiply(BW1,BW2))
    total = np.sum(t)
    DI=2*total/(np.sum(BW1)+np.sum(BW2))
    DI=DI*100


#prepare the header file and write the mhd image
spacing = np.array(list(LGE.GetSpacing()))
origin = np.array(list(LGE.GetOrigin()))
direction = np.array(list(LGE.GetDirection()))
scarCNN = sitk.GetImageFromArray(y_pred_multi)
scarCNN.SetSpacing(spacing)
scarCNN.SetOrigin(origin)
scarCNN.SetDirection(direction)


#
#SimpleITK.WriteImage(scarCNN, '0485-scar-CNN.mhd')
#scarCNN.SetSpacing(spacing)
#
#
#headertext = 'ObjectType = Image\nNDims = 3\nDimSize = 77 201 187\nElementType = MET_USHORT\nElementDataFile = scarCNN.raw\n'    #(this tag must be last in a MetaImageHeader)
#
#SimpleITK.
#spacing = np.array(list(reversed(LGE.GetSpacing())))
#spacing = 'ElementSpacing = ' + str(spacing[0]) + ' ' + str(spacing[1]) + ' ' + str(spacing[2]) 
##set dimensions
#DimSize='DimSize = ' + str(y_pred_multi.shape[0]) + ' ' + str(scarCNN.shape[0]) + ' ' + str(scarCNN.shape[1])# + ' ' + str(scarCNN.shape[2])
#ObjectType = Image
#ElementType = MET_USHORT
#ElementDataFile = scarCNN.raw
#ElementSpacing = 1.29999995232 0.625 0.625
#
#
#scar = sitk.GetArrayFromImage(LGE)
#img = sitk.GetImageFromArray(scar)
#img.GetSize()
#



#ry to write the results
SimpleITK.WriteImage(scarCNN, '0485-scar-CNN.mhd')
#convert a SimpleITK object into an array
SimpleITK.GetImageFromArray(LGE_3D)    
type(scarCNN)

model.summary()
