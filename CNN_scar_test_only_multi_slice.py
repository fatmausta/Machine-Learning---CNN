# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:11:16 2017

@author: fusta
"""
# Create your first MLP in Keras
#from PatchPreperation_blocks_windows import PatchMaker 
#from PatchPreperation_blocks_windows_listappend import PatchMaker 
#from Introduce import PatchMaker, runCNNModel, DiceIndex, saveResults 
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
from skimage.util import view_as_windows
from skimage.util import view_as_blocks

patch_size = 3
window_size = 25
nclasses = 5
epocs = 1
patchsize_sq = np.square(patch_size)
windowsize_sq = np.square(window_size)
numpy.random.seed(windowsize_sq-1)
test_slice = range(33,34)
train_slice = range(44,45)
DATAPATH = 'DataCNNScar/'
datapath = 'C:\\Users\\fusta\\Dropbox\\1_Machine_Learning\\DataCNNScar\\'
def runCNNModel(dataset_training, dataset_testing, test_img_shape, pads, epocs, patch_size, window_size, nclasses, pid_test, datapath, test_slice):
    # preprocessing
    X_training = np.zeros((len(dataset_training),windowsize_sq))
    Y_training = np.zeros((len(dataset_training),1))
    X_testing = np.zeros((len(dataset_testing),windowsize_sq))
    Y_testing = np.zeros((len(dataset_testing),1))

    for p in range(0,len(dataset_testing)):
        X_testing[p]=dataset_testing[p][0]
        Y_testing[p]=dataset_testing[p][1]

    for p in range(0,len(dataset_training)):
        X_training[p]=dataset_training[p][0]
        Y_training[p]=dataset_training[p][1]

    #Reshape my dataset for my model       
    X_training = X_training.reshape(X_training.shape[0], 1 , window_size, window_size)
    X_testing = X_testing.reshape(X_testing.shape[0], 1, window_size, window_size)
    X_training = X_training.astype('float32')
    X_testing = X_testing.astype('float32')
    X_training /= 255
    X_testing /= 255
    Y_training = np_utils.to_categorical(Y_training, nclasses)
    Y_testing = np_utils.to_categorical(Y_testing, nclasses)
    model = keras.models.load_model('Model.h5')
    #predict classes
    y_pred = model.predict_classes(X_testing)    
    y_pred = y_pred.astype('float32')
    y_pred = np_utils.to_categorical(y_pred, nclasses)
    #evaluate the model    
    scores = model.evaluate(X_testing, y_pred)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    print(classification_report(Y_testing, y_pred))#, target_names = target_names))
    y_pred = y_pred.argmax(1).astype('float32')
  
    #save your model
#    model.save('Model.h5')
    y_pred_multi = []#np.empty((len(pid_test)))
    
    #reshaoaping then test data, test patients and test slsices
    for p in range(0, len(pid_test)):
        y_pred_multi.append(np.reshape(y_pred, (len(test_slice), int(test_img_shape[p][0]/patch_size), int(test_img_shape[p][1]/patch_size)))* (255/(nclasses-1)))
    #visualizing the results in a  smaller scale
    #for each patient
    for p in range(0, len(pid_test)):
        #for each slice
        for s in range(0, len(test_slice)):
            plt.figure()
            plt.imshow(y_pred_multi[p][s]) 
    #scaling the result of CNN algorithm to make it the full size image
    
    y_pred_scaled= np.empty((len(pid_test), test_img_shape[p][0], test_img_shape[p][1]))
#    y_pred_scaled_cropped = np.empty((len(pid_test), test_img_shape[p][0], test_img_shape[p][1]))
    y_pred_scaled_cropped = np.empty((len(pid_test),))
    
    for p in range(0, len(pid_test)):
        for s in range(0, len(test_slice)):
            y_pred_scaled[p] =  y_pred_multi[p][s].repeat(patch_size, axis =0).repeat(patch_size, axis =1)
            #cropping the initial pads before making the patches
    for p in range(0, len(pid_test)):
        for s in range(0, len(test_slice)):
            y_pred_scaled_cropped.append(y_pred_scaled[p][:-pads[p+len(pid_train)][0],:-pads[p+len(pid_train)][1]])
     
#    y_pred_scaled_cropped.rehape((len(pid_test),y_pred_scaled_cropped[1] )    
    return y_pred_scaled_cropped
#    y_pred_scaled= np.empty((y_pred_multi.shape[0]*patch_size,y_pred_multi.shape[1]*patch_size))
#    for s in test_slice:
#        y_pred_scaled =  y_pred_multi.repeat(patch_size, axis =0).repeat(patch_size, axis =1)
#    #cropping the initial pads before making the patches
#    y_pred_scaled_cropped= y_pred_scaled[:-pads[0],:-pads[1]]
#    return y_pred_scaled_cropped


def PatchMaker(patch_size, window_size, nclasses, pid_train, pid_test, test_slice, train_slice, datapath):  
    #reading mhd images
#    pid_train = ('0329')
#    pid_test = ('0485')        
    pid_all = np.concatenate((pid_train, pid_test))
    patch_labels_training=[]
    patch_labels_testing=[]    
    window_intensities_training=[]
    window_intensities_testing=[]
    test_img_shape = []#np.empty((len(pid_test),2))
    pads=[]
    for pid in pid_all:
        LGE = SimpleITK.ReadImage(datapath + pid + '\\' + pid + '-LGE-cropped.mhd')
        scar = SimpleITK.ReadImage(datapath + pid + '\\' + pid + '-scar-cropped.mhd')
        #convert a SimpleITK object into an array
        LGE_3D = SimpleITK.GetArrayFromImage(LGE)
        scar_3D = SimpleITK.GetArrayFromImage(scar) 
        d_LGE = LGE_3D.shape[0]
        h_LGE = LGE_3D.shape[1]
        w_LGE = LGE_3D.shape[2] 

        nonzero = 0
        #make windows size and patch size evenly dvideble 
        if (window_size-patch_size)%2 != 0:
            window_size +=1    
        #calculate the amount of padding for heaght and width of a slice for patches
        rem_w = w_LGE%patch_size
        w_pad=patch_size-rem_w      
        rem_h = h_LGE%patch_size
        h_pad=patch_size-rem_h    
        pads.append((h_pad,w_pad))
        
        if pid in pid_test:
            for sl in test_slice:
                LGE_padded_slice=numpy.lib.pad(LGE_3D[sl,:,:], ((0,h_pad),(0,w_pad)), 'constant', constant_values=(0,0))
                scar_padded_slice=numpy.lib.pad(scar_3D[sl,:,:], ((0,h_pad),(0,w_pad)), 'constant', constant_values=(0,0))  
                LGE_patches = view_as_blocks(scar_padded_slice, block_shape = (patch_size,patch_size))
                LGE_patches = numpy.reshape(LGE_patches,(LGE_patches.shape[0]*LGE_patches.shape[1],patch_size,patch_size)) 
                #re-pad your padded image before you make your windows
                padding = int((window_size - patch_size)/2)
                LGE_repadded_slice = numpy.lib.pad(LGE_padded_slice, ((padding,padding),(padding,padding)), 'constant', constant_values=(0,0))
                #done with the labels, now we will do our windows, 
                LGE_windows = view_as_windows(LGE_repadded_slice, (window_size,window_size), step=patch_size)
                LGE_windows = numpy.reshape(LGE_windows,(LGE_windows.shape[0]*LGE_windows.shape[1],window_size,window_size))        
                #for each patches: 
                for p in range(0,len(LGE_patches)):            
                    label=int(numpy.divide(numpy.multiply(numpy.divide(numpy.sum(LGE_patches[p]),numpy.square(patch_size)),nclasses),1))
                    label = numpy.reshape(label, (1,1))           
                    if label==nclasses:
                        label -=1 #mmake sure the range for the classes do not exceed the maximum
                    #making your window  intensities a single row
                    intensities = numpy.reshape(LGE_windows[p],(window_size*window_size))
                    intensities = numpy.reshape(intensities, (1,window_size*window_size))
                    patch_labels_testing.append(label)
                    window_intensities_testing.append(intensities)
                    
        else:
            for sl in train_slice:
                LGE_padded_slice=numpy.lib.pad(LGE_3D[sl,:,:], ((0,h_pad),(0,w_pad)), 'constant', constant_values=(0,0))
                scar_padded_slice=numpy.lib.pad(scar_3D[sl,:,:], ((0,h_pad),(0,w_pad)), 'constant', constant_values=(0,0))  
                LGE_patches = view_as_blocks(scar_padded_slice, block_shape = (patch_size,patch_size))
                LGE_patches = numpy.reshape(LGE_patches,(LGE_patches.shape[0]*LGE_patches.shape[1],patch_size,patch_size)) 
                #re-pad your padded image before you make your windows
                padding = int((window_size - patch_size)/2)
                LGE_repadded_slice = numpy.lib.pad(LGE_padded_slice, ((padding,padding),(padding,padding)), 'constant', constant_values=(0,0))
                #done with the labels, now we will do our windows, 
                LGE_windows = view_as_windows(LGE_repadded_slice, (window_size,window_size), step=patch_size)
                LGE_windows = numpy.reshape(LGE_windows,(LGE_windows.shape[0]*LGE_windows.shape[1],window_size,window_size))        
                #for each patches: 
                for p in range(0,len(LGE_patches)):            
                    label=int(numpy.divide(numpy.multiply(numpy.divide(numpy.sum(LGE_patches[p]),numpy.square(patch_size)),nclasses),1))
                    label = numpy.reshape(label, (1,1))           
                    if label==nclasses:
                        label -=1 #mmake sure the range for the classes do not exceed the maximum
                    #making your window  intensities a single row
                    intensities = numpy.reshape(LGE_windows[p],(window_size*window_size))
                    intensities = numpy.reshape(intensities, (1,window_size*window_size))
                    patch_labels_training.append(label)
                    window_intensities_training.append(intensities)        
        
        if pid in pid_test:
            print(pid)
            test_img_shape.append(LGE_padded_slice.shape)        #Now, intensities=windows and patches labels will be comboined
            
        training_data= list(zip(numpy.uint8(window_intensities_training),numpy.uint8(patch_labels_training)))
        testing_data= list(zip(numpy.uint8(window_intensities_testing),numpy.uint8(patch_labels_testing)))  
    return training_data, testing_data, test_img_shape, pads, pid_all

#Dice Calculation
def DiceIndex(BW1, BW2):
    BW1 = BW1.astype('float32')
    
    BW2 = BW2.astype('float32')
    #elementwise multiplication
    t= (np.multiply(BW1,BW2))
    total = np.sum(t)
    DI=2*total/(np.sum(BW1)+np.sum(BW2))
    DI=DI*100
    return DI
   
##MAIN SECTION    
##pids = ('0485', '0329')#,'0364', '0417', '0424', '0450', '0473', '0485', '0493', '0494', '0495', '0515', '0519', '0529', '0546', '0562', '0565', '0574', '0578', '0587', '0591', '0601', '0632', '0715', '0730', '0917', '0921', '0953', '1036', '1073', '1076', '1115', '1166', '1168', '1171', '1179')
pid_train = np.array(['0329','0364','0417', '0424', '0450', '0473', '0493', '0494', '0495', '0515'])#, '0519', '0529', '0546', '0562', '0565', '0574', '0578', '0587', '0591', '0601', '0632', '0715', '0730', '0917', '0921', '0953', '1036', '1073', '1076', '1115', '1166', '1168', '1171', '1179'])
pid_test = np.array([('0485')])#' '0485', '1166', '1168', '1171', '1179')])#for pid in pids:
(dataset_training, dataset_testing, test_img_shape, pads, pid_all) = PatchMaker(patch_size, window_size, nclasses, pid_train, pid_test, test_slice, train_slice, datapath)
y_pred_scaled_cropped = runCNNModel(dataset_training, dataset_testing, test_img_shape, pads, epocs, patch_size, window_size, nclasses, pid_test, datapath, test_slice)

#read the GT as single slice for comparison
p=0
for pid in pid_test:
    scarGT = SimpleITK.ReadImage(datapath + pid + '\\' + pid + '-scar-cropped.mhd')
    scar3D = sitk.GetArrayFromImage(scarGT)
    BW2 = y_pred_scaled_cropped[p]/255# = y_pred_scaled_cropped
    BW1 = scar3D[test_slice,:,:].reshape((BW2.shape))
    Dice = DiceIndex(BW1, BW2)
    print('Training Slices: %s Testing Slices: %s' %(train_slice, test_slice))
    print('Dice for pid %s: %2.3f percent \n\n' %(pid, Dice))
    p+=1