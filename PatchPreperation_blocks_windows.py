# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:31:34 2017

@author: fusta
"""
# -*- coding: utf-8 -*-

import os
import numpy as np
import numpy
import SimpleITK
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
from skimage.util import view_as_blocks

def PatchMaker(patch_size, window_size, nclasses):  
    #reading mhd images
    LGE = SimpleITK.ReadImage("0485-LGE-cropped.mhd")
    scar = SimpleITK.ReadImage("0485-myo-cropped.mhd")
    
    #slice number of theb twsting data
    test_slice = 39
        
    #convert a SimpleITK object into an array
    LGE_3D = SimpleITK.GetArrayFromImage(LGE)
    scar_3D = SimpleITK.GetArrayFromImage(scar)
    type(LGE_3D)
    #geT the size of an image
    
    LGE_3D.shape
    
    #creating patches
    d_LGE = LGE_3D.shape[0]
    h_LGE = LGE_3D.shape[1]
    w_LGE = LGE_3D.shape[2]
#    LGE_slice = LGE_3D[test_slice,:,:]
      
    nonzero = 0
    #make sure your patch size and window size is both even or both odd
#    nclasses=5
#    patch_size = 5
#    window_size = 24
    if (window_size-patch_size)%2 != 0:
        #make windows size and patch size evenly dvideble 
        window_size +=1
    
    #calculate the amount of padding for heaght and width of a slice for patches
    rem_w = w_LGE%patch_size
    w_pad=patch_size-rem_w      
    rem_h = h_LGE%patch_size
    h_pad=patch_size-rem_h
      
    patch_labels_training=numpy.empty([1,1])
    patch_labels_testing=numpy.empty([1,1])
    windowslist_training=numpy.empty([1,window_size*window_size])
    windowslist_testing=numpy.empty([1,window_size*window_size])
    for sl in range(38,50):

        LGE_padded_slice=numpy.lib.pad(LGE_3D[sl,:,:], ((0,h_pad),(0,w_pad)), 'constant', constant_values=(0,0))
        scar_padded_slice=numpy.lib.pad(scar_3D[sl,:,:], ((0,h_pad),(0,w_pad)), 'constant', constant_values=(0,0))
        
        LGE_patches = view_as_blocks(scar_padded_slice, block_shape = (patch_size,patch_size))
        LGE_patches = numpy.reshape(LGE_patches,(LGE_patches.shape[0]*LGE_patches.shape[1],patch_size,patch_size))
 
        #re-pad your padded image before you make your windows
        #calculate padding for each side
        padding = int((window_size - patch_size)/2)
        LGE_padded_slice = numpy.lib.pad(LGE_padded_slice, ((padding,padding),(padding,padding)), 'constant', constant_values=(0,0))
        #done with the labels, now we will do our windows, 
        LGE_windows = view_as_windows(LGE_padded_slice, (window_size,window_size), step=patch_size)
        LGE_windows = numpy.reshape(LGE_windows,(LGE_windows.shape[0]*LGE_windows.shape[1],window_size,window_size))
        
        #for each patches: 
        for p in range(0,len(LGE_patches)):
            
            label=int(numpy.divide(numpy.multiply(numpy.divide(numpy.sum(LGE_patches[p]),numpy.square(patch_size)),nclasses),1))
            label = numpy.reshape(label, (1,1))
            
            if label==nclasses:
                label -=1 #mmake sure the range for the classes do not exceed the maximum
#        #count non-zero labels 
            if label!= 0 : 
                nonzero += 1
            #making your window  intensities a single row
            intensities = numpy.reshape(LGE_windows[p],(window_size*window_size))
            intensities = numpy.reshape(intensities, (1,window_size*window_size))
        
            if sl == test_slice:
                patch_labels_testing=numpy.concatenate((patch_labels_testing,label), axis=0)
                windowslist_testing =numpy.concatenate((windowslist_testing,intensities), axis=0)

            else:
                patch_labels_training = numpy.concatenate((patch_labels_training, label), axis=0)
                windowslist_training =numpy.concatenate((windowslist_training, intensities), axis=0)
       
    #removing the nan object we had at the beginning  of our array
    patch_labels_training = patch_labels_training[1:]
    patch_labels_testing = patch_labels_testing[1:]    
    windowslist_training = windowslist_training[1:]
    windowslist_testing = windowslist_testing[1:]
# 
    #Now, intensities=windows and patches labels will be comboined
    training_data=numpy.concatenate((windowslist_training,patch_labels_training), axis=1)
    testing_data=numpy.concatenate((windowslist_testing,patch_labels_testing), axis=1)
    training_data=numpy.uint8(training_data)
    testing_data=numpy.uint8(testing_data)
    
    numpy.savetxt('training.csv', training_data ,fmt='%s', delimiter=',' ,newline='\r\n') 
    numpy.savetxt('testing.csv', testing_data ,fmt='%s', delimiter=',' ,newline='\r\n') 
    print(nonzero)                        
    
    
#PatchMaker(5, 25, 5)