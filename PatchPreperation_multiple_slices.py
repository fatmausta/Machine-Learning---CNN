# -*- coding: utf-8 -*-
"""
Created on Mon Jul 10 14:52:52 2017

@author: fusta
"""

import os
import numpy
import SimpleITK
import matplotlib.pyplot as plt

def PatchMaker(patch_size, nclasses):  
    #reading mhd images
    LGE = SimpleITK.ReadImage("0485-LGE-cropped.mhd")
    scar = SimpleITK.ReadImage("0485-scar-cropped.mhd")
    
    #slice number of theb twsting data
    test_slice = 38
        
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
        
    list_training= []
    list_testing= []
    #def shape_input_data(patch_size = 10):

    for sl in range(0,d_LGE):
        for c in range(0, w_LGE-patch_size, patch_size):
            c0 = c
            c1 = c + patch_size    
#            cols = [c0:c1]   
            for r in range(0, h_LGE-patch_size, patch_size):
                r0 = r
                r1 = r + patch_size
#                rows = [r0:r1]
                LGE_patch = LGE_3D[sl,r0:r1,c0:c1]
                GTscar_patch = scar_3D[sl,r0:r1,c0:c1]
                patch_label=int(numpy.divide(numpy.multiply(numpy.divide(numpy.sum(GTscar_patch),numpy.square(patch_size)),nclasses),1))
                if patch_label==nclasses:
                    patch_label -=1 #mmake sure the range for the classes do not exceed the maximum
                patch_scalar = numpy.reshape(LGE_patch,(numpy.square(patch_size)))
                patch_scalar = numpy.append(patch_scalar,patch_label)
                if sl == test_slice:
                    list_testing.append(numpy.array(patch_scalar).tolist())
                else:
                    list_training.append(numpy.array(patch_scalar).tolist())
                    
    numpy.savetxt('training.csv', list_training ,fmt='%s', delimiter=',' ,newline='\r\n') 
    numpy.savetxt('testing.csv', list_testing ,fmt='%s', delimiter=',' ,newline='\r\n') 
    
    