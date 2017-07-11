# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 16:54:32 2017

@author: fusta
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 15:30:09 2017

@author: fusta
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jul  7 10:31:18 2017

@author: fusta
"""
#def PatchPreperation(patch_size = 10, pid = '0485')
import os
import numpy
import SimpleITK
import matplotlib.pyplot as plt

def PatchMaker(patch_size, nclasses):  
    #reading mhd images
    LGE = SimpleITK.ReadImage("0485-LGE-cropped.mhd")
    scar = SimpleITK.ReadImage("0485-scar-cropped.mhd")
    
    #imshow mhd images
    idxSlice = 38
    #s
    #sitk_show((LGE[:,:,idxSlice]))
    #sitk_show((scar[:,:,idxSlice]))
    
    LGE_slice = (LGE[:,:,idxSlice])
    
    #convert a SimpleITK object into an array
    LGE_slice = SimpleITK.GetArrayFromImage(LGE[:,:,idxSlice])
    GTscar_slice = SimpleITK.GetArrayFromImage(scar[:,:,idxSlice])
    
    LGE_3D = SimpleITK.GetArrayFromImage(LGE)
    
    type(LGE_3D)
    #geT the size of an image
    LGE_3D.shape
    
    #size of 2D and 3D array
    numpy.ndarray.max(LGE_3D)
    
    #creating patches
    h_LGE = LGE_3D.shape[1]
    w_LGE = LGE_3D.shape[2]
    
#    patch_size = 5  
    
    #rows = range(0,h_LGE)
    #cols = range(0,w_LGE)
    
    #LGE_patch = LGE_slice[rows][:,cols]
    #plt.imshow(LGE_patch)
    
    #creating the patches
    nscar = 0
    llist= []
    type(llist)
    #def shape_input_data(patch_size = 10):
    for c in range(0, w_LGE-patch_size, patch_size):
        c0 = c
        c1 = c + patch_size    
        cols = range(c0,c1)    
        for r in range(0, h_LGE-patch_size, patch_size):
            r0 = r
            r1 = r + patch_size
            rows = range(r0,r1)
            LGE_patch = LGE_slice[rows][:,cols]
            GTscar_patch = GTscar_slice[rows][:,cols]
            patch_label=int(numpy.divide(numpy.multiply(numpy.divide(numpy.sum(GTscar_patch),numpy.square(patch_size)),nclasses),1))
#            patch_label=int((patch_label) == 5)*4
#            patch_label=int(numpy.divide(numpy.multiply(numpy.divide(250,500),100),20))
#            patch_label=int(numpy.sum(GTscar_patch)>numpy.divide(numpy.square(patch_size),2))
            patch_scalar = numpy.reshape(LGE_patch,(numpy.square(patch_size)))
            patch_scalar = numpy.append(patch_scalar,patch_label)
            llist.append(numpy.array(patch_scalar).tolist())
        #        patch_scalar[index] = numpy.reshape(LGE_patch,(1,400))
        #        if(numpy.sum(GTscar_patch)>=1):
            if(patch_label==1):
                nscar = nscar+1
            #plt.imshow(LGE_patch)
    #        print(cols)
    #        print(rows)
    numpy.savetxt('test.csv', llist ,fmt='%s', delimiter=',' ,newline='\r\n') 
    print(nscar)
#    
#
#patch_label=int(numpy.divide(numpy.multiply(numpy.divide(0,numpy.square(5)+1),5),1))
#patch_label