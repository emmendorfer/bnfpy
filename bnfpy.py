# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 22:22:41 2021

@author: Leonardo R. Emmendorfer
"""

import numpy as np
from math import sqrt
from skimage.transform import resize


def difcolor(V,R):
    (npt,ncor)=V.shape
    dif=np.zeros((npt))
    for i in range(npt):
        dif[i]=sqrt(sum((V[i,:]-R)**2)/ncor)
    return dif

def mergeimg(i1,i2,param):
    return(np.round(i1*param+i2*(1-param)).astype(np.uint8))


def bnf(magnified,lrimg,radius=2,beta=1,offset=0):
    '''Computes the Best Neighbor (BN) filter from a magnified image.'''

# The BN filter was proposed in:
# L. R. Emmendorfer, 2020, July. An empirical evaluation of sharpening filters 
# applied to magnified images. In 2020 International Conference on Systems, 
# Signals and Image Processing (IWSSIP) (pp. 187-192). 
# URL http://iwssip2020.ic.uff.br/


# Usage: bnf(magnified,lrimg,radius=2,beta=1,offset=0)

#   Besides the input <magnified> image, also and its low-resolution original must be pprovided (lrimge). 
#   The parameter <lrimage> can, alternatively, represent a Nearest Neighbor (NN) interpolation of ]
# same the LR source. 
#   The parameter <radius> defines the size of the filter window, and <beta> is the intensity 
# of the filter.
#   Optionally, an offset can be provided, wihch will be applied to adjust the location of a NN  
# interpolation given in <lrimage>.

    if beta==0:
        return(magnified)
    (lx,ly,CL)=magnified.shape
    (Im,In,NCL)=lrimg.shape
    radius=round(radius)	
    if Im>=lx or In>=ly:    # is lrimg a NN interpolation fom the LR original image?
       orig2=lrimg
       orig2[0:lx-offset,0:ly-offset,:] = lrimg[offset:lx,offset:ly,:]
       lrimg=orig2[0:lx,0:ly,:]
               
    else: # lrimg is the LR input prior to magnification.
        lrimg=np.round(resize(lrimg, (lx, ly), order=0)*255)
        lrimg=lrimg.astype(np.uint8)
    Gab=lrimg
    IN=np.zeros((lx,ly,CL))

    for ii in range(lx):
        for jj in range(ly):			
            lim1=ii-radius
            lim2=ii+radius+1  # this makes the actual range be [lim1,lim2]
            lim3=jj-radius
            lim4=jj+radius+1  # this makes the actual range be [lim1,lim2]
            if (lim1<0):
                lim1=0
            if (lim2>lx):
                lim2=lx		
            if (lim3<0):
                lim3=0
            if (lim4>ly):
                lim4=ly
            Vsel=np.zeros(((lim2-lim1)*(lim4-lim3),CL))
            for CID in range(CL):
                Vsel[:,CID]=np.reshape(Gab[lim1:lim2,lim3:lim4,CID],((lim2-lim1)*(lim4-lim3)))			
            R=magnified[ii,jj,:]			  
            dif=difcolor(Vsel,R) #euclidean distances in RGB				
            pos=np.argmin(dif)
            IN[ii,jj,:]=Vsel[pos,:]  # the most similar pixel
    if beta<1:
        return mergeimg(IN,magnified,beta)  
    else:
        return np.round(IN).astype(np.uint8)

         