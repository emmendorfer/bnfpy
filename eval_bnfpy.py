# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 08:58:57 2021

A demonstration on the usage and evaluation of the Best Neighbor image sharpening filter.
A comparison to the USM filter is also shown.

The evaluation is perfored using PSNR and LPIPS metrics.
LPIPS can be found in https://github.com/richzhang/PerceptualSimilarity

An study including other benckmark image sets and base super-resolution methods 
is described in:
    
 L. R. Emmendorfer; J. C. de Jesus; T.T. Schein, 2021 (submitted)
 To Sharpen or Not? A Study on the Effect of Two Simple 
 Sharpening Filters Applied to Magnified Images.    

The BN filter was proposed in:

 L. R. Emmendorfer, 2020, July. An empirical evaluation of sharpening filters 
 applied to magnified images. In 2020 International Conference on Systems, 
 Signals and Image Processing (IWSSIP) (pp. 187-192). 
 URL http://iwssip2020.ic.uff.br/



@author: Leonardo R. Emmendorfer
"""
import torch
import lpips
import numpy as np
import math
from bnfpy import bnf
from skimage.filters import unsharp_mask
from imageeval import PSNR 
from skimage.io import imsave
import cv2


loss_fn = lpips.LPIPS(net='alex', spatial=False)

nomeint=["baboon", "barbara", "bridge", "coastguard",
"comic", "face", "flowers", "foreman", "lenna","man",
"monarch", "pepper", "ppt3", "zebra"]

fator=3
 
ex_d0=np.zeros((14),dtype=float)
ex_d1=np.zeros((14),dtype=float)
ex_d2=np.zeros((14),dtype=float)
resp1=np.zeros((14,4),dtype=float)
resp2=np.zeros((14,4),dtype=float)


for i in range(14):
    print(nomeint[i])
    nome=nomeint[i]
    refimg=cv2.imread('.//demo_img//'+nome+'_original_cropped.png')[:,:,::-1]
    (lx,ly,CL)=refimg.shape
    interpimg =cv2.imread('.//demo_img//'+nome+'_raisr.png')[:,:,::-1]
    (lxi,lyi,CLi)=interpimg.shape
    if lxi>lx or lyi>ly:
        interpimg=interpimg[0:lx,0:ly,:]  #the magnified image cannot be greater than the reference original image
    ex_ref = lpips.im2tensor(refimg)
    orig=cv2.imread('.//demo_img//'+nome+'_lr.png')[:,:,::-1]
    ex_p0 = lpips.im2tensor(interpimg)
    bnfimg=bnf(interpimg,orig,2,0.64,0) 
    ex_p1 = lpips.im2tensor(bnfimg)    
    imsave('.//output//'+nome+'_bnf.png' ,bnfimg)

    ex_d0[i] = loss_fn.forward(ex_ref,ex_p0)
    ex_d1[i] = loss_fn.forward(ex_ref,ex_p1)
    
    ex_p2 = np.copy(interpimg).astype(np.float)
    for j in range(3):
        ex_p2[...,j] = unsharp_mask(interpimg[...,j], radius=2, amount=0.62)*255.
              
    ex_p2 = ex_p2.astype(np.uint8)   
    imsave('.//output//'+nome+'_usm.png' ,ex_p2)

    ex_d2[i] = loss_fn.forward(ex_ref,lpips.im2tensor(ex_p2))    
     
    resp2[i,0]=PSNR(refimg[:,:,0:3],interpimg[:,:,0:3])
    resp2[i,1]=PSNR(refimg[:,:,0:3],bnfimg[:,:,0:3])
    resp2[i,2]=PSNR(refimg[:,:,0:3],ex_p2[:,:,0:3])
    resp1[i,3]=i+1;
    resp2[i,3]=i+1;    

resp1[:,0]=ex_d0
resp1[:,1]=ex_d1
resp1[:,2]=ex_d2
print("Results (avg $\pm$ s.s.d):")
print()

print("PSNR (higher is better):")
print("no filter: %10.2f"% resp2[:,0].mean(),end="$\pm$")
print("%10.2f"% math.sqrt(np.var(resp2[:,0])))

print("USM: %10.2f"% resp2[:,2].mean(),end="$\pm$")
print("%10.2f"% math.sqrt(np.var(resp2[:,2])))

print("BNF: %10.2f"% resp2[:,1].mean(),end="$\pm$")
print("%10.2f"% math.sqrt(np.var(resp2[:,1])))

print()
print("LPIPS (lower is better):")
print("no filter: %10.2f"% resp1[:,0].mean(),end="$\pm$")
print("%10.2f"% math.sqrt(np.var(resp1[:,0])))

print("USM: %10.2f"% resp1[:,2].mean(),end="$\pm$")
print("%10.2f"% math.sqrt(np.var(resp1[:,2])))

print("BNF: %10.2f"% resp1[:,1].mean(),end="$\pm$")
print("%10.2f"% math.sqrt(np.var(resp1[:,1])))


  