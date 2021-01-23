# -*- coding: utf-8 -*-
"""
Created on Fri Jan 15 21:43:04 2021

@author: Leonardo Ramos Emmendorfer
"""
from numpy import mean
from numpy import inf
from math import sqrt 
from math import log10

def PSNR(i1,i2):
    mse=mean((i1-i2)**2)
    if mse==0:
        return inf
    max_pixel=255.0
    psnr=20*log10(max_pixel/sqrt(mse))
    return psnr
