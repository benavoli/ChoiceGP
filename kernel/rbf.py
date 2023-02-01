#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 10:10:46 2022

@author: benavoli
"""
import numpy as np
from scipy.spatial.distance import cdist


def RBF(X1,X2,params,diag_=False):
    lengthscale=params['lengthscale']['value']
    variance   =params['variance']['value']
    if diag_==False:
        diffs = cdist(np.atleast_2d(X1)/ lengthscale, np.atleast_2d(X2) / lengthscale, metric='sqeuclidean')
    else:
        diffs = np.sum((np.atleast_2d(X1)/ lengthscale-np.atleast_2d(X2)/ lengthscale)*(np.atleast_2d(X1)/ lengthscale-np.atleast_2d(X2)/ lengthscale),axis=1)
    return variance*np.exp(-0.5 * diffs)