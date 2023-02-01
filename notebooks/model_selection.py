#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 08:28:26 2022

@author: benavoli
"""

import numpy as np

import psiloo as psiloo
import pandas as pd
import sys
sys.path.append('../GPpref/')
from model.erroneousChoice import  erroneousChoice
from kernel import jaxrbf
from utility import  paramz
from botorch.utils.multi_objective.pareto import is_non_dominated
import torch
from scipy.stats import multivariate_normal

def compute_accuracy(FCA,FRA):
    acc1=0
    if len(FRA)>0:
        dd= np.vstack([FCA,FRA])
    else:
        dd=np.vstack([FCA])
    res=np.array(is_non_dominated(torch.from_numpy(dd ))+0)
    #print(res)
    if (min(res[0:FCA.shape[0]])==1):
        acc1=1
        if len(FRA)>0:
            if (max(res[FCA.shape[0]:])==0):
                acc1=1
            else:
                acc1=0        
    return acc1

def compute_score(F, CA,RA):
    ACC=[]
    for ii in range(len(CA)):
        #print(ii)
        if len(RA[ii])>0:
            ACC.append(compute_accuracy(F[CA[ii]],F[RA[ii]]))
        else:
            ACC.append(compute_accuracy(F[CA[ii]],[]))
    return np.mean(ACC)


def model_Selection(X,dimA, model_jitter=1e-4, X_test=[], CA_tr=[], RA_tr=[], CA_te=[],RA_te=[], minm=1,maxm=4, niter=3000, epsilon=1e-3):
    Scores= pd.DataFrame(columns=['latentdim', 'accuracy_train', 'loo','accuracy_test'])
    for latentd in range(minm,maxm+1):
        data={'X': X,
          'CA': CA_tr, #[choice_tr[i][0] for i in range(len(choice_tr))],# CA_tr,
          'RA': RA_tr, #[choice_tr[i][1] for i in range(len(choice_tr))],#RA_tr AAAAAAAAAAAAAAAAAAA
          'dimA':dimA
              }

        params = {}
        for i in range(latentd):
            params['lengthscale_'+str(i)]={'value':0.5*np.ones(data["X"].shape[1],float), 
                                        'range':np.vstack([[0.1, 30000.0]]*data["X"].shape[1]),
                                        'transform': paramz.logexp()}
            params['variance_'+str(i)]   ={'value':np.array([5.0]), 
                                            'range':np.vstack([[1.0, 500.0]]),
                                            'transform': paramz.logexp()}
    
    
        # define kernel and hyperparams
        Kernel = jaxrbf.RBF
    
        # define preference model 
        model = erroneousChoice(data,Kernel,params,
                                latentd,ARD=True,
                                jitter=model_jitter)
        model.optimize_hyperparams(niterations=niter, kernel_hypers_fixed=False)
        predictions = model.predict_VI(X)
        samples1 = multivariate_normal(predictions[0][:,0],
                                       predictions[1]+epsilon*np.eye(predictions[0].shape[0])).rvs(4000)
        Likesamples=[]
        for s in range(samples1.shape[0]):
            F= samples1[s,:].reshape(latentd,X.shape[0]).T
            Res1=[]
            row=0
            for ii in model.GroupCA:
                if len(ii)==1:
                    if np.isnan(ii)[0]==True:
                        rr=0
                    else:
                        rr= model._loglike_CA(F,np.atleast_2d(model.CAr[row]) )
                        row=row+1
                else:
                    rr=[]
                    for jj in ii:
                        rr.append(model._loglike_CA(F,np.atleast_2d(model.CAr[row]) ))
                        row=row+1            
            
                Res1.append(np.sum(rr))
            Res2=[]
            row=0
            for ii in model.GroupRA:
                if len(ii)==0:
                    rr=0
                else:
                    rr=[]
                    for jj in ii:
                        rr.append(model._loglike_RA(F,np.atleast_2d(model.RAr[row]) ))
                        row=row+1
                Res2.append(np.sum(rr))  
            Likesamples.append(np.hstack(Res1)+np.hstack(Res2))
        
        print(np.vstack(Likesamples).shape)
       # log_like = np.hstack([ model._log_likelihood(samples1[i,:]) for i in range(samples1.shape[0])])
        loo=psiloo.psisloo(np.vstack(Likesamples))[0]
        print("score=", loo)
        
        acc_tr=None
        if len(CA_tr)>0:
            f = model.predict_VI(X)[0]
            F = f.reshape(latentd,X.shape[0]).T
            acc_tr=compute_score(F, CA_tr,RA_tr)
            print("accuracy train=",acc_tr)
        
        acc_te=None
        if len(X_test)>0:
            f = model.predict_VI(X_test)[0]
            F = f.reshape(latentd,X_test.shape[0]).T
            acc_te=compute_score(F, CA_te,RA_te)
            
        Scores = Scores.append({'latentdim': latentd, 'accuracy_train': acc_tr,  'loo': loo, 'accuracy_test':acc_te}, ignore_index=True)
        print(Scores)
    return Scores

