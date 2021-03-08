import numpy as np
from simplex import euclidean_proj_l1ball as proj1
import glmnet_python
from glmnet import glmnet
from glmnet import glmnetSet
from sklearn import linear_model
import scipy
from glmnetCoef import glmnetCoef
from cvglmnet import cvglmnet; from cvglmnetCoef import cvglmnetCoef
import random
import csv
import pickle
import nltk
from hdfunc import *


coviddata= np.load('covidata15.npz')
omega = np.load('omega15.npy')

X = coviddata['X']
Y = coviddata['Y']/coviddata['dnom']
temp = X
#X[:, np.arange(63, 94)] =(-temp[:, np.arange(63, 94)] + temp[:, np.arange(32, 63)])
#X[:, np.arange(32, 63)] = ((temp[:, np.arange(63, 94)] + temp[:, np.arange(32, 63)])/2)


#X = (X -np.mean(X, 0)[np.newaxis, :])/np.std(X, 0)[np.newaxis, :]#np.sqrt(np.Sum((X +100)**2, 1))[:, np.newaxis]
Y = Y

p = X.shape[1]
nsim = 100
errorv = ((np.arange(0, 1, 0.1)))
mres = np.zeros((nsim, 2,  p))
gtnaive = []
lenaive =[]
gtregcal =[]
leregcal = []
gtmpr = []
lempr = []
gtcoefvalue = []
lecoefvalue = []
#np.log(allY[:500, 0])
#2.5 * np.mean(np.log(allY[:500, 0])
offset =np.zeros((len(Y), 1))# - np.log(allY[:500, 0]) +17.65773399
ix = np.arange(len(Y))
mgrerror = np.zeros(nsim)
naiveerror = np.zeros(nsim)
ix = np.arange(len(Y))
m0omega = omega.copy()
m0omega[:] = 0
lamseq = np.arange(1e-2, 0, -1e-5)
tempY = Y
Y = tempY 
boost = False
offset = np.log(1000)
lamres = np.zeros((len(lamseq), 4))

mscale = np.arange(0.5, 4, 0.5)
scaleres = np.zeros((len(mscale), 4))
lammpr = np.arange(0, 1, 1e-1)
lamres = np.zeros((len(lammpr), 4))
for l in range(len(lammpr)):
    for j in range(nsim):
        if boost == False:
            testix =  np.random.choice(ix, 40, replace=False)
            trainix =np.delete(ix, testix)#
        else:
            trainix = np.random.choice(ix, 119, replace=True)#
            testix =np.delete(ix, trainix)#
        W = X.copy()
        res = main(w = W[trainix, :] , y = Y[trainix] , scale = 0, offset = offset, b0 = 2, eta = 2e+4, nitr = 1000, lambdac = 2e-1, k = 10, momega = omega, lamseq = lamseq)
        mgrerror[j]= np.mean(np.abs(Y[testix] - np.exp(np.matmul(W[testix, ], res[0][0] ) -offset- np.matmul(np.matmul(res[0][0].transpose(), omega), res[0][0])/2)))
        naiveerror[j]= np.mean(np.abs(Y[testix] - np.exp(np.matmul(W[testix, ], res[1][0]))))
        print(np.abs(mgrerror[j]) - np.abs(naiveerror[j]))
        mres[j, :, :] = np.vstack((res[0][0], res[1][0]))
    lamres[l, :] = np.hstack((np.quantile((np.abs(mgrerror) - np.abs(naiveerror))/Y, (0.05, 0.5, 0.95)), np.mean(np.abs(mgrerror)/Y)))

mres1 = np.zeros(mres.shape)
mres1[:, :, 32:63] = mres[:, :, 32:63] + mres[:, :, 63:94]  
mres1[:, :, 63:94] = (mres[:, :, 32:63] - mres[:, :, 63:94]  )/2
mres1[:, :, :32] = mres[:,:,  :32]
qmgr = np.quantile(mres1[:, 0, :],  (0.025,  0.975), axis=0)
ix1 = np.arange(0, 94)
((qmgr[0, :] >0 ) * (qmgr[1, :] >0 ) )[ix1]
((qmgr[0, :] <0 ) * (qmgr[1, :] <0 ) )[ix1]
qnaive = np.quantile(mres1[:, 1, :],  (0.025,  0.975), axis=0)
((qnaive[0, :] >0 ) * (qnaive[1, :] >0 )) [ix1]
((qnaive[0, :] <0 ) * (qnaive[1, :] <0 ) )[ix1]
np.savez('abserror15.npz', mgrerror = mgrerror, naiveerror= naiveerror, mres = mres)
np.savez('boores13.npz', mres = mres1)
    o = np.argsort(res[1][0][np.where(res[1]>0)[1]])
    gtnaive.append(name[np.where(res[1]>0)[1]][o])
    o = np.argsort(np.abs(res[1][0][np.where(res[1]<0)[1]]))
    lenaive.append(name[np.where(res[1]<0)[1]][o])
    o = np.argsort(res[2][0][np.where(res[2]>0)[1]])
    gtregcal.append(name[np.where(res[2]>0)[1]][o])
    o = np.argsort(np.abs(res[2][0][np.where(res[2]<0)[1]]))
    leregcal.append(name[np.where(res[2]<0)[1]][o])
    o = np.argsort(res[0][0][np.where(res[0]>0)[1]])
    gtmpr.append(name[np.where(res[0]>0)[1]][o])
    gtcoefvalue.append(mres[j, 0, np.where(res[0]>0)[1]][o])
    o = np.argsort(np.abs(res[0][0][np.where(res[0]<0)[1]]))
    lempr.append(name[np.where(res[0]<0)[1]][o])
    lecoefvalue.append(mres[j, 0, np.where(res[0]<0)[1]][o])


mexp= np.exp(np.matmul(X, mres[0, 0, ]) - offset)
base = np.log(np.var(Y)/np.mean(mexp))
ratio = np.sum((Y - mexp)**2/np.var(Y))
phi = np.sum((Y - mexp)**2/(ratio * mexp))
np.save('mres.npy', mres)
np.save('res3.npy',  res[3])

onresmpr =mres[:, 0, :]
onresnaive =mres[:, 1, :]
np.save('jcres0.npy',  onresmpr)
np.save('jcres1.npy', onresnaive)


o = np.argsort(res[3][np.where(res[3]>0)])
(name[np.where(res[3]>0)][o])
o = np.argsort(np.abs(res[3][np.where(res[3]<0)]))
(name[np.where(res[3]<0)][o])


o = np.argsort(res[0][0][np.where(res[0]>0)[1]])
(name[np.where(res[0]>0)[1]][o])
o = np.argsort(np.abs(res[0][0][np.where(res[0]<0)[1]]))
(name[np.where(res[0]<0)[1]][o])
