import numpy as np
from simplex import euclidean_proj_l1ball as proj1
import glmnet_python
from glmnet import glmnet
from glmnet import glmnetSet
from sklearn import linear_model
import scipy
from glmnetCoef import glmnetCoef
from cvglmnet import cvglmnet; from cvglmnetCoef import cvglmnetCoef
from numba import vectorize
from  mainfunrev import mainfun

nsim = np.int(100)
mseres = np.zeros((3, 5, 2))
msenaive = np.zeros((3, 5, 2))
mseregcal = np.zeros((3, 5, 2))
parray = np.array([64, 128, 256])
narray = np.array([100, 200, 300, 400, 500])
carray = np.array([1.5, 1, 0.5])
mk = np.zeros((3, 5, 2))
nk = np.array([6, 10])
p = 64
mpr64 = np.zeros((5, 2,  nsim, p))
naive64 = np.zeros((5, 2, nsim, p))
regcal64 = np.zeros((5,  2, nsim, p))
mbeta64  = np.zeros((5,  2, p))
reptime = 100
for pindx in range(1):
    for nindx in range(5):
        for cindx in range(2):
            p = parray[pindx]
            n = narray[nindx]
            c = nk[cindx]
            mpr64[nindx, cindx, :, :], naive64[nindx,  cindx, :, :], regcal64[nindx, cindx, :, :], mbeta64[nindx,  cindx, :]= mainfun(p, n, c, nsim,10,8e4,  reptime)

np.median(np.sqrt(np.sum(np.abs(mpr64[nindx, cindx, :, :] - mbeta64[nindx, cindx, :])**2, 1)), 0)/np.sqrt(k)
temp = np.load('revgamma.npz')
mpr64 = temp.f.mpr64
mpr128 = temp.f.mpr128
mpr256 = temp.f.mpr256
naive64 = temp.f.naive64
naive128 = temp.f.naive128
naive128 = temp.f.naive128
mbeta64 = temp.f.mbeta64
mbeta128 = temp.f.mbeta128
mbeta256 = temp.f.mbeta256
for nindx in range(5):
    for cindx in range(2):
       mseres[0, nindx, cindx]=  np.median(np.sqrt(np.sum(np.abs(mpr64[nindx, cindx, :, :] - mbeta64[nindx, cindx, :])**2, 1)), 0)
       msenaive[0, nindx, cindx]=  np.median(np.sqrt(np.sum(np.abs(naive64[nindx, cindx, :, :] - mbeta64[nindx, cindx, :])**2, 1)), 0)


np.savez('revtdis', parray = parray, narray = narray, nk = nk, mpr64 = mpr64, mpr128 = mpr128, mpr256 = mpr256, naive64 = naive64, naive128 = naive128, naive256 = naive256, mbeta64 = mbeta64, mbeta128 = mbeta128, mbeta256 = mbeta256)



mseresm = np.zeros((3, 5, 2))
msenaivem = np.zeros((3, 5, 2))
mseregcalm = np.zeros((3, 5, 2))
for nindx in range(5):
    for cindx in range(2):
       mseresm[1, nindx, cindx]=  np.median(np.sqrt(np.sum(np.abs(mpr128m[nindx, cindx, :, :] - mbeta128m[nindx, cindx, :])**2, 1)), 0)/np.sqrt(nk[cindx])
       msenaivem[1, nindx, cindx]=  np.median(np.sqrt(np.sum(np.abs(naive128m[nindx, cindx, :, :] - mbeta128m[nindx, cindx, :])**2, 1)), 0)/np.sqrt(nk[cindx])
        


c = 2, k = 3, n = 200, p = 64, L2 = 0.2600
c = 1.5, k = 4, n = 200, p = 64, L2 = 0.3186
c = 1, k = 6, n = 200, p = 64, L2 = 0.3863
c = 0.5, k = 13, n = 200, p = 64, L2 = 0.6509


c = 2, k= 3, n = 200, p = 128, L2 = 0.3556
c = 1.5, k = 4, n = 200, p = 128, L2 = 0.3669
c = 1, k = 6, n = 200, p = 128, L2 = 0.4195
c = 0.5, k = 12, n = 200, p = 128, L2 = 0.6649


c = 2, k= 3, n = 200, p = 256, L2 = 0.3126
c = 1.5, k = 4, n = 200, p = 256, L2 = 0.3838
c = 1, k = 6, n = 200, p = 256, L2 = 0.4592
c = 0.5, k = 12, n = 200, p = 256, L2 = 0.6794



c = 2, k = 2, n = 100, p = 64, L2 = 0.3674
c= 1.5, k= 3, n = 100, p = 64, L2 = 0.4556
c= 1, k= 5, n = 100, p = 64, L2 = 0.5307
c = 0.5, k = 10, n = 100, p = 64, L2 = 0.8801


c= 2, k = 2, n = 100, p = 128, L2 = 0.3815
c = 1.5, k = 3, n = 100, p = 128, L2 = 0.4269
c= 1, k = 4, n = 100, p = 128, L2 = 0.5893
c = 0.5, k = 9, n = 100, p = 128, L2 = 0.8311

c= 2, k = 2, n = 100, p = 256, L2 = 0.4042
c = 1.5, k = 3, n = 100, p = 256, L2 = 0.4569
c= 1, k = 4, n= 100, p = 256, L2 = 0.6485
c = 0.5, k = 8, n = 100, p = 256, L2 = 0.7775


c = 2, k = 5, n  = 500, p = 64, L2 = 0.2205
c = 1.5, k = 7, n  = 500, p = 64, L2 = 0.2829
c = 1, k = 11, n  = 500, p = 64, L2 = 0.3261


c = 2, k = 5, n  = 500, p = 128, L2 = 0.2324
c = 1.5, k = 7, n  = 500, p = 128, L2 = 0.2633
c = 1, k = 10, n  = 500, p = 128, L2 = 0.3624

c = 2, k = 5, n  = 512, p = 256, L2 = 0.2635
c= 1.5, k = 7, n = 512, p = 256, L2 = 0.2912
c= 1, k = 9, n = 512, p = 256, L2 = 0.3539
