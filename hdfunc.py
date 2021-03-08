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



def admmlasso(beta, z, u, w, y, omega, lam, eta, rho):
    n = y.shape[0]
    p = w.shape[1]
    expterm = np.exp(np.matmul(w, beta) - np.matmul(np.matmul(np.transpose(beta), omega), beta)/2)
    omegabeta =   np.matmul(omega, beta)
    omegabeta = omegabeta[np.newaxis, :]
    beta = ((eta + rho)**(-1)) *((np.matmul(np.transpose(y),  w) - np.matmul(np.transpose(expterm), (w - omegabeta)))/n + beta * eta  + rho * (z) - u)
    x = beta+u/rho
    z= (x >= lam/rho) *(x - lam/rho) + (x < -lam/rho)*( x +  lam/rho)
#    print(abs(beta + u/rho))
    u = u  + rho * (beta - z)
    return beta, z, u

def lassonoerror(beta, w, y, omega, lam, eta, offset):
    n = y.shape[0]
    p = w.shape[1]
    expterm = np.exp(np.matmul(w, beta)-offset)
    
    dLbeta = -(np.matmul(np.transpose(y),  w) - np.matmul(np.transpose(expterm), (w)))/n
    Y =  p *  np.sqrt(eta/2) * (beta - dLbeta/(eta))
    X =  p *  np.eye(p) * np.sqrt(eta/2)
    fit = glmnet(x = scipy.float64(X), y = scipy.float64(Y),   lambdau = scipy.float64([lam]), intr = False)
    beta = np.array(glmnetCoef(fit, s= scipy.float64([0])))[1:, 0]#clf.coef_
    return beta

def lassotrans(beta, w, y, omega, lam, eta, offset):
    n = y.shape[0]
    p = w.shape[1]
    expterm = np.exp(np.matmul(w, beta) -offset- np.matmul(np.matmul(np.transpose(beta), omega), beta)/2)
    omegabeta =   np.matmul(omega, beta)
    omegabeta = omegabeta[np.newaxis, :]
    dLbeta = -(np.matmul(np.transpose(y),  w) - np.matmul(np.transpose(expterm), (w - omegabeta)))/n
    Y =  p *  np.sqrt(eta/2) * (beta - dLbeta/(eta))
    X =  p *  np.eye(p) * np.sqrt(eta/2)
    #print(X.shape)
    #print(Y.shape)
    fit = glmnet(x = scipy.float64(X), y = scipy.float64(Y),  lambdau = scipy.float64([lam]), intr = False)
    beta = np.array(glmnetCoef(fit, s= scipy.float64([0])))[1:, 0]#clf.coef_
    return beta

def mcmcinteg(w0, x, outx, sigmainv):
    u =  (w0[np.newaxis, :] - np.matmul(x, sqrtinvomega))
    kernel = np.diag(np.matmul(np.matmul(u,  sigmainv), np.transpose(u)))
    upper= np.matmul(np.diag(np.exp(-1/2 *  kernel)), outx)
    dom = np.exp(-1/2 * kernel )
    expect = np.sum(upper, 0)/np.sum(dom + 1e-60)
    return expect
def mcmcinteg1d(w0, x, outx):
    u =  (w0 - x)
    kernel = (u * u)
    upper= ((np.exp(-1/2 *  kernel)) * outx)
    dom = np.exp(-1/2 *kernel)
    expect = np.sum(upper)/np.sum(dom + 1e-6)
    return expect

mk = np.array([18, 16, 14, 12, 10, 8, 6, 4])
mc = np.zeros(len(mk))


def main(w, y,  scale, offset, b0, eta, nitr, lambdac, k, momega, lamseq):
    upper = np.max(w)
    lower = np.min(w)
    sdw = np.std(w)
    n = len(y)
    p = w.shape[1]
    nsim = 1
    omega = np.zeros((p, p))
    omega = np.cov(np.transpose(w))
    omega[:2, :2] = 0
    omega = omega * scale
    omega = momega
    # if scale >0:
    #     invomega = np.linalg.inv(omega)
    #     sqrtomega = scipy.linalg.sqrtm(omega)
    #     sqrtinvomega = scipy.linalg.sqrtm(invomega)
    naivebeta = np.zeros((nsim, p))
    regcalbeta = np.zeros((nsim, p))
    resbeta = np.zeros((nsim, p))
    offset = scipy.float64(offset)
    outitr = 0
    fit = cvglmnet(x = scipy.float64(w), y = scipy.float64(y), lambdau = lamseq,    nfolds = int(10), family = 'poisson', intr = False)
    print(fit['lambda_min'])
    #scipy.float64([0])
    betaini1  = np.array(cvglmnetCoef(fit, s = 'lambda_min'))[1:, 0]
    naivebeta[outitr, :] = betaini1
    betaini1  = np.array(cvglmnetCoef(fit, s = scipy.float64([0])))[1:, 0]
    if False:
        w1 = np.matmul(w, sqrtinvomega)
        w0 = np.zeros((n, p))
        for j in range(n):
            expx = np.random.uniform(lower + 2 * sdw, upper - 2 *sdw ,  1000 * p)
            expx = expx.reshape((int(1e3), p))
            expxout = np.matmul(expx, sqrtinvomega)
            for i in range(p):
                w0[j, i] = mcmcinteg1d(w1[j, i], expxout[:, i], expxout[:, i])
        w0 = np.matmul(w0, sqrtomega)
        fit = cvglmnet(x = scipy.float64(w0), y = scipy.float64(y),  nfolds = int(10), family = 'poisson', intr = False)
        tempcoef = np.array(cvglmnetCoef(fit,  s = scipy.float64([0.5])))[1:, 0]
    else:
        tempcoef = betaini1
    regcalbeta[outitr, :]  = tempcoef
    betaini  = tempcoef
    if(np.sum(abs(tempcoef)) == 0):
            betaini = betaini1
    k = int(np.sum(abs(betaini) >0))
    #temp  = proj1(betaini1, b0 * np.sqrt(k) * (n/np.log(p))**(1/4 + 0.01)
    #print(b0 * np.sqrt(k) * (n/np.log(p))**(1/4 + 0.01))
    betaini = proj1(betaini, b0 * np.sqrt(k) * (n/np.log(p))**(1/4 + 0.01))
    square2norm = np.sqrt(np.sum(betaini**2) )
    #betanoerror = temp
    #square2temp = np.sqrt(np.sum(temp**2) )
    # if square2temp > b0  * (n/np.log(p))**(1/4 + 0.01):
    #         betanoerror = betanoerror * b0 *(n/np.log(p))**(1/4 + 0.01)/square2temp
    #betaini = betaini * b0 /square2norm
    mbeta = np.zeros(nitr)
    mmbeta = np.zeros((nitr, p))
    for j in range(0, nitr):
        #print(j)
        lam =  (np.log(p)/n)**(1/4) * lambdac
        eta = eta
        beta  = lassotrans(betaini,  w, y, omega, lam, eta, offset)[:, 0]
        beta = proj1(beta, b0 * np.sqrt(k) * (n/np.log(p))**(1/4 + 0.01))
        if(np.sum(beta >0) == 0):
            beta = betaini
            print('penalty large')
        square2norm = np.sqrt(np.sum(beta**2) )
        if square2norm  > b0  * (n/np.log(p))**(1/4 + 0.01):
            beta = beta * b0 *(n/np.log(p))**(1/4 + 0.01)/square2norm
        mbeta[j] = sum((beta-betaini)**2)/np.sqrt(np.sum(abs(beta)>0))
        betaini = beta
        mmbeta[j, :] = beta
        if(mbeta[j] == 0):
            break
    # print(np.where(mbeta >1e-12)[0][-2])
    # beta = mmbeta[np.where(mbeta >1e-12)[0][-2], :]
    #print(mbeta)
    resbeta[outitr, :] = beta
    #print(outitr)
    outitr = outitr + 1
    mpr= resbeta[0:outitr, :]
    naive = naivebeta[0:outitr, :]
    regcal = regcalbeta[0:outitr, :]
    return mpr, naive, regcal

