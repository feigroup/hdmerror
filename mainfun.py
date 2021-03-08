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
from numba import vectorize, float64, int32, guvectorize
def mainfun(p, n, c, nsim, reptime):
    def lassotrans(beta, w, y, omega, lam, eta, offset):
        n = y.shape[0]
        expterm = np.exp(np.matmul(w, beta)-offset - np.matmul(np.matmul(np.transpose(beta), omega), beta)/2)
        omegabeta =   np.matmul(omega, beta)
        omegabeta = omegabeta[np.newaxis, :]
        dLbeta = -(np.matmul(np.transpose(y),  w) - np.matmul(np.transpose(expterm), (w - omegabeta)))/n
        Y =  p *  np.sqrt(eta/2) * (beta - dLbeta/(eta))
        X =  p *  np.eye(p) * np.sqrt(eta/2)
        fit = glmnet(x = scipy.float64(X), y = scipy.float64(Y),  lambdau = scipy.float64([lam]), intr = False)
        beta = np.array(glmnetCoef(fit, s= scipy.float64([0])))[1:, 0]#clf.coef_
        return beta
    
    def mcmcinteg(w0, x, outx, sigmainv):
        u =  (w0[np.newaxis, :] - x)
        kernel = np.diag(np.matmul(np.matmul(u,  sigmainv), np.transpose(u)))
        upper= np.matmul(np.diag(np.exp(-1/2 *  kernel)), outx)
        dom = np.exp(-1/2 * kernel )
        expect = np.sum(upper, 0)/np.sum(dom)
        return expect
    def mcmcinteg1d(w0, x, outx):
        u =  (w0 - x)
        kernel = (u * u)
        upper= ((np.exp(-1/2 *  kernel)) * outx)
        dom = np.exp(-1/2 *kernel)
        expect = np.sum(upper)/np.sum(dom)
        return expect

    k = int(np.round(1/c * (n/np.log(p)) ** (1/2 - 0.01)))
    beta0 = np.zeros(p)
    ix = np.random.choice(p, k, replace=False)
    beta0[ix] = np.linspace(1, 2, k)
    b0 = np.sqrt(np.sum(beta0**2))
    eps = 1e-4
    maxiter = 1000
    omega = np.zeros((p, p))
    rho = 0.1
    for j in range(0, p):
        for i in range(0, p):
            omega[i, j] = 0.5**(np.abs(i - j))
    omega = omega * 0.04
    invomega = np.linalg.inv(omega)
    sqrtomega = scipy.linalg.sqrtm(omega)
    sqrtinvomega = scipy.linalg.sqrtm(invomega)
    u = np.random.multivariate_normal(np.zeros(p), omega, 10000)
    x = np.random.uniform(1, 2,  10000 * p)
    x.resize((10000, p))
    posmean = np.matmul(x, beta0)
    offset = np.log(np.std(np.exp(posmean))/40)
    y = np.random.poisson(np.exp(posmean - offset))
    naivebeta = np.zeros((nsim, p))
    regcalbeta = np.zeros((nsim, p))
    resbeta = np.zeros((nsim, p))
    offset = offset
    w0 = np.zeros((n, p))
    for outitr in range(nsim):            
        u = np.random.multivariate_normal(np.zeros(p), omega, n)
        x = np.random.uniform(1, 2,  n * p)
        x.resize((n, p))
        w = x + u
        posmean = np.matmul(x, beta0)
        y = np.random.poisson(np.exp(posmean - offset))
        fit = cvglmnet(x = scipy.float64(w), y = scipy.float64(y),  nfolds = int(10), family = 'poisson')
        betaini  = np.array(cvglmnetCoef(fit, s = 'lambda_min'))[1:, 0]
        naivebeta[outitr, :] = betaini
        nitr = reptime
        mbeta = np.zeros(nitr)
        mmbeta = np.zeros((nitr, p))
        for j in range(0, nitr):
            a, eta, v = np.linalg.svd(np.matmul(np.transpose(np.matmul(np.diag(np.exp(np.matmul(w, beta0) - offset)), w)), w)/n)
            lam = np.min(eta[:np.min([n, p])]) * (np.log(p)/n)**(1/4) * 50
            eta =  np.max(eta)
            beta  = lassotrans(betaini,  w, y, omega, lam, eta, offset)[:, 0]
            beta = proj1(beta, b0 * np.sqrt(k))
            square2norm = np.sqrt(np.sum(beta**2) )
            if square2norm  > b0:
                beta = beta * b0 /square2norm
            betaini = beta
            mbeta[j] = np.sum((beta- beta0)**2)
            mmbeta[j, :] = beta
            betaini = beta
        resbeta[outitr, :] = beta
        w0 = np.zeros((n, p))
        w = np.matmul(w, sqrtinvomega)
        w0 = np.zeros((n, p))
        for j in range(n):
            expx = np.random.uniform(1, 2,  1000 * p)
            expx = expx.reshape((int(1e3), p))
            expxout = np.matmul(expx, sqrtinvomega)
            for i in range(p):
                w0[j, i] = mcmcinteg1d(w[j, i], expxout[:, i], expxout[:, i])
        w0 = np.matmul(w0, sqrtomega)
        fit = cvglmnet(x = scipy.float64(w0), y = scipy.float64(y),  nfolds = int(10), family = 'poisson')
        regcalbeta[outitr, :]  = np.array(cvglmnetCoef(fit, s = 'lambda_min'))[1:, 0]
        #print(regcalbeta[outitr, ix])
        print(outitr)
    mseres = np.mean(np.sqrt(np.sum(np.abs(resbeta - beta0[np.newaxis, :])**2, 1)), 0)/np.sqrt(k)
    msenaive = np.mean(np.sqrt(np.sum(np.abs(naivebeta - beta0[np.newaxis, :])**2, 1)), 0)/np.sqrt(k)
    mseregcal = np.mean(np.sqrt(np.sum(np.abs(regcalbeta - beta0[np.newaxis, :])**2, 1)), 0)/np.sqrt(k)
    res = np.array([mseres, msenaive, mseregcal, k])
    return res
