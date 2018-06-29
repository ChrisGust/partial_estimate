import numpy as np
import pandas as p
from numba import jit
 
from statsmodels.base.model import GenericLikelihoodModel
from statsmodels.tsa.tsatools import lagmat
from statsmodels.api import add_constant
from scipy.stats import norm
 
from arch.univariate import ARX
 
@jit(nopython=True)
def qar_fast_likelihood(endog, exog, phis, quad_phi, gammas, sigma, slags, h):
 
    T,p1 = exog.shape
    p = p1-1
    q = gammas.shape[0]
 
    w = (1 - h/q)
 
    like = np.zeros(T)
 
    for i in range(T):
 
        # svec = [slags[plag1] * slags[plag2]
        #         for plag1 in range(self.p)
        #         for plag2 in range(plag1,self.p)]
 
        yhat = np.dot(exog[i], phis) + np.dot(slags[:p]**2, quad_phi)
 
        #sigmaT = sigma * (h +  w * np.exp(np.dot(slags[:q],gammas)/w).sum())
        #if q > 1:
        #    sigmaT = sigma * (h +  w * np.exp(np.dot(slags[:q],gammas)/w).sum())
        #else:
        #sigmaT = sigma * (h +  w * np.exp(np.dot(slags[:q],gammas)/w))
        sigmaT = (1 + np.dot(slags[:q], gammas) ) * sigma
        if sigmaT < 0:
            like[:] = -10000000.0
            return like
 
        u = endog[i]-yhat
 
        like[i] = np.log(np.sqrt(2.0*np.pi)) + np.log(sigmaT) + 0.5*(u/sigmaT)**2     
        #-norm(loc=0,scale=sigmaT).logpdf(u)
        u_scaled = u/sigmaT
 
        st = np.dot(slags[:p], phis[1:]) + sigma * u_scaled
        slags[1:] = slags[:-1]
        slags[0] = st
 
        #slags = np.r_[st, slags[:-1]]
 
    # see statsmodels PR 3253
    return like
 
 
class QAR(GenericLikelihoodModel):
    """Statsmodels likelihood class for Pseudo ARCH model.
 
    The model is described by:
    phi(L)y_t = sigma_t epsilon_t, epsilon_t ~ N(0,1)
    sigma_t = f(x_t)
 
    Parameters
    ----------
    endog : arraylike
        Vector of y_t of length T.
    ikbar : arraylike
        Vector of x_t of length T-p.  In BCE, this is the lagged k-period average of y_t.
    p : int
        lag length of phi
    model : str
        'stdev'    : f(x_t) = params[-2]/100 + params[-1]/1000*x_t
        'variance' : f(x_t) = sqrt(params[-2]/100 + params[-1]/1000*x_t)
        'exp'      : f(x_t) = np.exp(params[-2] + params[-1]*x_t)             
    """
    def __init__(self, endog, use_exp=True, h=0.1, p=1, q=1, **kwargs):
        exog, endog = lagmat(endog, p, trim='both',original='sep', use_pandas=True)
 
        super().__init__(endog,add_constant(exog),**kwargs)
 
        self.p = p
        self.q = q
        self.h = h
 
        for plag1 in range(1,p+1):
           self.exog_names.append('phi2%d%d' % (plag1,plag1))
        # for plag1 in range(1,p+1):
        #     for plag2 in range(plag1,p+1):
        #         self.exog_names.append('phi2%d%d' % (plag1,plag2))
 
        for qlag in range(1,q+1):
            self.exog_names.append('gamma%d' % qlag)
 
        self.exog_names.append('sigma')
 
        # for lag in range(1,max(p,q)+1):
        #     self.exog_names.append('s%d' % lag)
 
        self.nlinear_terms = self.p+1
        self.nquad_terms = int(self.p*(self.p+1)/2)
        self.ngamma_terms = self.q
        self.ninit_terms = max(self.p, self.q)
 
    def nloglikeobs(self, params, get_s=False):
        exog = self.exog
        endog = self.endog
 
        linear_phi = params[:self.nlinear_terms]
        quad_phi = params[self.nlinear_terms:self.nlinear_terms+self.nquad_terms]#/10
        gammas = params[self.nlinear_terms+self.nquad_terms:self.nlinear_terms+self.nquad_terms+self.ngamma_terms]
        sigma = params[self.nlinear_terms+self.nquad_terms+self.ngamma_terms]
        #slags = params[-self.ninit_terms:]/100
 
        slags = np.zeros(self.p)
        if not get_s:
            like = qar_fast_likelihood(endog, exog, linear_phi, quad_phi, gammas, sigma, slags, self.h)
            return np.array(like,dtype='float64')
       
        T = self.exog.shape[0]
        w = (1 - self.h/self.q)
 
        like = np.zeros(T)
 
        s = np.zeros(T)
 
        for i in range(T):
 
            # svec = [slags[plag1] * slags[plag2]
            #         for plag1 in range(self.p)
            #         for plag2 in range(plag1,self.p)]
 
            yhat = np.dot(exog[i], linear_phi) #+ np.dot(slags[:self.p]**2, quad_phi)
            sigmaT = sigma * (self.h +  w * np.exp(np.dot(slags[:self.q],gammas)/w).sum())
 
            if sigmaT < 0:
                like[:] = -10000000.0
                return like
 
            u = endog[i]-yhat
 
            like[i] = -norm(loc=0,scale=sigmaT).logpdf(u)
            u_scaled = u/sigmaT
 
            st = np.dot(slags[:self.p], linear_phi[1:]) + sigma * u_scaled
            slags[1:] = slags[:-1]
            slags[0] = st
            s[i] = st
 
        # see statsmodels PR 3253
        return np.array(like,dtype='float64'), s, endog
 
 
def estimate_qar(y, p=1, q=1, disp=1):
    """
    Estimates a QAR(p, q) on data y.
 
    disp
 
    Returns statsmodels.fitted object.
    """
    lags = p
    qarpq = QAR(y, p=lags,q=1)
 
   
 
    am = ARX(y, lags=lags, constant=True)
    first_stage = am.fit()
   
    params = np.r_[first_stage.params[:-1],
                   100*np.zeros(lags),
                   100*np.zeros(qarpq.q),
                   1*np.sqrt(np.abs(first_stage.params[-1]))]
                   #]
 
    results = qarpq.fit(maxiter=50000,start_params=params, disp=disp)
 
    return results
