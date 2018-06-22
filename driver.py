import numpy as np
import matplotlib.pyplot as plt
import partial_adj1 as pa
import pandas as pd
import importlib
import sys  
importlib.reload(pa)

#Read in nonlinear model and parameter values
params = {'beta' : 0.99,'delta':0.025,'alpha': 0.3,'a': 4.6,'b' : 0.05,'gamma_A' : 0.00,'gamma_V': 0.00,'rhomu': 0.7,'stdmu' : 0.01,\
             'rhobeta' : 0.5, 'stdbeta': 0.003}
phiitilde = params['b']*params['a']**2
#params['a'] = -20.0
#params['b'] = phiitilde/(params['a']**2)
paramplus = pa.get_steady(params)

#plot linex adjustment costs
inv = np.arange(0.8,1.2,0.0025)
adj = inv*paramplus['b']*(np.exp(paramplus['a']*(inv-1.0))-paramplus['a']*(inv-1.0)-1)
fig1, axs1 = plt.subplots(1,1)
axs1.plot(inv,adj,'k-',linewidth=3)
axs1.set_title('Linex Adjustment Cost Function')
axs1.set_xlabel('$x_t$')
axs1.set_ylabel('$S(x_t)x_t$')
plt.show()
#sys.exit('stop after plot')

#Solve linear model
import dsge
partial=dsge.DSGE.DSGE.read('partial_adj1_linear.yaml')
partiallin = partial.compile_model()
p0 = np.zeros(len(params))
for i,x in enumerate(params):
    p0[i] = params[x]

#get linear decision rule
innov = ['eps_beta','eps_mu']
#innov = ['eps_beta']
ninnov = len(innov)
if ninnov == 1:
    msv = ['kp','inv','betashk']
else:
    msv = ['kp','inv','betashk','mushk']  
nmsv = len(msv)
pdv = ['dinv','qq']
npdv = len(pdv)
(tt,rr,cc) = partiallin.solve_LRE(p0)
Aa = np.zeros([npdv,nmsv])
Bb = np.zeros([npdv,ninnov])
for i in np.arange(npdv):
    for j in np.arange(nmsv):
        Aa[i,j] = tt[partiallin.state_names.index(pdv[i]),partiallin.state_names.index(msv[j])]
    for k in np.arange(ninnov):   
        Bb[i,k] = rr[partiallin.state_names.index(pdv[i]),partiallin.shock_names.index(innov[k])]
lcoeff = Aa.copy()
lcoeff[:,-1] = Aa[:,-1]/paramplus['rhobeta']

#solve nonlinear model
if nmsv == 3:
    ncheb = 3*np.ones(nmsv-1,dtype=int)
    msvmax = np.zeros(nmsv-1)
else:
    ncheb = 3*np.ones(nmsv-2,dtype=int)
    msvmax = np.zeros(nmsv-2)
ngrid = np.ones(ninnov,dtype=int)
msvmax[0] = 0.1
msvmax[1] = 0.2
maxstd = 2.5
ngrid[0] = 7
if ninnov > 1:
    ngrid[1] = 3
eqswitch = 0
poly0 = pa.initialize_poly(npdv,ncheb,ngrid,msvmax,maxstd,eqswitch)
poly0 = pa.get_griddetails(poly0,paramplus)
acoeff0 = pa.get_initcoeffs(lcoeff,poly0)
eqswitch = 1
poly1 = pa.initialize_poly(npdv,ncheb,ngrid,msvmax,maxstd,eqswitch)
poly1 = pa.get_griddetails(poly1,paramplus)
acoeff1,convergence = pa.get_coeffs(acoeff0,paramplus,poly1,step=0.5)
if (convergence == False):
    sys.exit('Failed to solve nonlinear model')

#Get IRFs
irfshock = 1
if irfshock == 1:
    TT = 20
    varlist = ['kp','inv','qq','mu','beta','dinv','invrate']
    endogvarm1 = {x: 0.0 for x in varlist} 
    for x in varlist:
        endogvarm1[x] = paramplus[x]
        endogvarm1[x+'_d'] = 0.0
    #endogvarm1['beta_d'] = 0.012
    #endogvarm1['mu_d'] = 0.012
    endogvarm1['inv_d'] = 0.0
    endogvarm1_b = endogvarm1.copy()
    innov = np.zeros([poly1['ne']])
    #innov[0] = 2.0
    innov[1] = 2.0
    df1 = pa.simulate(TT,endogvarm1,endogvarm1_b,innov,paramplus,acoeff0,poly0,varlist,irfshock)
    df2 = pa.simulate(TT,endogvarm1,endogvarm1_b,innov,paramplus,acoeff1,poly1,varlist,irfshock)
    endogvarm1['beta_d'] = -endogvarm1['beta_d']
    endogvarm1_b['beta_d'] = endogvarm1['beta_d']
    endogvarm1['inv_d'] = -endogvarm1['inv_d']
    endogvarm1_b['inv_d'] = endogvarm1['inv_d']
    #absolute value of negative shock
    innov[0] = -innov[0]
    innov[1] = -innov[1]
    df3 = pa.simulate(TT,endogvarm1,endogvarm1_b,innov,paramplus,acoeff1,poly1,varlist,irfshock)
    df3 = -df3

    #Plot IRFs of linear and nonlinear models
    fig2, axs2 = plt.subplots(2, 2, tight_layout=True)
    varplot = ['inv','dinv','qq','invrate']
    for i in np.arange(len(varplot)):
        if i <= 1:
            df1[varplot[i]].plot(ax=axs2[i,0],style='r:',linewidth=3)
            df2[varplot[i]].plot(ax=axs2[i,0],style='b-',linewidth=3)
            axs2[i,0].set_title(varplot[i])
        else:
            df1[varplot[i]].plot(ax=axs2[i-2,1],style='r:',linewidth=3,label='Linear')
            df2[varplot[i]].plot(ax=axs2[i-2,1],style='b-',linewidth=3,label='Nonlinear')
            axs2[i-2,1].set_title(varplot[i])    
    axs2[1,1].legend()
    plt.show()

    #Plot IRFs of positive and negative shock (flips negative shock)
    fig3, axs3 = plt.subplots(2, 2, tight_layout=True)
    for i in np.arange(len(varplot)):
        if i <= 1:
            df3[varplot[i]].plot(ax=axs3[i,0],style='r:',linewidth=3)
            df2[varplot[i]].plot(ax=axs3[i,0],style='b-',linewidth=3)
            axs2[i,0].set_title(varplot[i])
        else:
            df3[varplot[i]].plot(ax=axs3[i-2,1],style='r:',linewidth=3,label='Negative')
            df2[varplot[i]].plot(ax=axs3[i-2,1],style='b-',linewidth=3,label='Positive')
            axs2[i-2,1].set_title(varplot[i])   
    axs3[1,1].legend()
    plt.show()
else:
    TT = 100
    varlist = ['kp','inv','qq','mu','beta','dinv','invrate']
    endogvarm1 = {x: 0.0 for x in varlist} 
    for x in varlist:
        endogvarm1[x] = paramplus[x]
        endogvarm1[x+'_d'] = 0.0
    endogvarm1_b = endogvarm1.copy()
    innov = np.zeros([poly1['ne']])
    df1 = pa.simulate(TT,endogvarm1,endogvarm1_b,innov,paramplus,acoeff0,poly0,varlist,irfshock)
    df2 = pa.simulate(TT,endogvarm1,endogvarm1_b,innov,paramplus,acoeff1,poly1,varlist,irfshock)
    
















