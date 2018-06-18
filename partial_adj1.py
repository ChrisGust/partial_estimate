import numpy as np
import pandas as pd
import sys
import polydef as po

#############################################################
#Initialize Polynomial Used to approximate solution.
##############################################################

def initialize_poly(nfunc,ncheb,ngrid,msvmax,shockmaxstd,eqswitch):
#Returns some details of polynomial approximation (the ones that do not depend on model parameters).

    poly = {'nfunc': nfunc, 'maxstd': shockmaxstd, 'eqswitch' : eqswitch}
    poly['ne'] = len(ngrid)
    poly['nmsv'] = len(ncheb)
    if poly['ne'] > 2:
        print('number of shocks on linear interpolation part of the grid  must be less than 2. Stopping now.')
        sys.exit()
    poly['ncheb'] = ncheb
    poly['npoly'] = np.product(poly['ncheb'])
    poly['ninnov'] = poly['ne']
    poly['ngrid'] = ngrid
    poly['nquad'] = np.ones(poly['ninnov'],dtype=int)
    poly['rhos'] = np.zeros(poly['ne'])
    poly['stds'] = np.zeros(poly['ne'])
    poly['nquad'][0] = 3
    poly['ns'] = np.prod(poly['ngrid'])
    poly['nqs'] = np.prod(poly['nquad'])
    poly['pgrid'],poly['bbt'],poly['bbtinv'] = po.nonsparsegrid(poly['nmsv'],poly['npoly'],poly['ncheb'])
    poly['msvbounds'] = np.zeros(2*poly['nmsv'])
    poly['msvbounds'][0] = -msvmax[0]
    poly['msvbounds'][poly['nmsv']] = msvmax[0]
    poly['msvbounds'][1] = -msvmax[1]
    poly['msvbounds'][poly['nmsv']+1] = msvmax[1]
    return(poly)

def get_quadstates(poly,paramplus):
    #returns exogenous states at each quadrature point and each exogenous grid point

    ninnov = poly['ninnov']
    nqs = poly['nqs']
    ns = poly['ns']
    npoly = poly['npoly']
    ne = ninnov
    quadgrid,quadweight = po.get_quadgrid(poly['nquad'],ninnov,nqs)
    sfut = np.zeros([ns,nqs,ninnov])
    ind_sfut = np.zeros([ns,nqs,ne],dtype=int)

    #meanpreserve = -paramplus['stdbeta']**2/(2*(1.0+paramplus['rhobeta']))
    #print(meanpreserve)
    for j in np.arange(nqs):
        for i in np.arange(ns):
            sfut[i,j,0] = paramplus['rhobeta']*poly['exoggrid'][i] + paramplus['stdbeta']*quadgrid[j,0] #+ meanpreserve
            if ne == 2:
                sfut[i,j,1] = paramplus['rhomu']*poly['exoggrid'][i] + paramplus['stdmu']*quadgrid[j,0]
            ind_sfut[i,j,:] = po.get_index(sfut[i,j,:],ne,poly['ngrid'],poly['steps'],poly['bounds'])    
    return(sfut,ind_sfut,quadweight)

def get_griddetails(poly,paramplus):
    #Return details of polynomial associated with constructing the grids used

    nmsv = poly['nmsv']
    poly['scmsv2xx'] = np.zeros([2*nmsv])
    poly['scxx2msv'] = np.zeros([2*nmsv])
    poly['scmsv2xx'][0:nmsv] = 2.0/(poly['msvbounds'][nmsv:2*nmsv]-poly['msvbounds'][0:nmsv])
    poly['scmsv2xx'][nmsv:2*nmsv] = -2.0*poly['msvbounds'][0:nmsv]/(poly['msvbounds'][nmsv:2*nmsv]-poly['msvbounds'][0:nmsv])-1.0
    poly['scxx2msv'][0:nmsv] = 0.5*(poly['msvbounds'][nmsv:2*nmsv]-poly['msvbounds'][0:nmsv])
    poly['scxx2msv'][nmsv:2*nmsv] = poly['msvbounds'][0:nmsv] + 0.5*(poly['msvbounds'][nmsv:2*nmsv]-poly['msvbounds'][0:nmsv])

    #for the approximating function.
    poly['rhos'][0] = paramplus['rhobeta']
    poly['stds'][0] = paramplus['stdbeta']
    if poly['ne'] == 2:
        poly['rhos'][1] = paramplus['rhomu']
        poly['stds'][1] = paramplus['stdmu']
    
    poly['exoggrid'],poly['exogindex'],poly['steps'],poly['bounds'],poly['ind2poly'] = po.get_exoggrid(poly['ngrid'],poly['ne'], \
        poly['ns'],poly['rhos'],poly['stds'],poly['maxstd'])
    #meanspread = -paramplus['stdbeta']**2/(2*(1.0+paramplus['rhobeta'])*(1.0-paramplus['rhobeta']))
    poly['exoggrid'][:,0] = poly['exoggrid'][:,0] #+ meanspread

    poly['quadgrid'],poly['quadweight'] = po.get_quadgrid(poly['nquad'],poly['ninnov'],poly['nqs'])
    poly['sfutquad'],poly['ind_sfutquad'],poly['quadweights'] = get_quadstates(poly,paramplus)
    return(poly)

def get_initcoeffs(lcoeff,poly):
    #iterate until convergence to find fixed point

    nfunc = poly['nfunc']
    npoly = poly['npoly']
    ns = poly['ns']
    nmsv = poly['nmsv']
            
    acoeff0 = np.zeros([ns,nfunc*npoly])    
    for i in np.arange(ns):
        polyapp = np.zeros([npoly,nfunc])
        shk = poly['exoggrid'][i,:]
        for ip in np.arange(npoly):
            msvm1 = po.msv2xx(poly['pgrid'][ip,:],nmsv,poly['scxx2msv'])
            msvm1plus = np.append(msvm1,shk)
            polyapp[ip,:] = np.dot(lcoeff,msvm1plus)
        alphass = np.dot(poly['bbtinv'],polyapp)
        for ip in np.arange(npoly):
            for ifunc in np.arange(nfunc):
                acoeff0[i,ifunc*npoly+ip] = alphass[ip,ifunc]
    return(acoeff0)

# #############################################################
# #Functions for solving the model.
# #############################################################

def get_steady(params):
    gamma = np.exp(params['gamma_A']+params['gamma_V'])
    temp = params['alpha']*params['beta']/(gamma-params['beta']*(1.0-params['delta']))
    kss = gamma*temp**(1/(1-params['alpha']))
    iss = (1.0-(1.0-params['delta'])/gamma)*kss 
    qss = 1.0
    steady = {'kp': kss, 'inv': iss, 'qq' : qss, 'gamma' : gamma, 'mu' : 1.0, 'invrate' : gamma*iss/kss, 'dinv' : gamma}
    paramplus = params.copy()
    paramplus.update(steady)
    return(paramplus)

def modelvariables(polyapprox,xtilm1,shocks,paramplus,eqswitch,ne):
    #return model variables given expected output and inflation.
    dvar = {}
    lvar = {}
            
    kk = xtilm1[0]
    invm1 = xtilm1[1]
    dvar['inv'] = polyapprox[0]+invm1
    dvar['qq'] = polyapprox[1]
    lvar['inv'] = np.exp(dvar['inv']+np.log(paramplus['inv']))  
    lvar['qq'] = np.exp(dvar['qq']+np.log(paramplus['qq']))
    dvar['beta'] = shocks[0]
    lvar['beta'] = np.exp(dvar['beta']+np.log(paramplus['beta']))
    if (ne == 2):
        dvar['mu'] = shocks[1]
        lvar['mu'] = np.exp(dvar['mu'])
    else:
        dvar['mu'] = 0.0
        lvar['mu'] = 1.0
        
    dvar['dinv'] = dvar['inv']-invm1
    lvar['dinv'] = np.log(paramplus['gamma']) + dvar['dinv']
    dvar['invrate'] = dvar['inv'] - kk
    lvar['invrate'] = np.log(paramplus['gamma']) + np.log(lvar['inv']) - kk - np.log(paramplus['kp'])
        
    if (eqswitch == 0):
        dvar['kp'] = ((1-paramplus['delta'])/paramplus['gamma'])*kk+(paramplus['inv']/paramplus['kp'])*(dvar['inv']+dvar['mu'])
        lvar['kp'] = np.exp(dvar['kp']+np.log(paramplus['kp']))
    else:
        #stemp = 
        lvar['kp'] = (1-paramplus['delta'])*( np.exp(kk+np.log(paramplus['kp']))/paramplus['gamma'] ) + lvar['inv']
        dvar['kp'] = np.log(lvar['kp'])-np.log(paramplus['kp'])
    return(dvar,lvar)

def calc_euler(ind_state,gridindex,acoeff,poly,paramplus):
    npoly = poly['npoly']
    nmsv = poly['nmsv']
    ne = poly['ne']           
    polycur = np.zeros([poly['nfunc']])
    polyvarm1 = poly['bbt'][gridindex,:]
    for ifunc in np.arange(poly['nfunc']):
        polycur[ifunc] = np.dot(acoeff[ind_state,ifunc*npoly:(ifunc+1)*npoly],polyvarm1)

    msvm1 = po.msv2xx(poly['pgrid'][gridindex,:],nmsv,poly['scxx2msv'])
    kk = msvm1[0]
    invm1 = msvm1[1]
    shocks = poly['exoggrid'][ind_state,:]
    (dvar,lvar) = modelvariables(polycur,msvm1,shocks,paramplus,poly['eqswitch'],poly['ne'])
    msv = np.array([dvar['kp'],dvar['inv']])
    
    ind = npoly*ind_state+gridindex
    ind_futmat = poly['ind_sfutquad'][ind_state,:,:] 
    ss_futmat = poly['sfutquad'][ind_state,:,:]

    exp_qeq = 0.
    exp_ieq = 0.
    qweight = (1.0-paramplus['delta'])/paramplus['gamma']
    for j in np.arange(poly['nqs']):
        xx1 = po.msv2xx(msv,nmsv,poly['scmsv2xx']) 
        polyfut = po.get_linspline(xx1,ss_futmat[j,:],ind_futmat[j,:],acoeff,poly['exoggrid'],poly['steps'],poly['nfunc'],\
            poly['ngrid'],npoly,nmsv,poly['ncheb'],ne)
        (dvarp,lvarp) = modelvariables(polyfut,msv,ss_futmat[j,:],paramplus,poly['eqswitch'],poly['ne'])
        if (poly['eqswitch'] == 0):
            exp_qeq = exp_qeq + poly['quadweight'][j]*qweight*dvarp['qq'] 
            exp_ieq = exp_ieq + poly['quadweight'][j]*paramplus['beta']*(dvarp['inv']-dvar['inv'])
        else:       
            #exp_qeq = exp_qeq + poly['quadweight'][j]*qweight*lvarp['qq']
            exp_qeq = exp_qeq + poly['quadweight'][j]*qweight*dvarp['qq'] 
            Sprimep = paramplus['b']*paramplus['a']*(np.exp(paramplus['a']*(lvarp['inv']/lvar['inv']-1))-1)
            exp_ieq = exp_ieq + poly['quadweight'][j]*lvar['beta']*lvarp['qq']*lvarp['mu']*Sprimep*(lvarp['inv']/lvar['inv'])**2
    polynew = np.zeros(poly['nfunc'])
    if poly['eqswitch'] == 0:
        polynew[0] = (1.0/(paramplus['a']**2*paramplus['b']))*(dvar['qq']+dvar['mu'])+exp_ieq
        polynew[1] = -(1-paramplus['beta']*qweight)*(1.0-paramplus['alpha'])*dvar['kp'] + paramplus['beta']*exp_qeq + dvar['beta']
    else:
        linvm1 = np.exp(invm1+np.log(paramplus['inv']))
        dinv_l = lvar['inv']/linvm1
        adjcost = paramplus['b']*(np.exp(paramplus['a']*(dinv_l-1))-paramplus['a']*(dinv_l-1)-1)
        stemp = (lvar['qq']*lvar['mu']-1.0+exp_ieq)/(lvar['qq']*lvar['mu']*np.exp(dvar['inv']-invm1))  
        sarg = 1.0+(1.0/paramplus['a'])*np.log(1.0+stemp/(paramplus['b']*paramplus['a']))
        polynew[0] = np.log(sarg)
        #rkp = (paramplus['alpha']/paramplus['gamma'])*(lvar['kp']/paramplus['gamma'])**(paramplus['alpha']-1)
        #polynew[1] = np.log( lvar['beta']*(rkp+exp_qeq) )
        polynew[1] = -(1-paramplus['beta']*qweight)*(1.0-paramplus['alpha'])*dvar['kp'] + paramplus['beta']*exp_qeq + dvar['beta']
    res = np.sum(np.abs(polynew-polycur))/poly['nfunc']
    # if (gridindex == 1):
    #     print('---------------------')
    #     print('kk = ', kk)
    #     print('invm1 = ', invm1)
    #     print('dvar =', dvar)
    #     print(np.abs(polynew-polycur))
    #     print('polycur = ', polycur)
    #     print('exp_qeq = ', exp_qeq)
    #     print('exp_ieq = ', exp_ieq)
    #     print('-----------------')
    #     print('ind_state = ', ind_state)
    #     print('gridindex = ', gridindex)
    #     print('msvm1 = ', msvm1)
    #     print('shocks = ', shocks)
    #     print('---------------------')
    #     print('polynew = ', polynew)
    #     print('res = ', res)
    #     sys.exit('in calc')
    return(polynew,res)

def get_coeffs(acoeff0,paramplus,poly,step=0.75,niter=1000,stol=1e-04):
    #iterate until convergence to find fixed point

    nfunc = poly['nfunc']
    npoly = poly['npoly']
    ns = poly['ns']
            
    acoeff = acoeff0
    acoeffnew = np.zeros([ns,nfunc*npoly])    
    convergence = False

    for tt in np.arange(niter):
        avgerror = 0.
        for i in np.arange(ns):
            res_avg = 0.
            polyappnew = np.zeros([npoly,nfunc])
            for ip in np.arange(npoly):
                polyappnew[ip,:],res = calc_euler(i,ip,acoeff,poly,paramplus)                
                res_avg = res_avg + res/npoly
            alphass = np.dot(poly['bbtinv'],polyappnew)
            for ip in np.arange(npoly):
                for ifunc in np.arange(nfunc):
                    acoeffnew[i,ifunc*npoly+ip] = alphass[ip,ifunc]
            avgerror = avgerror + res_avg/ns
        #print(avgerror)
        
        if np.any(np.isnan(acoeffnew)):  #return acoeff
            break
        if (avgerror < stol):
            convergence = True
            break
        acoeff = (1.0-step)*acoeff + step*acoeffnew
    return(acoeff,convergence)

# #############################################################
# #Functions for simulating the model.
# #############################################################

def decr(endogvarm1,innov,paramplus,acoeff,poly):
    #decision rule with variables in levels and deviation from steady state.
    endogvar = {}
    kk = endogvarm1['kp_d']
    invm1 = endogvarm1['inv_d']
    msvm1 = np.array([kk,invm1])
    shocks = np.zeros(poly['ne'])
    shocks[0] = paramplus['rhobeta']*endogvarm1['beta_d']+paramplus['stdbeta']*innov[0]
    if poly['ne'] == 2:
        shocks[1] = paramplus['rhomu']*endogvarm1['mu_d']+paramplus['stdmu']*innov[1]
    ind_shocks = po.get_index(shocks,poly['ne'],poly['ngrid'],poly['steps'],poly['bounds'])    
    xx1 = po.msv2xx(msvm1,poly['nmsv'],poly['scmsv2xx'])
    polycur = po.get_linspline(xx1,shocks,ind_shocks,acoeff,poly['exoggrid'],poly['steps'],poly['nfunc'],\
            poly['ngrid'],poly['npoly'],poly['nmsv'],poly['ncheb'],poly['ne'])
    (dvar,lvar) = modelvariables(polycur,msvm1,shocks,paramplus,poly['eqswitch'],poly['ne'])
    for x in lvar:
        endogvar[x] = lvar[x]
        endogvar[x+'_d'] = dvar[x]
    return(endogvar)

def simulate_irf(TT,endogvarm1_shk,endogvarm1_base,innov_shk,paramplus,acoeff,poly,varlist):
    #Compute IRFS as deviation from unshocked baseline.
    nvars = len(varlist)
    ne = poly['ne']
    invdf = pd.DataFrame(np.zeros([TT,nvars]),columns=varlist)
    invdf['mu_innov'] = 0.0
    invdf['beta_innov'] = 0.0
    invdf.loc[0,'beta_innov'] = innov_shk[0]
    if ne > 1:
        invdf.loc[0,'mu_innov'] = innov_shk[1]
    innov = np.zeros(poly['ne']) 
    endogvarm1_s = endogvarm1_shk
    endogvarm1_b = endogvarm1_base
    for tt in np.arange(TT):
        innov[0] = invdf.loc[tt,'beta_innov']
        if ne > 1:
            innov[1] = invdf.loc[tt,'mu_innov']
        endogvar = decr(endogvarm1_s,innov,paramplus,acoeff,poly)
        innov[0] = 0.0
        if ne > 1:
            innov[1] = 0.0
        endogvar_b = decr(endogvarm1_b,innov,paramplus,acoeff,poly)
        for x in varlist:
            invdf.loc[tt,x] = 100.0*(endogvar[x+'_d']-endogvar_b[x+'_d'])
        endogvarm1_s = endogvar
        endogvarm1_b = endogvar_b
    return(invdf)

##########################################################################
#Model Development
###########################################################################

#Get steady state
params = {'beta' : 0.99,'delta':0.025,'alpha': 0.3,'a': 0.63,'b' : 10.0,'gamma_A' : 0.002,'gamma_V': 0.003,'rhomu': 0.7,'stdmu' : 0.01,\
           'rhobeta' : 0.9, 'stdbeta': 0.001}
phiitilde = params['b']*params['a']**2
params['a'] = -10.0
params['b'] = phiitilde/(params['a']**2)
paramplus = get_steady(params)

#Solve linear model
import dsge
partial=dsge.DSGE.DSGE.read('partial_adj1_linear.yaml')
partiallin = partial.compile_model()
p0 = np.zeros(len(params))
for i,x in enumerate(params):
    p0[i] = params[x]

# #get linear decision rule
innov = ['eps_beta']
ninnov = len(innov)
msv = ['kp','inv','betashk']  
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

# #solve nonlinear model
ncheb = 3*np.ones(nmsv-1,dtype=int)
ngrid = np.ones(ninnov,dtype=int)
ngrid[0] = 7
msvmax = np.zeros(nmsv-1)
msvmax[0] = 0.1
msvmax[1] = 0.2
maxstd = 2.5
if ninnov > 1:
    sys.exit('Stopping without solving nonlinear model')
eqswitch = 0
poly0 = initialize_poly(npdv,ncheb,ngrid,msvmax,maxstd,eqswitch)
poly0 = get_griddetails(poly0,paramplus)
acoeff0 = get_initcoeffs(lcoeff,poly0)
eqswitch = 1
poly1 = initialize_poly(npdv,ncheb,ngrid,msvmax,maxstd,eqswitch)
poly1 = get_griddetails(poly1,paramplus)
acoeff1,convergence = get_coeffs(acoeff0,paramplus,poly1,step=0.5)
if (convergence == False):
     sys.exit('Failed to solve nonlinear model')
















