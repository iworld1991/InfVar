# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## SMM Estimation of Theories of Expectation Formation with Inflation Expectation
#
# - The code is organized in following ways
#
#   1. Each pair of a theory of expectation formation (re, se, ni, de, denim etc) and an assumed process of inflation process (ar1 or sv)  are encapsulated in a specific python class. 
#     - the class initializes corresponding parameters of the inflation process and expectation formation 
#     - and embodies a specific function that generates all the simulated moments of both inflation and expectations 
#     
#   2. A generally written objective function that computes the distances in moments as a function of parameters specific to the chosen model, moments, and the data. 
#   3.  The general function is to be used to compute the specific objective function that takes parameter as the only input for the minimizer to work
#   4.  Then a general function that does an optimization algorithm takes the specific objective function and estimates the parameters

# +
import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numba as nb
from numba import jit, njit, float64, int64
from numba.experimental import jitclass
from numba import types
from numba.typed import Dict
import pandas as pd
from statsmodels.tsa.api import AutoReg as AR
import copy as cp

pd.options.display.float_format = '{:,.2f}'.format

plt.style.use('ggplot')

lw = 4


# -

# ## Model 

# + code_folding=[2, 23, 47, 72, 120]
## auxiliary functions 
@njit
def hstepvar(h,
             ρ,
             σ):
    '''
    inputs
    ------
    h: forecast horizon
    ρ: ar(1) persistence
    σ: ar(1) volatility (standard deviation)
    
    outputs
    -------
    scalar: h-step-forward variance 
    '''
    ## h-step ahead variance for an ar(1)
    var = 0
    for i in range(h):
        var += ρ**(2*i)*σ**2 
    return var 

@njit
def hstepvarSV(h,
               σs_now,
               γ):
    '''
    inputs
    ------
    h: forecast horizon
    σs_now, 2 x 1 vector [sigma_eta, sigma_eps]
    γ, volatility, scalar. 
    
    outputs
    -------
    scalar: h-step-forward variance 
    '''
    ratio = 0
    for k in range(h):
        ratio += np.exp(-0.5*(k+1)*γ)
    var_eta = σs_now[0]**2*ratio
    var_eps = σs_now[1]**2*np.exp(-0.5*h*γ)
    var = var_eta + var_eps
    return var

### AR1 simulator 
@njit
def SimAR1(ρ,
           σ,
           T):
    '''
    inputs
    ------
    T: nb of periods for the simulated history 
    ρ: ar(1) persistence
    σ: ar(1) volatility (standard deviation)
    
    outputs
    -------
    xxx: T periods of history of AR(1) with the first 20 periods burned 
    '''
        
    t_burn = 20
    xxx = np.zeros(T+1+t_burn)
    shocks = np.random.randn(T+1+t_burn)*σ
    xxx[0] = 0 
    for i in range(T+t_burn):
        xxx[i+1] = ρ*xxx[i] + shocks[i+1]
    return xxx[1+t_burn:]

### UC-SV simulator 
@njit
def SimUCSV(γ,
            nobs,
            p0 = 0,
            seed = False):
    """
    input
    ======
    p: permanent 
    t: transitory 
    
    output
    ======
    y: the draw of series
    p: the permanent component
    svols_p: permanent volatility 
    svols_t: transitory volatility 
    """
    if seed == True:
        np.random.seed(12344)
    else:
        pass
    svols_p_shock = np.random.randn(nobs+1)*γ
    svols_t_shock = np.random.randn(nobs+1)*γ
    
    svols_p = np.zeros(nobs+1)
    svols_p[0] = 0.001
    svols_t = np.zeros(nobs+1)
    svols_t[0] = 0.001
    for i in range(nobs):
        svols_p[i+1] = np.sqrt( np.exp(np.log(svols_p[i]**2) + svols_p_shock[i+1]) ) 
        svols_t[i+1] = np.sqrt( np.exp(np.log(svols_t[i]**2) + svols_t_shock[i+1]) ) 
    shocks_p = np.multiply(np.random.randn(nobs+1),svols_p)  
    shocks_t = np.multiply(np.random.randn(nobs+1),svols_t)
    
    p = np.zeros(nobs+1)
    t = np.zeros(nobs+1)
    
    ## initial level of eta, 0 by default
    p[0] = p0
    
    for i in range(nobs):
        p[i+1] = p[i] + shocks_p[i+1]
        t[i+1] = shocks_t[i+1]
        
    y = p + t
    return y, p, svols_p, svols_t

@njit
def d1tod2(x):
    '''
    inputs
    ------
    x: 1-dimension array 
    
    outputs
    -------
    2-dimension array 
    '''
    return x.reshape(1,-1)


# + code_folding=[1, 37]
@njit
def ObjGen(model,
           paras,
           data_mom_dict,
           moment_choice,
           how = 'expectation',
           n_exp_paras = 0):
    '''
    inputs
    ------
    model: a model class, i.e, sear represnting sticky expectation and ar(1) 
    paras: an array vector of the parameters, which potentially includes both inflation process and expectation 
    data_mom_dic: a dictionary storing all data moments
    moment_choice: a list of moments, i.e. ['FE','FEATV','Var']
    how: string taking values of 'expectation','process','joint'
    n_exp_paras: nb of parameters for expectation model 
    
    outputs
    -------
    distance: the scalor of the moment distances to be minimized
    '''
    if how =='expectation':
        model.exp_para = paras
    elif how=='process':
        model.process_para = paras
    elif how=='joint':
        model.exp_para = paras[0:n_exp_paras]
        model.process_para = paras[n_exp_paras:]
        
    # simulated moments 
    model_mom_dict = model.SMM()
    diff = np.array([model_mom_dict[mom] - data_mom_dict[mom] for mom in moment_choice]) 
    distance = np.linalg.norm(diff)
    
    return distance

@njit
def ObjWeight(model,
           paras,
           data_mom_dict,
           moment_choice,
           weight,
           how = 'expectation',
           n_exp_paras = 0):
    '''
    - same as above ObjGen, except for with one additional argument weight for weighted distance 
    '''
    
    if how =='expectation':
        model.exp_para = paras
    elif how=='process':
        model.process_para = paras
    elif how=='joint':
        model.exp_para = paras[0:n_exp_paras]
        model.process_para = paras[n_exp_paras:]
        
    # simulated moments 
    model_mom_dict = model.SMM()
    diff = np.array([model_mom_dict[mom] - data_mom_dict[mom] for mom in moment_choice])
    distance = np.dot(np.dot(diff,weight),diff.T)  ## need to make sure it is right. 
    return distance


# + code_folding=[1]
## parameter estimation non-jitted because jit does not support scipy.optimize
def ParaEst(ObjSpec,
            para_guess,
            method = 'Nelder-Mead',
            bounds = None,
            options = {'disp': True}):
    """
    an estimating function that minimizes OjbSpec function that gives parameter estimates
    """
    results = minimize(ObjSpec,
                         x0 = para_guess,
                         method = method,
                         bounds = bounds,
                         options = options)
    if results['success']==True:
        parameter = results['x']
    else:
        parameter = np.array([])
    
    return parameter 


# -

# ### Rational Expectation (RE) + AR1

# + code_folding=[]
model_data = [
    ('exp_para', float64[:]),             # parameters for expectation formation, empty for re
    ('process_para', float64[:]),         # parameters for inflation process, 2 entries for AR1 
    ('horizon', int64),                   # forecast horizons 
    ('real_time',float64[:]),             # real time data on inflation 
    ('history',float64[:]),               # a longer history of inflation 
    ('realized',float64[:])               # realized inflation 
]


# + code_folding=[2, 14, 18, 34, 39, 73]
@jitclass(model_data)
class RationalExpectationAR:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
        
    def SimForecasts(self):
        ## parameters
        n = len(self.real_time)
        ρ,σ = self.process_para
        horizon = self.horizon

        ## information set 
        real_time = self.real_time
        
        ## forecast moments 
        Disg = np.zeros(n)
        nowcast = real_time
        forecast = ρ**horizon*nowcast
        Var = hstepvar(horizon,ρ,σ)* np.ones(n)
        FE = forecast - self.realized           ## forecast errors depend on realized shocks
        FEATV = np.zeros(n)
        forecast_moments = {"FE":FE,
                            "Disg":Disg,
                            "Var":Var}
        return forecast_moments
    
    def SMM(self):
        
        ρ,σ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = 0.0
        InfVar = σ**2/(1-ρ**2)
        InfATV = ρ*InfVar
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
        VarVar_sim = np.var(Vars_sim)
        VarATV_sim = np.cov(np.stack( (Vars_sim[1:],Vars_sim[:-1]),axis = 0))[0,1]
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim,
                      'VarVar':VarVar_sim,
                      'VarATV':VarATV_sim}
        return SMMMoments


# -

# ### Rational Expectation (RE) + SV

# + code_folding=[0]
model_sv_data = [
    ('exp_para', float64[:]),             # parameters for expectation formation, empty for re
    ('process_para', float64[:]),         # parameters for inflation process, 2 entries for AR1 
    ('horizon', int64),                   # forecast horizons 
    ('real_time',float64[:,:]),             # real time data on inflation 
    ('history',float64[:,:]),               # a longer history of inflation 
    ('realized',float64[:])               # realized inflation 
]


# + code_folding=[2, 14, 18, 50]
@jitclass(model_sv_data)
class RationalExpectationSV:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
        
    def SimForecasts(self):
        ## parameters
        n = len(self.real_time[0,:])
        γ = self.process_para
            
        ## inputs
        real_time = self.real_time
        horizon = self.horizon
        
        ## forecast moments 
        ## now the informationset needs to contain differnt components seperately.
        ## therefore, real_time fed in the begining is a tuple, not just current eta, but also current sigmas. 
        
        infoset = real_time 
        y_real_time = infoset[0,:]  ## permanent income componenet 
        nowcast = infoset[1,:]  ## permanent income componenet 
        forecast = nowcast
        σs_now = infoset[2:3,:]  ## volatility now 
        
        Var = np.zeros(n)
        for i in range(n):
            Var[i] = hstepvarSV(horizon,
                                σs_now = σs_now[:,i],
                                γ = γ[0]) ## γ[0] instead of γ is important make sure the input is a float
        FE = forecast - self.realized ## forecast errors depend on realized shocks 
        Disg = np.zeros(n)
        
        forecast_moments = {"FE":FE,
                            "Disg":Disg,
                            "Var":Var}
        return forecast_moments
        
    def SMM(self):
        
        γ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = np.nan
        InfVar = np.nan
        InfATV = np.nan
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
        VarVar_sim = np.var(Vars_sim)
        VarATV_sim = np.cov(np.stack( (Vars_sim[1:],Vars_sim[:-1]),axis = 0))[0,1]
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim,
                      'VarVar':VarVar_sim,
                      'VarATV':VarATV_sim}
        return SMMMoments


# + code_folding=[0, 6]
### create a RESV instance 

p0_fake = 0
γ_fake = 0.1
σs_now_fake = [0.2,0.3]

ucsv_fake = SimUCSV(γ_fake,
                    nobs = 200,
                    p0 = p0_fake,
                    ) 

xx_real_time,xx_p_real_time,vol_p_real_time,vol_t_real_time = ucsv_fake  

xx_realized = xx_real_time[1:-1]

xx_real_time= np.array([xx_real_time,
                        xx_p_real_time,
                        vol_p_real_time,
                        vol_t_real_time]
                      )[:,0:-2]


## initialize 
resv = RationalExpectationSV(exp_para = np.array([]),
                             process_para = np.array([0.1]),
                             real_time = xx_real_time,
                             history = xx_real_time) ## history does not matter here, 

## get the realization 

resv.GetRealization(xx_realized)
# -

# #### Estimation inflation only 

# + code_folding=[0, 14, 29]
## generate an instance of the model

ρ0,σ0 = 0.95,0.1

history0 = SimAR1(ρ0,
                  σ0,
                  200)
real_time0 = history0[11:-2]

realized0 = history0[12:-1]


## initialize an re instance 

rear0 = RationalExpectationAR(exp_para = np.array([]),
                              process_para = np.array([ρ0,σ0]),
                              real_time = real_time0,
                              history = history0,
                              horizon = 1)

rear0.GetRealization(realized0)


## fake data moments dictionary 

data_mom_dict_re = rear0.SMM()


## specific objective function for estimation 
moments0 = ['InfAV',
           'InfVar',
           'InfATV']


## specific objective function 

def Objrear(paras):
    scalor = ObjGen(rear0,
                    paras= paras,
                    data_mom_dict = data_mom_dict_re,
                    moment_choice = moments0,
                    how = 'process')
    return scalor


# + code_folding=[]
## invoke estimation 

#ParaEst(Objrear,
#        para_guess = (0.8,0.1)
#       )
# -

# ### Sticky Expectation (SE) + AR1

# + code_folding=[3]
### Some jitted functions that are needed (https://github.com/numba/numba/issues/1269)

@njit
def np_apply_along_axis(func1d, axis, arr):
    assert arr.ndim == 2
    assert axis in [0, 1]
    if axis == 0:
        result = np.empty(arr.shape[1])
        for i in range(len(result)):
            result[i] = func1d(arr[:, i])
    else:
        result = np.empty(arr.shape[0])
        for i in range(len(result)):
            result[i] = func1d(arr[i, :])
    return result

@njit
def np_mean(array, axis):
    return np_apply_along_axis(np.mean, axis, array)

@njit
def np_std(array, axis):
    return np_apply_along_axis(np.std, axis, array)

@njit
def np_var(array, axis):
    return np_apply_along_axis(np.var, axis, array)

@njit
def np_max(array, axis):
    return np_apply_along_axis(np.max, axis, array)


@njit
def np_min(array, axis):
    return np_apply_along_axis(np.min, axis, array)


# + code_folding=[2, 18, 81]
@jitclass(model_data)
class StickyExpectationAR:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
        
    def SimForecasts(self,
                     n_sim = 500):        
        ## parameters and inputs 
        real_time = self.real_time
        history  = self.history
        
        n = len(real_time)
        ρ,σ = self.process_para
        lbd = self.exp_para
        horizon = self.horizon
        
        n_history = len(history) # of course equal to len(history)
        n_burn = len(history) - n
        
        ## simulation
        np.random.seed(12345)
        update_or_not_val = np.random.uniform(0,
                                              1,
                                              size = (n_sim,n_history))
        update_or_not_bool = update_or_not_val>=1-lbd
        update_or_not = update_or_not_bool.astype(np.int64)
        most_recent_when = np.empty((n_sim,n_history),dtype = np.int64)
        nowcasts_to_burn = np.empty((n_sim,n_history),dtype = np.float64)
        Vars_to_burn = np.empty((n_sim,n_history),dtype = np.float64)
        
        # look back for the most recent last update for each point of time  
        for i in range(n_sim):
            for j in range(n_history):
                most_recent = j 
                for x in range(j):
                    if update_or_not[i,j-x]==1 and most_recent<=x:
                        most_recent = most_recent
                    elif update_or_not[i,j-x]==1 and most_recent>x:
                        most_recent = x
                most_recent_when[i,j] = most_recent
                nowcasts_to_burn[i,j] = history[j - most_recent_when[i,j]]*ρ**most_recent_when[i,j]
                Vars_to_burn[i,j]= hstepvar((most_recent_when[i,j]+horizon),
                                            ρ,
                                            σ)
        
        ## burn initial forecasts since history is too short 
        nowcasts = nowcasts_to_burn[:,n_burn:] 
        forecasts = ρ**horizon*nowcasts
        Vars = Vars_to_burn[:,n_burn:]
        FEs = forecasts - self.realized
        
        ## compuate population moments
        forecasts_mean = np_mean(forecasts,axis = 0)
        forecasts_var = np_var(forecasts,axis = 0)
        FEs_mean = forecasts_mean - self.realized
            
        Vars_mean = np_mean(Vars,axis = 0) ## need to change 
        
        forecasts_vcv = np.cov(forecasts.T)
        forecasts_atv = np.array([forecasts_vcv[i+1,i] for i in range(n-1)])
        FEs_vcv = np.cov(FEs.T)
        FEs_atv = np.array([FEs_vcv[i+1,i] for i in range(n-1)]) ## this is no longer needed
        
        forecast_moments_sim = {"FE":FEs_mean,
                                "Disg":forecasts_var,
                                "Var":Vars_mean}
        return forecast_moments_sim
            
    def SMM(self):
        
        ρ,σ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = 0.0
        InfVar = σ**2/(1-ρ**2)
        InfATV = ρ*InfVar
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
        VarVar_sim = np.var(Vars_sim)
        VarATV_sim = np.cov(np.stack( (Vars_sim[1:],Vars_sim[:-1]),axis = 0))[0,1]
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim,
                      'VarVar':VarVar_sim,
                      'VarATV':VarATV_sim}
        return SMMMoments


# -

# ### Sticky Expectation (SE) + SV

# + code_folding=[2, 14, 18, 88]
@jitclass(model_sv_data)
class StickyExpectationSV:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
        
    def SimForecasts(self,
                     n_sim = 502):        
        ## inputs 
        real_time = self.real_time
        history  = self.history
        n = len(real_time[0,:])
        horizon = self.horizon
        n_history = len(history[0,:]) # of course equal to len(history)
        n_burn = n_history - n
        
        ## get the information set 
        infoset = history 
        y_now,p_now, sigmas_p_now, sigmas_t_now= infoset[0,:],infoset[1,:],infoset[2,:],infoset[3,:]
        sigmas_now = np.concatenate((sigmas_p_now,sigmas_t_now),axis=0).reshape((2,-1))
        
        ## parameters 
        γ = self.process_para
        lbd = self.exp_para
       
        ## simulation
        np.random.seed(12345)
        update_or_not_val = np.random.uniform(0,
                                              1,
                                              size = (n_sim,n_history))
        update_or_not_bool = update_or_not_val>=1-lbd
        update_or_not = update_or_not_bool.astype(np.int64)
        most_recent_when = np.empty((n_sim,n_history),dtype = np.int64)
        nowcasts_to_burn = np.empty((n_sim,n_history),dtype = np.float64)
        Vars_to_burn = np.empty((n_sim,n_history),dtype = np.float64)
        
        # look back for the most recent last update for each point of time  
        for i in range(n_sim):
            for j in range(n_history):
                most_recent = j 
                for x in range(j):
                    if update_or_not[i,j-x]==1 and most_recent<=x:
                        most_recent = most_recent
                    elif update_or_not[i,j-x]==1 and most_recent>x:
                        most_recent = x
                most_recent_when[i,j] = most_recent
                ###########################################################################
                nowcasts_to_burn[i,j] = p_now[j - most_recent_when[i,j]]
                Vars_to_burn[i,j]= hstepvarSV((most_recent_when[i,j]+horizon),
                                              sigmas_now[:,j-most_recent_when[i,j]],
                                              γ[0])
                ###############################################################
        
        ## burn initial forecasts since history is too short 
        nowcasts = nowcasts_to_burn[:,n_burn:] 
        forecasts = nowcasts
        Vars = Vars_to_burn[:,n_burn:]
        FEs = forecasts - self.realized
        
        ## compuate population moments
        forecasts_mean = np_mean(forecasts,axis = 0)
        forecasts_var = np_var(forecasts,axis = 0)
        FEs_mean = forecasts_mean - self.realized
            
        Vars_mean = np_mean(Vars,axis = 0) ## need to change 
        
        forecasts_vcv = np.cov(forecasts.T)
        forecasts_atv = np.array([forecasts_vcv[i+1,i] for i in range(n-1)])
        FEs_vcv = np.cov(FEs.T)
        FEs_atv = np.array([FEs_vcv[i+1,i] for i in range(n-1)]) ## this is no longer needed
        
        forecast_moments_sim = {"FE":FEs_mean,
                                "Disg":forecasts_var,
                                "Var":Vars_mean}
        return forecast_moments_sim
        
    def SMM(self):
        
        γ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = np.nan
        InfVar = np.nan
        InfATV = np.nan
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
        VarVar_sim = np.var(Vars_sim)
        VarATV_sim = np.cov(np.stack( (Vars_sim[1:],Vars_sim[:-1]),axis = 0))[0,1]
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim,
                      'VarVar':VarVar_sim,
                      'VarATV':VarATV_sim}
        return SMMMoments


# + code_folding=[]
## intialize the ar instance 
sear0 = StickyExpectationAR(exp_para = np.array([0.2]),
                            process_para = np.array([ρ0,σ0]),
                            real_time = real_time0,
                            history = history0,
                            horizon = 1)

sear0.GetRealization(realized0)

# + code_folding=[]
## intialize the sv instnace 
sesv0 = StickyExpectationSV(exp_para = np.array([0.3]),
                           process_para = np.array([0.1]),
                           real_time = xx_real_time,
                           history = xx_real_time) ## history does not matter here, 

## get the realization 

sesv0.GetRealization(xx_realized)
# -

# #### Estimating SE with RE 

# + code_folding=[0]
## only expectation estimation 

moments0 = ['FE',
            'FEVar',
            'FEATV',
            'Disg']

def Objsear_re(paras):
    scalor = ObjGen(sear0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_re,
                    moment_choice = moments0,
                    how = 'expectation')
    return scalor


## invoke estimation 
#ParaEst(Objsear_re,
#        para_guess = (0.2),
#        method='Nelder-Mead',
#        bounds = ((0,1),)
#       )
# -

# #### Estimating SE with SE 

# + code_folding=[0]
## get a fake data moment dictionary under a different parameter 

sear1 = StickyExpectationAR(exp_para = np.array([0.4]),
                            process_para = np.array([ρ0,σ0]),
                            real_time = real_time0,
                            history = history0,
                            horizon = 1)
sear1.GetRealization(realized0)
data_mom_dict_se = sear1.SMM()

moments0 = ['FE',
            'FEVar',
            'FEATV',
            'Disg']

def Objsear_se(paras):
    scalor = ObjGen(sear0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_se,
                    moment_choice = moments0,
                    how = 'expectation')
    return scalor


## invoke estimation 
#ParaEst(Objsear_se,
#        para_guess = (0.2),
#        method='Nelder-Mead',
#        bounds = ((0,1),)
#       )
# -

# #### Joint Estimation 

# + code_folding=[0, 2, 10]
## for joint estimation 

moments1 = ['InfAV',
            'InfVar',
            'InfATV',
            'FE',
            'FEVar',
            'FEATV',
            'Disg']

def Objsear_joint(paras):
    scalor = ObjGen(sear0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_se,
                    moment_choice = moments1,
                    how ='joint',
                    n_exp_paras = 1)
    return scalor

## invoke estimation 
#ParaEst(Objsear_joint,
#        para_guess = np.array([0.2,0.8,0.2]),
#        method='Nelder-Mead'
#       )


# -

# ### Noisy Information (NI) + AR1
#

# + code_folding=[1]
@njit
def SteadyStateVar(process_para,
                   exp_para):
    ## steady state variance for kalman filtering 
    ρ,σ = process_para
    sigma_pb,sigma_pr = exp_para
    a = ρ**2*(sigma_pb**2+sigma_pr**2)
    b = (sigma_pb**2+sigma_pr**2)*σ**2+(1-ρ**2)*sigma_pb**2*sigma_pr**2
    c = -σ**2*sigma_pb**2*sigma_pr**2
    nowcast_var_ss = (-b+np.sqrt(b**2-4*a*c))/(2*a)
    return nowcast_var_ss


# + code_folding=[1, 2, 14, 18, 117, 151]
@jitclass(model_data)
class NoisyInformationAR:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
        
    def SimForecasts(self,
                     n_sim = 500):
        ## inputs 
        
        real_time = self.real_time
        history = self.history
        realized = self.realized
        n = len(real_time)
        n_history = len(history)
        n_burn = len(history) - n
        
        ## parameters 
        ρ,σ = self.process_para
        sigma_pb,sigma_pr = self.exp_para

        #######################
        ## using uncertainty at steady state of the Kalman filtering
        var_init = SteadyStateVar(self.process_para,
                                  self.exp_para)    ## some initial level of uncertainty, will be washed out after long simulation
        ##################
        sigma_v = np.array([[sigma_pb**2,0.0],[0.0,sigma_pr**2]]) ## variance matrix of signal noises 
        horizon = self.horizon      
        
        ## simulate signals 
        nb_s = 2                                    ## the number of signals 
        H = np.array([[1.0],[1.0]])                 ## an multiplicative matrix summing all signals
        
        # randomly simulated signals 
        np.random.seed(12434)
        signal_pb = self.history+sigma_pb*np.random.randn(n_history)   ## one series of public signals 
        signals_pb = signal_pb.repeat(n_sim).reshape((-1,n_sim)).T     ## shared by all agents
        np.random.seed(13435)
        signals_pr = self.history + sigma_pr*np.random.randn(n_sim*n_history).reshape((n_sim,n_history))
                                                                 ### private signals are agent-specific 
    
        ## prepare matricies 
        nowcasts_to_burn = np.zeros((n_sim,n_history))  ### nowcasts matrix of which the initial simulation is to be burned 
        nowcasts_to_burn[:,0] = history[0]
        nowvars_to_burn = np.zeros((n_sim,n_history))   ### nowcasts uncertainty matrix
        nowvars_to_burn[:,0] = var_init
        Vars_to_burn = np.zeros((n_sim,n_history))      ### forecasting uncertainty matrix 
        
        
        ## fill the matricies for individual moments        
        for i in range(n_sim):
            signals_this_i = np.concatenate((signals_pb[i,:],signals_pr[i,:]),axis=0).reshape((2,-1))
            ## the histories signals specific to i: the first row is public signals and the second is private signals 
            Pkalman = np.zeros((n_history,nb_s))
            ## Kalman gains of this agent for respective signals 
            Pkalman[0,:] = 0  ## some initial values 
            
            for t in range(n_history-1):
                step1_vars_to_burn = ρ**2*nowvars_to_burn[i,t] + σ**2
                ## priror uncertainty 
                
                inv = np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v) 
                ## the inverse of the noiseness matrix  
                
                inv_sc = np.dot(np.dot(H.T,inv),H)
                ## the total noisiness as a scalar 
                
                var_reduc = step1_vars_to_burn*inv_sc*step1_vars_to_burn
                ## reduction in uncertainty from the update
                
                nowvars_this_2d = np.array([[step1_vars_to_burn]]) - var_reduc
                ## update equation of nowcasting uncertainty 
                
                nowvars_to_burn[i,t+1] = nowvars_this_2d[0,0] 
                ## nowvars_this_2d is a 2-d matrix with only one entry. We take the element and set it to the matrix
                ### this is necessary for Numba typing 
                
                Pkalman[t+1,:] = step1_vars_to_burn*np.dot(H.T,np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v))
                ## update Kalman gains recursively using the signal extraction ratios 
                
                Pkalman_all = np.dot(Pkalman[t+1,:],H)[0] 
                ## the weight to the prior forecast 
    
                nowcasts_to_burn[i,t+1] = (1-Pkalman_all)*ρ*nowcasts_to_burn[i,t]+ np.dot(Pkalman[t+1,:],signals_this_i[:,t+1])
                ## kalman filtering updating for nowcasting: weighted average of prior and signals 
                
            for t in range(n_history):
                Vars_to_burn[i,t] = ρ**(2*horizon)*nowvars_to_burn[i,t] + hstepvar(horizon,ρ,σ)
                
        nowcasts = nowcasts_to_burn[:,n_burn:]
        forecasts = ρ**horizon*nowcasts 
        Vars = Vars_to_burn[:,n_burn:]
        
        ## compuate population moments
        forecasts_mean = np_mean(forecasts,axis=0)
        forecasts_var = np_var(forecasts,axis=0)
        FEs_mean = forecasts_mean - realized
            
        Vars_mean = np_mean(Vars,axis=0) ## need to change for time-variant volatility
        
        forecast_moments_sim = {"FE":FEs_mean,
                                "Disg":forecasts_var,
                                "Var":Vars_mean}
        return forecast_moments_sim
    
    def SMM(self):
        
        ρ,σ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = 0.0
        InfVar = σ**2/(1-ρ**2)
        InfATV = ρ*InfVar
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
        VarVar_sim = np.var(Vars_sim)
        VarATV_sim = np.cov(np.stack( (Vars_sim[1:],Vars_sim[:-1]),axis = 0))[0,1]
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim,
                      'VarVar':VarVar_sim,
                      'VarATV':VarATV_sim}
        return SMMMoments


# -

# ### Noisy Information (NI) + SV
#
#

# + code_folding=[1, 2, 14, 18, 62, 123, 157]
@jitclass(model_sv_data)
class NoisyInformationSV:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
              
    def SimForecasts(self,
                     n_sim = 500):
        ## inputs 
        real_time = self.real_time
        history  = self.history
        n = len(real_time[0,:])
        horizon = self.horizon
        n_history = len(history[0,:]) # of course equal to len(history)
        n_burn = n_history - n
        
        ## get the information set 
        infoset = history 
        y_now, p_now, sigmas_p_now, sigmas_t_now= infoset[0,:],infoset[1,:],infoset[2,:],infoset[3,:]
        sigmas_now = np.concatenate((sigmas_p_now,sigmas_t_now),axis=0).reshape((2,-1))
        
        ## process parameters
        γ = self.process_para
        ## exp parameters 
        sigma_pb,sigma_pr = self.exp_para
        var_init = sigmas_now[0,0]**2+sigmas_now[1,0]**2
        
        ## other parameters 
        sigma_v = np.array([[sigma_pb**2,0.0],[0.0,sigma_pr**2]]) ## variance matrix of signal noises         
        ## simulate signals 
        nb_s = 2                                    ## the number of signals 
        H = np.array([[1.0],[1.0]])                 ## an multiplicative matrix summing all signals
        
        # randomly simulated signals 
        np.random.seed(12434)
        ##########################################################
        signal_pb = p_now+sigma_pb*np.random.randn(n_history)   ## one series of public signals 
        signals_pb = signal_pb.repeat(n_sim).reshape((-1,n_sim)).T     ## shared by all agents
        np.random.seed(13435)
        signals_pr = p_now + sigma_pr*np.random.randn(n_sim*n_history).reshape((n_sim,n_history))
        #####################################################################################

        ## prepare matricies 
        nowcasts_to_burn = np.zeros((n_sim,n_history))
        nowcasts_to_burn[:,0] = p_now[0]
        nowvars_to_burn = np.zeros((n_sim,n_history))
        nowvars_to_burn[:,0] = var_init
        Vars_to_burn = np.zeros((n_sim,n_history))
        
        ## fill the matricies for individual moments        
        for i in range(n_sim):
            signals_this_i = np.concatenate((signals_pb[i,:],signals_pr[i,:]),axis=0).reshape((2,-1))
            ## the histories signals specific to i: the first row is public signals and the second is private signals 
            Pkalman = np.zeros((n_history,nb_s))
            ## Kalman gains of this agent for respective signals 
            Pkalman[0,:] = 0  ## some initial values 
            
            for t in range(n_history-1):
                step1var = hstepvarSV(1,
                                      sigmas_now[:,t],
                                      γ[0])
                step1_vars_to_burn = nowvars_to_burn[i,t] + step1var
                ## priror uncertainty 
                
                inv = np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v) 
                ## the inverse of the noiseness matrix  
                
                inv_sc = np.dot(np.dot(H.T,inv),H)
                ## the total noisiness as a scalar 
                
                var_reduc = step1_vars_to_burn*inv_sc*step1_vars_to_burn
                ## reduction in uncertainty from the update
                
                nowvars_this_2d = np.array([[step1_vars_to_burn]]) - var_reduc
                ## update equation of nowcasting uncertainty 
                
                nowvars_to_burn[i,t+1] = nowvars_this_2d[0,0] 
                ## nowvars_this_2d is a 2-d matrix with only one entry. We take the element and set it to the matrix
                ### this is necessary for Numba typing 
                
                Pkalman[t+1,:] = step1_vars_to_burn*np.dot(H.T,np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v))
                ## update Kalman gains recursively using the signal extraction ratios 
                
                Pkalman_all = np.dot(Pkalman[t+1,:],H)[0] 
                ## the weight to the prior forecast 
    
                nowcasts_to_burn[i,t+1] = (1-Pkalman_all)*nowcasts_to_burn[i,t]+ np.dot(Pkalman[t+1,:],signals_this_i[:,t+1])
                ## kalman filtering updating for nowcasting: weighted average of prior and signals 
                
            for t in range(n_history):
                stephvar = hstepvarSV(horizon,
                                      sigmas_now[:,t],
                                      γ[0])
                Vars_to_burn[i,t] = nowvars_to_burn[i,t] + stephvar
                
        nowcasts = nowcasts_to_burn[:,n_burn:]
        forecasts = nowcasts 
        Vars = Vars_to_burn[:,n_burn:]

        
        ## compuate population moments
        forecasts_mean = np_mean(forecasts,axis=0)
        forecasts_var = np_var(forecasts,axis=0)
        FEs_mean = forecasts_mean - self.realized
        Vars_mean = np_mean(Vars,axis=0) ## need to change for time-variant volatility
        
        forecast_moments_sim = {"FE":FEs_mean,
                                "Disg":forecasts_var,
                                "Var":Vars_mean}
        return forecast_moments_sim
        
    def SMM(self):
        
        γ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = np.nan
        InfVar = np.nan
        InfATV = np.nan
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
        VarVar_sim = np.var(Vars_sim)
        VarATV_sim = np.cov(np.stack( (Vars_sim[1:],Vars_sim[:-1]),axis = 0))[0,1]
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim,
                      'VarVar':VarVar_sim,
                      'VarATV':VarATV_sim}
        return SMMMoments

# + code_folding=[1]
## intialize the ar instance 
niar0 = NoisyInformationAR(exp_para = np.array([0.1,0.2]),
                            process_para = np.array([ρ0,σ0]),
                            real_time = real_time0,
                            history = history0,
                            horizon = 1)

niar0.GetRealization(realized0)

# + code_folding=[0]
## initial a sv instance
nisv0 = NoisyInformationSV(exp_para = np.array([0.3,0.2]),
                           process_para = np.array([0.1]),
                           real_time = xx_real_time,
                           history = xx_real_time) ## history does not matter here, 

## get the realization 

nisv0.GetRealization(xx_realized)
# -

# #### Estimating NI using RE data 

# + code_folding=[0, 11]
moments0 = ['FE',
            'FEATV',
            'FEVar',
            'Disg',
            'DisgATV',
            'DisgVar',
            'Var',
            'VarVar',
            'VarATV']

def Objniar_re(paras):
    scalor = ObjGen(niar0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_re,
                    moment_choice = moments0,
                    how = 'expectation')
    return scalor

## invoke estimation 
#ParaEst(Objniar_re,
#        para_guess = np.array([0.2,0.1]),
#        method='L-BFGS-B',
#       bounds=((0,None),(0,None),))


# -

# #### Estimate NI with NI

# + code_folding=[2, 12, 19]
## get a fake data moment dictionary under a different parameter 

niar1 = NoisyInformationAR(exp_para = np.array([0.5,0.9]),
                            process_para = np.array([ρ0,σ0]),
                            real_time = real_time0,
                            history = history0,
                            horizon = 1)

niar1.GetRealization(realized0)

data_mom_dict_ni = niar1.SMM()

moments0 = ['FE',
            'FEVar',
            'FEATV',
            'Disg',
            'DisgVar',
            'Var']

def Objniar_ni(paras):
    scalor = ObjGen(niar0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_ni,
                    moment_choice = moments0,
                    how = 'expectation')
    return scalor

## invoke estimation 

#ParaEst(Objniar_ni,
#        para_guess = np.array([0.2,0.3]),
#        method='L-BFGS-B',  # see to work for both L-BFGS-B and Nelder-Mead
#        bounds = ((0,None),(0,None),)
#       )


## works correctly 
# -

# #### Joint Estimation
#

# + code_folding=[0, 2]
## for joint estimation 

moments1 = ['InfAV',
            'InfVar',
            'InfATV',
            'FE',
            'FEVar',
            'FEATV',
            'Disg',
           'DisgVar',
           'DisgATV',
           'Var',
           'VarVar',
           'VarATV']

def Objniar_joint(paras):
    scalor = ObjGen(niar0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_ni,
                    moment_choice = moments1,
                    how ='joint',
                    n_exp_paras = 2)
    return scalor

## invoke estimation 
#ParaEst(Objniar_joint,
#        para_guess = np.array([0.2,0.3,0.8,0.2]),
#        method='Nelder-Mead'
#       )

## works correctly w


# -

# ###  Diagnostic Expectation(DE) + AR1

# + code_folding=[1, 2, 14, 71]
@jitclass(model_data)
class DiagnosticExpectationAR:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
        
    def SimForecasts(self,
                     n_sim = 500):
        ## inputs 
        real_time = self.real_time
        history  = self.history
        realized = self.realized
        n = len(real_time)
        horizon = self.horizon 
        n_burn = len(history) - n
        n_history = len(history)  # of course equal to len(history)
        
        ## parameters 
        ρ,σ = self.process_para
        theta,theta_sigma = self.exp_para
        
    
        ## simulation
        np.random.seed(12345)
        thetas = theta_sigma*np.random.randn(n_sim) + theta  ## randomly drawn representativeness parameters
        
        nowcasts_to_burn = np.empty((n_sim,n_history))
        Vars_to_burn = np.empty((n_sim,n_history))
        nowcasts_to_burn[:,0] = history[0]
        Vars_to_burn[:,:] = hstepvar(horizon,ρ,σ)
        
        ## diagnostic and extrapolative expectations 
        for i in range(n_sim):
            this_theta = thetas[i]
            for j in range(n_history-1):
                nowcasts_to_burn[i,j+1] = history[j+1]+ this_theta*(history[j+1]-ρ*history[j])  # can be nowcasting[j-1] instead
        
        ## burn initial forecasts since history is too short 
        nowcasts = nowcasts_to_burn[:,n_burn:]
        forecasts = ρ**horizon*nowcasts
        Vars = Vars_to_burn[:,n_burn:]
        
        FEs = forecasts - realized
        
        ## compuate population moments
        forecasts_mean = np_mean(forecasts,axis = 0)
        forecasts_var = np_var(forecasts,axis = 0)
        FEs_mean = forecasts_mean - realized  
        Vars_mean = np_mean(Vars,axis = 0) ## need to change 
        
        #forecasts_vcv = np.cov(forecasts.T)
        #forecasts_atv = np.array([forecasts_vcv[i+1,i] for i in range(n-1)])
        #FEs_vcv = np.cov(FEs.T)
        
        forecast_moments_sim = {"FE":FEs_mean,
                                "Disg":forecasts_var,
                                "Var":Vars_mean}
        return forecast_moments_sim
    
    def SMM(self):
        
        ρ,σ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = 0.0
        InfVar = σ**2/(1-ρ**2)
        InfATV = ρ*InfVar
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
        VarVar_sim = np.var(Vars_sim)
        VarATV_sim = np.cov(np.stack( (Vars_sim[1:],Vars_sim[:-1]),axis = 0))[0,1]
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim,
                      'VarVar':VarVar_sim,
                      'VarATV':VarATV_sim}
        return SMMMoments


# -

# ###  Diagnostic Expectation(DE) + SV

# + code_folding=[1, 2, 18, 55, 57]
@jitclass(model_sv_data)
class DiagnosticExpectationSV:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
        
    def SimForecasts(self,
                     n_sim = 500):
        ## inputs 
        real_time = self.real_time
        history  = self.history
        realized = self.realized
        n = len(real_time[0,:])
        horizon = self.horizon
        n_history = len(history[0,:]) # of course equal to len(history)
        n_burn = n_history - n
        
        
        ## get the information set 
        infoset = history 
        y_now, p_now, sigmas_p_now, sigmas_t_now= infoset[0,:],infoset[1,:],infoset[2,:],infoset[3,:]
        sigmas_now = np.concatenate((sigmas_p_now,sigmas_t_now),axis=0).reshape((2,-1))
        
        ## process parameters
        γ = self.process_para
        ## exp parameters 
        theta,theta_sigma= self.exp_para
        
    
        ## simulation of representativeness parameters 
        np.random.seed(12345)
        thetas = theta_sigma*np.random.randn(n_sim) + theta  ## randomly drawn representativeness parameters
        
        
        ## simulation of individual forecasts     
        nowcasts_to_burn = np.empty((n_sim,n_history))
        nowcasts_to_burn[:,0] = p_now[0]
        
        Vars_to_burn = np.empty((n_sim,n_history))
        Vars_to_burn[:,0] = hstepvarSV(horizon,
                                       sigmas_now[:,0],
                                       γ[0])
        
        for i in range(n_sim):
            this_theta = thetas[i]
            for j in range(n_history-1):
                ###########################################################################################
                nowcasts_to_burn[i,j+1] = y_now[j+1]+ this_theta*(y_now[j+1]- p_now[j])  # can be nowcasting[j-1] instead
                Var_now_re = hstepvarSV(horizon,
                                        sigmas_now[:,j+1],
                                        γ[0])
                Vars_to_burn[i,j+1] = Var_now_re #+ this_theta*(Var_now_re - Vars_to_burn[i,j])
                ######### this line of codes needs to be double checked!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                ##########################################################################################
                
        ## burn initial forecasts since history is too short 
        nowcasts = nowcasts_to_burn[:,n_burn:]
        forecasts = nowcasts
        Vars = Vars_to_burn[:,n_burn:]
        
        ## compuate population moments
        forecasts_mean = np_mean(forecasts,axis=0)
        forecasts_var = np_var(forecasts,axis=0)
        FEs_mean = forecasts_mean - realized
        Vars_mean = np_mean(Vars,axis = 0) ## need to change 
        
        forecast_moments_sim = {"FE":FEs_mean,
                                "Disg":forecasts_var,
                                "Var":Vars_mean}
        return forecast_moments_sim


    def SMM(self):
        
        γ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = np.nan
        InfVar = np.nan
        InfATV = np.nan
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
        VarVar_sim = np.var(Vars_sim)
        VarATV_sim = np.cov(np.stack( (Vars_sim[1:],Vars_sim[:-1]),axis = 0))[0,1]
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim,
                      'VarVar':VarVar_sim,
                      'VarATV':VarATV_sim}
        return SMMMoments

# + code_folding=[0]
## intialize the ar instance 
dear0 = DiagnosticExpectationAR(exp_para = np.array([0.5,0.2]),
                                process_para = np.array([ρ0,σ0]),
                                real_time = real_time0,
                                history = history0,
                                horizon = 1)

dear0.GetRealization(realized0)

# + code_folding=[0]
## initial a sv instance
desv0 = DiagnosticExpectationSV(exp_para = np.array([0.3,0.2]),
                               process_para = np.array([0.1]),
                               real_time = xx_real_time,
                               history = xx_real_time) ## history does not matter here, 

## get the realization 

desv0.GetRealization(xx_realized)
# -

# #### Estimating DE using RE data

# + code_folding=[2, 9]
## only expectation estimation 

moments0 = ['FE',
            'FEVar',
            'FEATV',
            'Disg',
            'DisgVar',
            'Var']

def Objdear_re(paras):
    scalor = ObjGen(dear0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_re,
                    moment_choice = moments0,
                    how = 'expectation')
    return scalor


## invoke estimation 
#ParaEst(Objdear_re,
#        para_guess = np.array([0.2,0.3]),
#        method='Nelder-Mead')
# -

# #### Estimating DE using DE data 

# + code_folding=[0]
## get a fake data moment dictionary under a different parameter 

dear1 = DiagnosticExpectationAR(exp_para = np.array([0.3,0.1]),
                                process_para = np.array([ρ0,σ0]),
                                real_time = real_time0,
                                history = history0,
                                horizon = 1)

dear1.GetRealization(realized0)

data_mom_dict_de = dear1.SMM()


## only expectation estimation 

moments0 = ['FE',
            'FEVar',
            'FEATV',
            'Disg',
            'DisgVar',
            'Var']

def Objdear_de(paras):
    scalor = ObjGen(dear0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_de,
                    moment_choice = moments0,
                    how = 'expectation')
    return scalor


## invoke estimation 
#ParaEst(Objdear_de,
#        para_guess = np.array([0.2,0.3]),
#        method='Nelder-Mead')
# -

# #### Joint Estimation 

# + code_folding=[0, 2]
## for joint estimation 

moments1 = ['InfAV',
            'InfVar',
            'InfATV',
            'FE',
            'FEVar',
            'FEATV',
            'Disg']

def Objdear_joint(paras):
    scalor = ObjGen(dear0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_de,
                    moment_choice = moments1,
                    how ='joint',
                    n_exp_paras = 2)
    return scalor

## invoke estimation 
#ParaEst(Objdear_joint,
#        para_guess = np.array([0.2,0.3,0.8,0.2]),
#        method='Nelder-Mead')


# -

# ###  Sticky Expectation and Noisy Information Hybrid(SENI) + AR1

# + code_folding=[1, 14, 18]
@jitclass(model_data)
class SENIHybridAR:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
        
    def SimForecasts(self,
                     n_sim = 500):
        ## inputs 
        real_time = self.real_time
        history  = self.history
        realized = self.realized
        n = len(real_time)
        horizon = self.horizon 
        n_history =len(history)
        n_burn = len(history) - n
        horizon = self.horizon      

        ## parameters 
        ρ,σ = self.process_para
        lbd, sigma_pb,sigma_pr = self.exp_para
                
        #### The first NI part of the model 
        var_init = 5    ## some initial level of uncertainty, will be washed out after long simulation
        sigma_v = np.array([[sigma_pb**2,0.0],[0.0,sigma_pr**2]]) ## variance matrix of signal noises 
        
        ## simulate signals 
        nb_s = 2                                    ## the number of signals 
        H = np.array([[1.0],[1.0]])                 ## an multiplicative matrix summing all signals
        
        # randomly simulated signals 
        np.random.seed(12434)
        signal_pb = self.history+sigma_pb*np.random.randn(n_history)   ## one series of public signals 
        signals_pb = signal_pb.repeat(n_sim).reshape((-1,n_sim)).T     ## shared by all agents
        np.random.seed(13435)
        signals_pr = self.history + sigma_pr*np.random.randn(n_sim*n_history).reshape((n_sim,n_history))
                                                                 ### private signals are agent-specific 
            
        ## SE part of the model, which governs if updating for each agent at each point of time 
        ## simulation of updating profile
        ## simulation
        np.random.seed(12345)
        update_or_not_val = np.random.uniform(0,
                                              1,
                                              size = (n_sim,n_history))
        update_or_not_bool = update_or_not_val>=1-lbd
        update_or_not = update_or_not_bool.astype(np.int64)
        most_recent_when = np.empty((n_sim,n_history),dtype = np.int64)    
        ########################################################################
        nowsignals_pb_to_burn = np.empty((n_sim,n_history),dtype = np.float64)
        nowsignals_pr_to_burn = np.empty((n_sim,n_history),dtype=np.float64)
        ######################################################################
    
        # look back for the most recent last update for each point of time  
        for i in range(n_sim):
            for j in range(n_history):
                most_recent = j 
                for x in range(j):
                    if update_or_not[i,j-x]==1 and most_recent<=x:
                        most_recent = most_recent
                    elif update_or_not[i,j-x]==1 and most_recent>x:
                        most_recent = x
                most_recent_when[i,j] = most_recent
                ################################################################################
                nowsignals_pr_to_burn[i,j] = signal_pb[j - most_recent_when[i,j]]
                nowsignals_pr_to_burn[i,j] = signals_pr[i,j - most_recent_when[i,j]]
                ## both above are the matrices of signals available to each agent depending on if updating
                #####################################################################################
                
        
        ## The second NI part of the model 
        ## Once sticky signals are prepared, agents filter as NI
        
        ## prepare matricies 
        nowcasts_to_burn = np.zeros((n_sim,n_history))  ### nowcasts matrix of which the initial simulation is to be burned 
        nowcasts_to_burn[:,0] = history[0]
        nowvars_to_burn = np.zeros((n_sim,n_history))   ### nowcasts uncertainty matrix
        nowvars_to_burn[:,0] = var_init
        Vars_to_burn = np.zeros((n_sim,n_history))      ### forecasting uncertainty matrix 
        
        ## fill the matricies for individual moments  
        for i in range(n_sim):
            signals_this_i = np.concatenate((nowsignals_pb_to_burn[i,:],nowsignals_pr_to_burn[i,:]),axis=0).reshape((2,-1))
            ## the histories signals specific to i: the first row is public signals and the second is private signals 
            Pkalman = np.zeros((n_history,nb_s))
            ## Kalman gains of this agent for respective signals 
            Pkalman[0,:] = 0  ## some initial values 
            
            for t in range(n_history-1):
                step1_vars_to_burn = ρ**2*nowvars_to_burn[i,t] + σ**2
                ## priror uncertainty 
                
                inv = np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v) 
                ## the inverse of the noiseness matrix  
                
                inv_sc = np.dot(np.dot(H.T,inv),H)
                ## the total noisiness as a scalar 
                
                var_reduc = step1_vars_to_burn*inv_sc*step1_vars_to_burn
                ## reduction in uncertainty from the update
                
                nowvars_this_2d = np.array([[step1_vars_to_burn]]) - var_reduc
                ## update equation of nowcasting uncertainty 
                
                nowvars_to_burn[i,t+1] = nowvars_this_2d[0,0] 
                ## nowvars_this_2d is a 2-d matrix with only one entry. We take the element and set it to the matrix
                ### this is necessary for Numba typing 
                
                Pkalman[t+1,:] = step1_vars_to_burn*np.dot(H.T,np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v))
                ## update Kalman gains recursively using the signal extraction ratios 
                
                Pkalman_all = np.dot(Pkalman[t+1,:],H)[0] 
                ## the weight to the prior forecast 
                nowcasts_to_burn[i,t+1] = (1-Pkalman_all)*ρ*nowcasts_to_burn[i,t]+ np.dot(Pkalman[t+1,:],signals_this_i[:,t+1])
                ## kalman filtering updating for nowcasting: weighted average of prior and signals 
                
            for t in range(n_history):
                Vars_to_burn[i,t] = ρ**(2*horizon)*nowvars_to_burn[i,t] + hstepvar(horizon,ρ,σ)
        
        
        ## burn initial histories  
        nowcasts = nowcasts_to_burn[:,n_burn:]
        forecasts = ρ**horizon*nowcasts 
        Vars = Vars_to_burn[:,n_burn:]
        
        ## compuate population moments
        forecasts_mean = np_mean(forecasts,axis=0)
        forecasts_var = np_var(forecasts,axis=0)
        
        FEs_mean = forecasts_mean - realized
            
        Vars_mean = np_mean(Vars,axis=0) ## need to change for time-variant volatility
        
        forecast_moments_sim = {"FE":FEs_mean,
                                "Disg":forecasts_var,
                                "Var":Vars_mean}
        return forecast_moments_sim
        
    def SMM(self):
        
        ρ,σ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = 0.0
        InfVar = σ**2/(1-ρ**2)
        InfATV = ρ*InfVar
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
        VarVar_sim = np.var(Vars_sim)
        VarATV_sim = np.cov(np.stack( (Vars_sim[1:],Vars_sim[:-1]),axis = 0))[0,1]
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim,
                      'VarVar':VarVar_sim,
                      'VarATV':VarATV_sim}
        return SMMMoments


# -

# ###  Sticky Expectation and Noisy Information Hybrid(SENI) + SV
#
#

# + code_folding=[1, 2, 14, 18, 164, 198]
@jitclass(model_sv_data)
class SENIHybridSV:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
        
    def SimForecasts(self,
                     n_sim = 500):
        ## inputs 
        real_time = self.real_time
        history  = self.history
        realized = self.realized
        n = len(real_time[0,:])
        horizon = self.horizon
        n_history = len(history[0,:]) # of course equal to len(history)
        n_burn = n_history - n
        
        ## get the information set 
        infoset = history 
        y_now, p_now, sigmas_p_now, sigmas_t_now= infoset[0,:],infoset[1,:],infoset[2,:],infoset[3,:]
        sigmas_now = np.concatenate((sigmas_p_now,sigmas_t_now),axis=0).reshape((2,-1))
        
        
        ## process parameters
        γ = self.process_para
        ## exp parameters 
        lbd,sigma_pb,sigma_pr = self.exp_para
        var_init = 1
        
        ## other parameters 
        sigma_v = np.array([[sigma_pb**2,0.0],[0.0,sigma_pr**2]]) ## variance matrix of signal noises         
        ## simulate signals 
        nb_s = 2                                    ## the number of signals 
        H = np.array([[1.0],[1.0]])                 ## an multiplicative matrix summing all signals
        
        
        ## The first NI part of the model 
        # randomly simulated signals 
        np.random.seed(12434)
        ##########################################################
        signal_pb = p_now+sigma_pb*np.random.randn(n_history)   ## one series of public signals 
        signals_pb = signal_pb.repeat(n_sim).reshape((-1,n_sim)).T     ## shared by all agents
        np.random.seed(13435)
        signals_pr = p_now + sigma_pr*np.random.randn(n_sim*n_history).reshape((n_sim,n_history))
        ##################################################################################### 
            
        ## SE part of the model, which governs if updating for each agent at each point of time 
        ## simulation of updating profile
        ## simulation
        np.random.seed(12345)
        update_or_not_val = np.random.uniform(0,
                                              1,
                                              size = (n_sim,n_history))
        update_or_not_bool = update_or_not_val>=1-lbd
        update_or_not = update_or_not_bool.astype(np.int64)
        most_recent_when = np.empty((n_sim,n_history),dtype = np.int64)    
        ########################################################################
        nowsignals_pb_to_burn = np.empty((n_sim,n_history),dtype = np.float64)
        nowsignals_pr_to_burn = np.empty((n_sim,n_history),dtype=np.float64)
        ######################################################################
    
        # look back for the most recent last update for each point of time  
        for i in range(n_sim):
            for j in range(n_history):
                most_recent = j 
                for x in range(j):
                    if update_or_not[i,j-x]==1 and most_recent<=x:
                        most_recent = most_recent
                    elif update_or_not[i,j-x]==1 and most_recent>x:
                        most_recent = x
                most_recent_when[i,j] = most_recent
                ################################################################################
                nowsignals_pr_to_burn[i,j] = signal_pb[j - most_recent_when[i,j]]
                nowsignals_pr_to_burn[i,j] = signals_pr[i,j - most_recent_when[i,j]]
                ## both above are the matrices of signals available to each agent depending on if updating
                #####################################################################################
                
        
        ## The second NI part of the model 
        ## Once sticky signals are prepared, agents filter as NI
        
        ## prepare matricies 
        nowcasts_to_burn = np.zeros((n_sim,n_history))  ### nowcasts matrix of which the initial simulation is to be burned 
        nowcasts_to_burn[:,0] = p_now[0]
        nowvars_to_burn = np.zeros((n_sim,n_history))   ### nowcasts uncertainty matrix
        nowvars_to_burn[:,0] = var_init
        Vars_to_burn = np.zeros((n_sim,n_history))      ### forecasting uncertainty matrix 
        
        
        ## fill the matricies for individual moments        
        for i in range(n_sim):
            signals_this_i = np.concatenate((nowsignals_pr_to_burn[i,:],nowsignals_pr_to_burn[i,:]),axis=0).reshape((2,-1))
            ## the histories signals specific to i: the first row is public signals and the second is private signals 
            Pkalman = np.zeros((n_history,nb_s))
            ## Kalman gains of this agent for respective signals 
            Pkalman[0,:] = 0  ## some initial values 
            
            for t in range(n_history-1):
                step1var = hstepvarSV(1,
                                      sigmas_now[:,t],
                                      γ[0])
                step1_vars_to_burn = nowvars_to_burn[i,t] + step1var
                ## priror uncertainty 
                
                inv = np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v) 
                ## the inverse of the noiseness matrix  
                
                inv_sc = np.dot(np.dot(H.T,inv),H)
                ## the total noisiness as a scalar 
                
                var_reduc = step1_vars_to_burn*inv_sc*step1_vars_to_burn
                ## reduction in uncertainty from the update
                
                nowvars_this_2d = np.array([[step1_vars_to_burn]]) - var_reduc
                ## update equation of nowcasting uncertainty 
                
                nowvars_to_burn[i,t+1] = nowvars_this_2d[0,0] 
                ## nowvars_this_2d is a 2-d matrix with only one entry. We take the element and set it to the matrix
                ### this is necessary for Numba typing 
                
                Pkalman[t+1,:] = step1_vars_to_burn*np.dot(H.T,np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v))
                ## update Kalman gains recursively using the signal extraction ratios 
                
                Pkalman_all = np.dot(Pkalman[t+1,:],H)[0] 
                ## the weight to the prior forecast 
    
                nowcasts_to_burn[i,t+1] = (1-Pkalman_all)*nowcasts_to_burn[i,t]+ np.dot(Pkalman[t+1,:],signals_this_i[:,t+1])
                ## kalman filtering updating for nowcasting: weighted average of prior and signals 
                
            for t in range(n_history):
                stephvar = hstepvarSV(horizon,
                                      sigmas_now[:,t],
                                      γ[0])
                Vars_to_burn[i,t] = nowvars_to_burn[i,t] + stephvar
                
        nowcasts = nowcasts_to_burn[:,n_burn:]
        forecasts = nowcasts 
        Vars = Vars_to_burn[:,n_burn:]

        
        ## compuate population moments
        forecasts_mean = np_mean(forecasts,axis=0)
        forecasts_var = np_var(forecasts,axis=0)
        
        FEs_mean = forecasts_mean - realized            
        Vars_mean = np_mean(Vars,axis=0) ## need to change for time-variant volatility
        
        forecast_moments_sim = {"FE":FEs_mean,
                                "Disg":forecasts_var,
                                "Var":Vars_mean}
        return forecast_moments_sim
              
    def SMM(self):
        
        γ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = np.nan
        InfVar = np.nan
        InfATV = np.nan
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
        VarVar_sim = np.var(Vars_sim)
        VarATV_sim = np.cov(np.stack( (Vars_sim[1:],Vars_sim[:-1]),axis = 0))[0,1]
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim,
                      'VarVar':VarVar_sim,
                      'VarATV':VarATV_sim}
        return SMMMoments

# + code_folding=[0]
## intialize the ar instance 
seniar0 = SENIHybridAR(exp_para = np.array([0.3,0.3,0.2]),
                       process_para = np.array([ρ0,σ0]),
                       real_time = real_time0,
                       history = history0,
                       horizon = 1)

seniar0.GetRealization(realized0)

# + code_folding=[0]
## initial a sv instance
senisv0 = SENIHybridSV(exp_para = np.array([0.5,0.23,0.32]),
                               process_para = np.array([0.1]),
                               real_time = xx_real_time,
                               history = xx_real_time) ## history does not matter here, 


senisv0.GetRealization(xx_realized)
# -

# #### Estimate Hybrid using RE data 

# + code_folding=[0]
## only expectation estimation 

moments0 = ['FE',
            'FEVar',
            'FEATV',
            'Disg',
            'DisgVar',
            'DisgVar',
            'Var']

def Objseniar_re(paras):
    scalor = ObjGen(seniar0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_re,
                    moment_choice = moments0,
                    how = 'expectation')
    return scalor

## invoke estimation 
#ParaEst(Objseniar_re,
#        para_guess = np.array([0.5,0.5,0.5]),
#        method='Nelder-Mead')


# -

# #### Estimate Hybrid using Hybrid 
#
#

# + code_folding=[0, 2, 13]
## get a fake data moment dictionary under a different parameter 

seniar1 = SENIHybridAR(exp_para = np.array([0.2,0.4,0.5]),
                     process_para = np.array([ρ0,σ0]),
                     real_time = real_time0,
                     history = history0,
                     horizon = 1)

seniar1.GetRealization(realized0)

data_mom_dict_seni= seniar1.SMM()

def Objseniar_seni(paras):
    scalor = ObjGen(seniar0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_seni,
                    moment_choice = moments0,
                    how = 'expectation')
    return scalor

## invoke estimation 
#ParaEst(Objseniar_seni,
#        para_guess = np.array([0.5,0.5,0.5]),
#        method='Nelder-Mead')


# -

# #### Joint Estimation 

# + code_folding=[0, 2]
## for joint estimation 

moments1 = ['InfAV',
            'InfVar',
            'InfATV',
            'FE',
            'FEVar',
            'FEATV',
            'Disg',
           'DisgVar',
           'DisgATV',
           'Var']

def Objseniar_joint(paras):
    scalor = ObjGen(seniar0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_seni,
                    moment_choice = moments1,
                    how ='joint',
                    n_exp_paras = 3)
    return scalor

## invoek estimation 
#ParaEst(Objseniar_joint,
#        para_guess = np.array([0.4,0.2,0.3,0.8,0.2]),
#        method='Nelder-Mead')


# -

# ###  Diagnostic Expectation and Noisy Information Hybrid(DENI) + AR1

# + code_folding=[1, 117]
@jitclass(model_data)
class DENIHybridAR:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
        
    def SimForecasts(self,
                     n_sim = 500):
        ## inputs 
        
        real_time = self.real_time
        history = self.history
        realized = self.realized
        n = len(real_time)
        n_history = len(history)
        n_burn = len(history) - n
        
        ## parameters 
        ρ,σ = self.process_para
        theta,sigma_pb,sigma_pr = self.exp_para

        #######################
        ## using uncertainty at steady state of the Kalman filtering
        var_init = SteadyStateVar(self.process_para,
                                  self.exp_para[1:])    ## some initial level of uncertainty, will be washed out after long simulation
        ##################
        sigma_v = np.array([[sigma_pb**2,0.0],[0.0,sigma_pr**2]]) ## variance matrix of signal noises 
        horizon = self.horizon      
        
        ## simulate signals 
        nb_s = 2                                    ## the number of signals 
        H = np.array([[1.0],[1.0]])                 ## an multiplicative matrix summing all signals
        
        # randomly simulated signals 
        np.random.seed(12434)
        signal_pb = self.history+sigma_pb*np.random.randn(n_history)   ## one series of public signals 
        signals_pb = signal_pb.repeat(n_sim).reshape((-1,n_sim)).T     ## shared by all agents
        np.random.seed(13435)
        signals_pr = self.history + sigma_pr*np.random.randn(n_sim*n_history).reshape((n_sim,n_history))
                                                                 ### private signals are agent-specific 
    
        ## prepare matricies 
        nowcasts_to_burn = np.zeros((n_sim,n_history))  ### nowcasts matrix of which the initial simulation is to be burned 
        nowcasts_to_burn[:,0] = history[0]
        nowvars_to_burn = np.zeros((n_sim,n_history))   ### nowcasts uncertainty matrix
        nowvars_to_burn[:,0] = var_init
        Vars_to_burn = np.zeros((n_sim,n_history))      ### forecasting uncertainty matrix 
        
        
        ## fill the matricies for individual moments        
        for i in range(n_sim):
            signals_this_i = np.concatenate((signals_pb[i,:],signals_pr[i,:]),axis=0).reshape((2,-1))
            ## the histories signals specific to i: the first row is public signals and the second is private signals 
            Pkalman = np.zeros((n_history,nb_s))
            ## Kalman gains of this agent for respective signals 
            Pkalman[0,:] = 0  ## some initial values 
            
            for t in range(n_history-1):
                step1_vars_to_burn = ρ**2*nowvars_to_burn[i,t] + σ**2
                ## priror uncertainty 
                
                inv = np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v) 
                ## the inverse of the noiseness matrix  
                
                inv_sc = np.dot(np.dot(H.T,inv),H)
                ## the total noisiness as a scalar 
                
                var_reduc = step1_vars_to_burn*inv_sc*step1_vars_to_burn
                ## reduction in uncertainty from the update
                
                nowvars_this_2d = np.array([[step1_vars_to_burn]]) - var_reduc
                ## update equation of nowcasting uncertainty 
                
                nowvars_to_burn[i,t+1] = nowvars_this_2d[0,0] 
                ## nowvars_this_2d is a 2-d matrix with only one entry. We take the element and set it to the matrix
                ### this is necessary for Numba typing 
                
                Pkalman[t+1,:] = step1_vars_to_burn*np.dot(H.T,np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v))
                ## update Kalman gains recursively using the signal extraction ratios 
                
                Pkalman_all = np.dot(Pkalman[t+1,:],H)[0] 
                ## the weight to the prior forecast 
    
                nowcasts_to_burn[i,t+1] = (1-(1+theta)*Pkalman_all)*ρ*nowcasts_to_burn[i,t]+ (1+theta)*np.dot(Pkalman[t+1,:],signals_this_i[:,t+1])
                ## kalman filtering updating for nowcasting: weighted average of prior and signals 
                
            for t in range(n_history):
                Vars_to_burn[i,t] = ρ**(2*horizon)*nowvars_to_burn[i,t] + hstepvar(horizon,ρ,σ)
                
        nowcasts = nowcasts_to_burn[:,n_burn:]
        forecasts = ρ**horizon*nowcasts 
        Vars = Vars_to_burn[:,n_burn:]
        
        ## compuate population moments
        forecasts_mean = np_mean(forecasts,axis=0)
        forecasts_var = np_var(forecasts,axis=0)
        FEs_mean = forecasts_mean - realized
            
        Vars_mean = np_mean(Vars,axis=0) ## need to change for time-variant volatility
        
        forecast_moments_sim = {"FE":FEs_mean,
                                "Disg":forecasts_var,
                                "Var":Vars_mean}
        return forecast_moments_sim
    
    def SMM(self):
        
        ρ,σ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = 0.0
        InfVar = σ**2/(1-ρ**2)
        InfATV = ρ*InfVar
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
        VarVar_sim = np.var(Vars_sim)
        VarATV_sim = np.cov(np.stack( (Vars_sim[1:],Vars_sim[:-1]),axis = 0))[0,1]
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim,
                      'VarVar':VarVar_sim,
                      'VarATV':VarATV_sim}
        return SMMMoments


# -

# ###  Diagnostic Expectation and Noisy Information Hybrid(DENI) + SV
#
#

# + code_folding=[1, 2, 14, 18, 62]
@jitclass(model_sv_data)
class DENIHybridSV:
    def __init__(self,
                 exp_para,
                 process_para,
                 real_time,
                 history,
                 horizon = 1):
        self.exp_para = exp_para
        self.process_para = process_para
        self.horizon = horizon
        self.real_time = real_time
        self.history = history

    def GetRealization(self,
                       realized_series):
        self.realized = realized_series
              
    def SimForecasts(self,
                     n_sim = 500):
        ## inputs 
        real_time = self.real_time
        history  = self.history
        n = len(real_time[0,:])
        horizon = self.horizon
        n_history = len(history[0,:]) # of course equal to len(history)
        n_burn = n_history - n
        
        ## get the information set 
        infoset = history 
        y_now, p_now, sigmas_p_now, sigmas_t_now= infoset[0,:],infoset[1,:],infoset[2,:],infoset[3,:]
        sigmas_now = np.concatenate((sigmas_p_now,sigmas_t_now),axis=0).reshape((2,-1))
        
        ## process parameters
        γ = self.process_para
        ## exp parameters 
        theta, sigma_pb,sigma_pr = self.exp_para
        var_init = 2
        
        ## other parameters 
        sigma_v = np.array([[sigma_pb**2,0.0],[0.0,sigma_pr**2]]) ## variance matrix of signal noises         
        ## simulate signals 
        nb_s = 2                                    ## the number of signals 
        H = np.array([[1.0],[1.0]])                 ## an multiplicative matrix summing all signals
        
        # randomly simulated signals 
        np.random.seed(12434)
        ##########################################################
        signal_pb = p_now+sigma_pb*np.random.randn(n_history)   ## one series of public signals 
        signals_pb = signal_pb.repeat(n_sim).reshape((-1,n_sim)).T     ## shared by all agents
        np.random.seed(13435)
        signals_pr = p_now + sigma_pr*np.random.randn(n_sim*n_history).reshape((n_sim,n_history))
        #####################################################################################

        ## prepare matricies 
        nowcasts_to_burn = np.zeros((n_sim,n_history))
        nowcasts_to_burn[:,0] = p_now[0]
        nowvars_to_burn = np.zeros((n_sim,n_history))
        nowvars_to_burn[:,0] = var_init
        Vars_to_burn = np.zeros((n_sim,n_history))
        
        ## fill the matricies for individual moments        
        for i in range(n_sim):
            signals_this_i = np.concatenate((signals_pb[i,:],signals_pr[i,:]),axis=0).reshape((2,-1))
            ## the histories signals specific to i: the first row is public signals and the second is private signals 
            Pkalman = np.zeros((n_history,nb_s))
            ## Kalman gains of this agent for respective signals 
            Pkalman[0,:] = 0  ## some initial values 
            
            for t in range(n_history-1):
                step1var = hstepvarSV(1,
                                      sigmas_now[:,t],
                                      γ[0])
                step1_vars_to_burn = nowvars_to_burn[i,t] + step1var
                ## priror uncertainty 
                
                inv = np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v) 
                ## the inverse of the noiseness matrix  
                
                inv_sc = np.dot(np.dot(H.T,inv),H)
                ## the total noisiness as a scalar 
                
                var_reduc = step1_vars_to_burn*inv_sc*step1_vars_to_burn
                ## reduction in uncertainty from the update
                
                nowvars_this_2d = np.array([[step1_vars_to_burn]]) - var_reduc
                ## update equation of nowcasting uncertainty 
                
                nowvars_to_burn[i,t+1] = nowvars_this_2d[0,0] 
                ## nowvars_this_2d is a 2-d matrix with only one entry. We take the element and set it to the matrix
                ### this is necessary for Numba typing 
                
                Pkalman[t+1,:] = step1_vars_to_burn*np.dot(H.T,np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v))
                ## update Kalman gains recursively using the signal extraction ratios 
                
                Pkalman_all = np.dot(Pkalman[t+1,:],H)[0] 
                ## the weight to the prior forecast 
    
                nowcasts_to_burn[i,t+1] = (1-(1+theta)*Pkalman_all)*nowcasts_to_burn[i,t]+ np.dot((1+theta)*Pkalman[t+1,:],signals_this_i[:,t+1])
                ## kalman filtering updating for nowcasting: weighted average of prior and signals 
                
            for t in range(n_history):
                stephvar = hstepvarSV(horizon,
                                      sigmas_now[:,t],
                                      γ[0])
                Vars_to_burn[i,t] = nowvars_to_burn[i,t] + stephvar
                
        nowcasts = nowcasts_to_burn[:,n_burn:]
        forecasts = nowcasts 
        Vars = Vars_to_burn[:,n_burn:]

        
        ## compuate population moments
        forecasts_mean = np_mean(forecasts,axis=0)
        forecasts_var = np_var(forecasts,axis=0)
        FEs_mean = forecasts_mean - self.realized
        Vars_mean = np_mean(Vars,axis=0) ## need to change for time-variant volatility
        
        forecast_moments_sim = {"FE":FEs_mean,
                                "Disg":forecasts_var,
                                "Var":Vars_mean}
        return forecast_moments_sim
        
    def SMM(self):
        
        γ = self.process_para
        
        #################################
        # inflation moments 
        #################################

        InfAV  = np.nan
        InfVar = np.nan
        InfATV = np.nan
        
        #################################
        # expectation moments 
        #################################
        ## simulate forecasts
        moms_sim = self.SimForecasts()
        
        FEs_sim = moms_sim['FE']
        Disgs_sim = moms_sim['Disg']
        Vars_sim = moms_sim['Var']
        
        ## SMM moments     
        FE_sim = np.mean(FEs_sim)
        FEVar_sim = np.var(FEs_sim)
        FEATV_sim = np.cov(np.stack( (FEs_sim[1:],FEs_sim[:-1]),axis = 0 ))[0,1]
        Disg_sim = np.mean(Disgs_sim)
        DisgVar_sim = np.var(Disgs_sim)
        DisgATV_sim = np.cov(np.stack( (Disgs_sim[1:],Disgs_sim[:-1]),axis = 0))[0,1]
        
        Var_sim = np.mean(Vars_sim)
        VarVar_sim = np.var(Vars_sim)
        VarATV_sim = np.cov(np.stack( (Vars_sim[1:],Vars_sim[:-1]),axis = 0))[0,1]
    
        SMMMoments = {"InfAV":InfAV,
                      "InfVar":InfVar,
                      "InfATV":InfATV,
                      "FE":FE_sim,
                      "FEVar":FEVar_sim,
                      "FEATV":FEATV_sim,
                      "Disg":Disg_sim,
                      "DisgVar":DisgVar_sim,
                      "DisgATV":DisgATV_sim,
                      "Var":Var_sim,
                      'VarVar':VarVar_sim,
                      'VarATV':VarATV_sim}
        return SMMMoments

# + code_folding=[0]
## intialize the ar instance 
deniar0 = DENIHybridAR(exp_para = np.array([0.1,0.4,0.3]),
                       process_para = np.array([ρ0,σ0]),
                            real_time = real_time0,
                            history = history0,
                            horizon = 1)

deniar0.GetRealization(realized0)
# -

# #### Estimating DENI using RE

# + code_folding=[0, 9]
moments0 = ['FE',
            'FEVar',
            'FEATV',
            'Disg',
            'DisgVar',
            'DisgVar',
            'Var']

def Objdeniar_re(paras):
    scalor = ObjGen(deniar0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_re,
                    moment_choice = moments0,
                    how = 'expectation')
    return scalor

## invoke estimation 
#ParaEst(Objdeniar_re,
#        para_guess = np.array([0.5,0.5,0.5]),
#        method='trust-constr',
#       bounds = ((0,1),(0,3),(0,3),))


# -

# #### Estimating DENI using DENI

# +
data_mom_dict_deni = deniar0.SMM()

def Objdeniar_deni(paras):
    scalor = ObjGen(deniar0,
                    paras = paras,
                    data_mom_dict = data_mom_dict_deni,
                    moment_choice = moments0,
                    how = 'expectation')
    return scalor

## invoke estimation 
#ParaEst(Objdeniar_deni,
#        para_guess = np.array([0.5,0.5,0.5]),
#        method='trust-constr',
#        bounds = ((0,1),(0,3),(0,3),))


# + code_folding=[0]
## initial a sv instance
denisv0 = DENIHybridSV(exp_para = np.array([0.1,0.3,0.2]),
                           process_para = np.array([0.1]),
                           real_time = xx_real_time,
                           history = xx_real_time) ## history does not matter here, 

## get the realization 

denisv0.GetRealization(xx_realized)
# -

# ## Data Estimation  

# ### Prepare the data 

# #### Real-time Inflation data

# + code_folding=[]
import pandas as pd
real_time_index = pd.read_excel('../OtherData/RealTimeData/RealTimeInfQ.xlsx')

real_time_index.index = pd.to_datetime(real_time_index['DATE'],format='%Y%m%d')
real_time_index = real_time_index.drop(columns=['DATE'])

## turn index into yearly inflation
#real_time_index = pd.concat([real_time_cpic,real_time_cpi], join='inner', axis=1)
real_time_index.columns=['RTCPI','RTCPICore']
real_time_inf = real_time_index.pct_change(periods=12)*100
real_time_inf = real_time_inf.dropna()
# -

# #### Inflation data 

# +
###############
## monthly ### 
##############

InfM = pd.read_stata('../OtherData/InfM.dta')
InfM = InfM[-InfM.date.isnull()]
dateM = pd.to_datetime(InfM['date'],format='%Y%m%d')
dateM_str = dateM .dt.year.astype(int).astype(str) + \
             "M" + dateM .dt.month.astype(int).astype(str)
InfM.index = pd.DatetimeIndex(dateM,freq='infer')

###############
## quarterly ##
###############

InfQ = InfM.resample('Q').last()
dateQ = pd.to_datetime(InfQ['date'],format='%Y%m%d')

dateQ_str = dateQ.dt.year.astype(int).astype(str) + \
             "Q" + dateQ.dt.quarter.astype(int).astype(str)

InfQ.index = pd.DatetimeIndex(dateQ_str,freq='infer')


###########################
#keep only needed variables 
############################

InfM = InfM[['Inf1y_CPIAU',
            'Inf1yf_CPIAU']]

InfQ = InfQ[['Inf1y_CPICore',
            'Inf1yf_CPICore']]
# -

# #### Expectation data

# + code_folding=[]
## expectation data from SPF 

PopQ = pd.read_stata('../SurveyData/InfExpQ.dta')  
PopQ = PopQ[-PopQ.date.isnull()]
dateQ = pd.to_datetime(PopQ['date'],format='%Y%m%d')
dateQ_str = dateQ.dt.year.astype(int).astype(str) + \
             "Q" + dateQ.dt.quarter.astype(int).astype(str)
PopQ.index = pd.DatetimeIndex(dateQ_str)
SPFCPI = PopQ[['SPFCPI_Mean','SPFCPI_FE','SPFCPI_Disg','SPFCPI_Var']].dropna(how='any')

## expectation data from SCE
PopM = pd.read_stata('../SurveyData/InfExpM.dta')
PopM = PopM[-PopM.date.isnull()]
dateM = pd.to_datetime(PopM['date'],format='%Y%m%d')
dateM_str = dateM.dt.year.astype(int).astype(str) + \
             "M" + dateM.dt.month.astype(int).astype(str)
PopM.index = pd.DatetimeIndex(dateM)
SCECPI = PopM[['SCE_Mean','SCE_FE','SCE_Disg','SCE_Var',
              'SCE_Mean_rd','SCE_FE_rd','SCE_Disg_rd','SCE_Var_rd']].dropna(how='any')

# +
print('SCE:\n')
print(SCECPI.mean())

print('\n')

print('SPF:\n')
print(SPFCPI.mean())

# + code_folding=[]
## Combine expectation data and real-time data 

SPF_est = pd.concat([SPFCPI,
                    real_time_inf,
                    InfQ], join='inner', axis=1)
SCE_est = pd.concat([SCECPI,
                     real_time_inf,
                     InfM], join='inner', axis=1)
# -

# #### History data 

# + code_folding=[]
## process parameters estimation AR1 
# period filter 
start_t='1995-01-01'
end_t = '2022-06-30'   

######################
### quarterly data ##
#####################

CPICQ = InfQ['Inf1y_CPICore'].copy().loc[start_t:end_t]

###################
### monthly data ##
###################

CPIM = InfM['Inf1y_CPIAU'].copy().loc[start_t:end_t]

# + code_folding=[]
## hisotries data, the series ends at the same dates with real-time data but startes earlier 

st_t_history = '2000-01-01'
ed_t_SPF = SPF_est.index[-1].strftime('%Y%m%d')
ed_t_SCE = SCE_est.index[-1].strftime('%Y-%m-%d')

## get the quarterly index 
indexQ = CPICQ.index

## get history data quarterly and monthly respectively 
af = indexQ >= st_t_history 
bf = indexQ <=ed_t_SPF
time_filter = np.logical_and(af,bf)
time_filter_idx = indexQ[time_filter]

historyQ = real_time_inf.loc[time_filter_idx]
historyM = real_time_inf.loc[st_t_history:ed_t_SCE]
# -

# #### Realization data

# + code_folding=[]
## realized 1-year-ahead inflation

realized_CPIC = np.array(SPF_est['Inf1yf_CPICore']) 
realized_CPI = np.array(SCE_est['Inf1yf_CPIAU']) 
# -

# #### AR1 parameters 

# + code_folding=[]
######################
### quarterly data ##
#####################

CPICQ_demean = CPICQ

ARmodel = AR(CPICQ_demean,lags=1,trend='n')
ar_rs = ARmodel.fit()
rhoQ_est = ar_rs.params[0]
sigmaQ_est = np.sqrt(sum(ar_rs.resid**2)/(len(CPICQ)-1))

###################
### monthly data ##
###################

#Y = np.array(CPIM[12:])
#X = np.array(CPIM[:-12])

CPIM_demean = CPIM
ARmodel2 = AR(CPIM_demean,lags=1,trend='n')
ar_rs2 = ARmodel2.fit()
rhoM_est = ar_rs2.params[0]
sigmaM_est = np.sqrt(sum(ar_rs2.resid**2)/(len(CPIM)-1))

# + code_folding=[]
print('quarterly AR(1) estimates for CPI core:')
print(rhoQ_est)
print(sigmaQ_est)
print('monthly AR(1) estimates for CPI headline:')
print(rhoM_est)
print(sigmaM_est)
# -

# #### Data moments 

# + code_folding=[]
#####################################
## preparing data moments
#####################################

## Be careful with frequency here when computing auto-correlation. 
### SPF: quarters 
### SCE: month

#####################
## Professionals ####
#####################

### inflation moments 

realized_CPIC = realized_CPIC[~np.isnan(realized_CPIC)]
InfAV_data = np.mean(realized_CPIC)
InfVar_data = np.var(realized_CPIC)
InfATV_data = np.cov(np.stack( (realized_CPIC[4:],realized_CPIC[:-4]),axis = 0 ))[0,1]
## annual autocovariance

### expectation moments 
exp_data_SPF = SPF_est[['SPFCPI_Mean','SPFCPI_FE','SPFCPI_Disg','SPFCPI_Var']]
exp_data_SPF = exp_data_SPF.rename(columns={"SPFCPI_Mean": "Forecast", "SPFCPI_FE": "FE",
                            "SPFCPI_Disg":"Disg","SPFCPI_Var":"Var"})

FEs_data = exp_data_SPF['FE']
Disgs_data = exp_data_SPF['Disg']
Vars_data = exp_data_SPF['Var']

FE_data = np.mean(FEs_data)
FEVar_data = np.var(FEs_data)
FEATV_data = np.cov(np.stack( (FEs_data[4:],FEs_data[:-4]),axis = 0))[0,1]  ## 4 quarters apart 
## annual autocovariance


Disg_data = np.mean(Disgs_data)
DisgVar_data = np.var(Disgs_data)
DisgATV_data = np.cov(np.stack( (Disgs_data[4:],Disgs_data[:-4]),axis = 0))[0,1]
Var_data = np.mean(Vars_data)
VarVar_data = np.var(Vars_data)
VarATV_data = np.cov(np.stack( (Vars_data[4:],Vars_data[:-4]),axis = 0))[0,1]
## annual autocovariance



data_moms_dct_SPF = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.float64,
)
data_moms_dct_SPF['InfAV'] = InfAV_data
data_moms_dct_SPF['InfVar'] = InfVar_data
data_moms_dct_SPF['InfATV'] = InfATV_data

data_moms_dct_SPF['FE'] = FE_data
data_moms_dct_SPF['FEVar'] = FEVar_data
data_moms_dct_SPF['FEATV'] = FEATV_data
data_moms_dct_SPF['Disg'] = Disg_data
data_moms_dct_SPF['DisgVar'] = DisgVar_data
data_moms_dct_SPF['DisgATV'] = DisgATV_data
data_moms_dct_SPF['Var'] = Var_data
data_moms_dct_SPF['VarVar'] = VarVar_data
data_moms_dct_SPF['VarATV'] = VarATV_data


##########################
### For households 
##########################

#############################################################################################
############!!!!! using "xx_rd" moments only if want to control for individual fixed effects
###########################################################################################

exp_data_SCE = SCE_est[['SCE_Mean','SCE_FE_rd','SCE_Disg_rd','SCE_Var_rd']]
exp_data_SCE = exp_data_SCE.rename(columns={"SCE_Mean": "Forecast", "SCE_FE_rd": "FE",
                                            "SCE_Disg_rd":"Disg","SCE_Var_rd":"Var"})


## inflation moments 

realized_CPI = realized_CPI[~np.isnan(realized_CPI)]

InfAV_data = np.mean(realized_CPI)
InfVar_data = np.var(realized_CPI)
InfATV_data = np.cov(np.stack( (realized_CPI[12:],realized_CPI[:-12]),axis = 0 ))[0,1]

## expectation moments 
FEs_data = exp_data_SCE['FE']
Disgs_data = exp_data_SCE['Disg']
Vars_data = exp_data_SCE['Var']

FE_data = np.mean(FEs_data)
FEVar_data = np.var(FEs_data)
FEATV_data = np.cov(np.stack( (FEs_data[12:],FEs_data[:-12]),axis = 0 ))[0,1]

Disg_data = np.mean(Disgs_data)
DisgVar_data = np.var(Disgs_data)
DisgATV_data = np.cov(np.stack( (Disgs_data[12:],Disgs_data[:-12]),axis = 0))[0,1]
Var_data = np.mean(Vars_data)
VarVar_data = np.var(Vars_data)
VarATV_data = np.cov(np.stack( (Vars_data[12:],Vars_data[:-12]),axis = 0))[0,1]


data_moms_dct_SCE = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.float64,
)
data_moms_dct_SCE['InfAV'] = InfAV_data
data_moms_dct_SCE['InfVar'] = InfVar_data
data_moms_dct_SCE['InfATV'] = InfATV_data
data_moms_dct_SCE['FE'] = FE_data
data_moms_dct_SCE['FEVar'] = FEVar_data
data_moms_dct_SCE['FEATV'] = FEATV_data
data_moms_dct_SCE['Disg'] = Disg_data
data_moms_dct_SCE['DisgVar'] = DisgVar_data
data_moms_dct_SCE['DisgATV'] = DisgATV_data
data_moms_dct_SCE['Var'] = Var_data
data_moms_dct_SCE['VarVar'] = VarVar_data
data_moms_dct_SCE['VarATV'] = VarATV_data


############# need to compute the unconditional moments here 
# -

print(dict(data_moms_dct_SPF))
print(dict(data_moms_dct_SCE))

# ### Data moments

# + code_folding=[]
## real time and history 

################
## quarterly ###
#################

real_time_Q_ar = np.array(SPF_est['RTCPICore'])
history_Q_ar = np.array(historyQ['RTCPICore'])
process_paraQ_est_ar = np.array([rhoQ_est,sigmaQ_est])

##############
## monthly ###
#############

real_time_M_ar = np.array(SCE_est['RTCPI'])
history_M_ar = np.array(historyM['RTCPI'])

process_paraM_est_ar = np.array([rhoM_est,
                                 sigmaM_est])

# + [markdown] code_folding=[]
# #### SV  parameters and data  

# + code_folding=[0]
################
## quarterly ##
################

### quarterly data 
### exporting inflation series for process estimation using UCSV model in matlab

CPICQ.to_excel("../OtherData/CPICQ.xlsx")  ## this is for matlab estimation of UCSV model

################
## monthly ####
################

### process parameters estimation 

### exporting monthly inflation series for process estimation using UCSV model in matlab
CPIM.to_excel("../OtherData/CPIM.xlsx")  ## this is for matlab estimation of UCSV model

##########################################################################
##########################################################################
##########################################################################
## use matlab code stockwatson.m to estimate UCSV model before moving on!
##########################################################################
##########################################################################
##########################################################################
##########################################################################


## process parameters estimation for SV 

################
## quarterly ##
################

### import the estimated results 
CPICQ_UCSV_Est = pd.read_excel('../OtherData/UCSVestQ.xlsx')  
CPICQ_UCSV_Est.index = pd.to_datetime(CPICQ_UCSV_Est['date'],format='%Y%m%d')
CPICQ_UCSV_Est = CPICQ_UCSV_Est.drop(columns=['date'])
CPICQ_UCSV_Est = CPICQ_UCSV_Est.rename(columns = {'sd_eps':'sd_p_est',
                                      'sd_eta':'sd_t_est',
                                      'tau':'p'})  ## Loading ucsv model estimates 



################
## monthly ####
################

### import the estimated results 
CPIM_UCSV_Est = pd.read_excel('../OtherData/UCSVestM.xlsx')  
CPIM_UCSV_Est.index =pd.to_datetime(CPIM_UCSV_Est['date'],format='%Y%m%d')
CPIM_UCSV_Est = CPIM_UCSV_Est.drop(columns=['date'])
CPIM_UCSV_Est = CPIM_UCSV_Est.rename(columns = {'sd_eps':'sd_p_est',
                                                'sd_eta':'sd_t_est',
                                                'tau':'p'})  ## Loading ucsv model estimates 

########################################################################################
## be careful with the order, I define eta as the permanent and eps to be the tansitory 
 ######################################################################################
    

### quarterly plot 

plt.figure(figsize=(10,5))
plt.plot(CPICQ_UCSV_Est['sd_p_est'],
         'r--',
          lw = lw,
         label='permanent volitility')
plt.plot(CPICQ_UCSV_Est['sd_t_est'],
         'k-.',
         lw = lw,
         label='transitory volitility')
plt.title('Estimated Stochastic Volatility of Core CPI Inflation')
plt.ylabel('std')
plt.legend(loc=1)
plt.savefig('../graphs/inflation/UCSVQ.png')
    
### monthly plot 

plt.figure(figsize=(10,5))
plt.plot(CPIM_UCSV_Est['sd_p_est'],
         'r--',
          lw = lw,
         label='permanent volitility')
plt.plot(CPIM_UCSV_Est['sd_t_est'],
         'k-.',
         lw = lw,
         label='transitory volitility')
plt.title('Estimated Stochastic Volatility of CPI Inflation')
plt.ylabel('std')
plt.legend(loc=1)
plt.savefig('../graphs/inflation/UCSVM.png')

# + code_folding=[0]
#########################################################
## specific to SV model  
######################################################

#############
## quarterly 
##############

n_burn_rt_historyQ = len(CPICQ_UCSV_Est) - len(historyQ)  

history_yQ = np.array(historyQ['RTCPICore'])
history_vol_pQ = np.array(CPICQ_UCSV_Est['sd_p_est'][n_burn_rt_historyQ:])**2  ## permanent
history_vol_tQ = np.array(CPICQ_UCSV_Est['sd_t_est'][n_burn_rt_historyQ:])**2 ## transitory

#history_volsQ = np.array([history_vol_pQ,
#                          history_vol_tQ])
history_pQ = np.array(CPICQ_UCSV_Est['p'][n_burn_rt_historyQ:])

## to burn 
n_burn_Q = len(history_pQ) - len(SPF_est['RTCPI'])
real_time_yQ = history_yQ[n_burn_Q:]
real_time_vol_pQ = history_vol_pQ[n_burn_Q:]
real_time_vol_tQ = history_vol_tQ[n_burn_Q:]
real_time_pQ = history_pQ[n_burn_Q:]

############
## monthly
############

n_burn_rt_historyM = len(CPIM_UCSV_Est) - len(historyM)  

history_yM = np.array(historyM['RTCPI'])
history_vol_pM = np.array(CPIM_UCSV_Est['sd_p_est'][n_burn_rt_historyM:])**2
history_vol_tM = np.array(CPIM_UCSV_Est['sd_t_est'][n_burn_rt_historyM:])**2

#history_volsM = np.array([history_vol_pM,
#                          history_vol_tM])  ## order is import 

history_pM = np.array(CPIM_UCSV_Est['p'][n_burn_rt_historyM:])

## to burn 
n_burn_M = len(history_pM) - len(SCE_est['RTCPI'])
real_time_yM = history_yM[n_burn_M:]
real_time_vol_pM = history_vol_pM[n_burn_M:]
real_time_vol_tM = history_vol_tM[n_burn_M:]
real_time_pM = history_pM[n_burn_M:]

# + code_folding=[0]
## generate histories and real time array 

history_Q_sv = np.array([history_yQ,
                           history_pQ,
                           history_vol_pQ,
                           history_vol_tQ])
history_M_sv = np.array([history_yM,
                           history_pM,
                           history_vol_pM,
                           history_vol_tM])

real_time_Q_sv = np.array([real_time_yQ,
                         real_time_pQ,
                         real_time_vol_pQ,
                         real_time_vol_tQ])
real_time_M_sv = np.array([real_time_yM,
                         real_time_pM,
                         real_time_vol_pM,
                         real_time_vol_tM])

# + code_folding=[]
## process parameters 
process_paraQ_est_sv = np.array([0.2])
process_paraM_est_sv = np.array([0.2])
# -

# ### Test  Estimation

# ### Estimation 

# + code_folding=[4, 11, 23, 33, 51, 60, 63, 68, 71, 76, 82, 88, 99, 108, 116, 126, 136, 147, 151, 156, 161, 167, 186, 229, 287]
agents_list = ['SPF','SCE']

process_list = ['AR','SV']

ex_model_list = ['SE',
                 'NI',
                 'DE',
                 'DENI'
                ]
nb_ex_model = len(ex_model_list)

moments_list = [['FE','FEVar','FEATV'],
               ['FE','FEVar','FEATV','Disg','DisgVar','DisgATV'],
               ['FE','FEVar','FEATV','Disg','DisgVar','DisgATV','Var','VarVar','VarATV']]
nb_moments = len(moments_list)

how_list =['2-step','Joint']

moments_list_general = ['FE','FE+Disg','FE+Disg+Var']

model_list = [sear0,niar0,dear0,deniar0,
             sesv0,nisv0,desv0,denisv0]

algorithm_list = ['trust-constr',
                 'trust-constr',
                 'trust-constr',
                 'trust-constr',
                 'trust-constr',
                 'trust-constr',
                 'trust-constr',
                 'trust-constr']


algorithm_joint_list = ['trust-constr',
                        'trust-constr',
                        'trust-constr',
                        'trust-constr',
                        None,
                        None,
                        None,
                        None]

bns_list =[((0,1),),
           ((0,3),(0,3),),
           ((-2,2),(0,5),),
           ((-3,3),(0,3),(0,3),),
           ((0,1),),
           ((0,3),(0,3),),
           ((-2,2),(0,np.inf),),
           ((-3,3),(0,3),(0,3),)]

bns_joint_list =[((0,1),(0.9,1),(0,np.inf),),
                 ((0,3),(0,3),(0.9,1),(0,np.inf),),
                 ((-2,2),(0,5),(0.9,1),(0,np.inf),),
                 ((-3,3),(0,3),(0,3),(0.9,1),(0,np.inf),),
                 None,
                 None,
                 None,
                 None]

data_mom_dict_list = [data_moms_dct_SPF,
                      data_moms_dct_SCE]

process_paras_list = [process_paraQ_est_ar,
                      process_paraQ_est_sv,
                      process_paraM_est_ar,
                      process_paraM_est_sv]

realized_list = [realized_CPIC.astype(np.float64),
                 realized_CPI.astype(np.float64)]

real_time_list = [np.array(real_time_Q_ar),
                 np.array(real_time_Q_sv),  ## 4 x t array 
                np.array(real_time_M_ar),
                 np.array(real_time_M_sv)]  ## 4 x t array 

history_list = [np.array(history_Q_ar),
               np.array(history_Q_sv),     ## 4 x t array 
               np.array(history_M_ar), 
               np.array(history_M_sv)]     ## 4 x t array 

## parameter guesses 
guesses_list = [np.array([0.3]),  ## se lbd 
               np.array([0.5,0.8]),  ## ni sigma_pb, sigma_pr
               np.array([0.3,0.4]),   ## de theta theta_sigma
               np.array([0.1,0.2,0.3])
               ]  ## deni theta, sigma_pb, sigma_pr

guesses_joint_list = [np.array([0.3,0.97,0.1]),            ## se lbd 
                       np.array([0.1,0.2,0.95,0.1]),      ## ni sigma_pb, sigma_pr
                       np.array([0.3,0.4,0.95,0.1]),      ## de theta theta_sigma
                       np.array([0.1,0.2,0.3,0.95,0.1]),  ## theta, sigma_pb, sigma_pr  
                       ## for sv models not used
                       np.array([0.3,0.2]),            ## se lbd 
                       np.array([0.1,0.2,0.2]),      ## ni sigma_pb, sigma_pr
                       np.array([0.3,0.4,0.2]),      ## de theta theta_sigma
                       np.array([0.1,0.2,0.2])
                     ]  ## deni theta, sigma_pb, sigma_pr]

n_exp_paras_list = [1,
                    2,
                    2,
                    3]


## names labels 


se_ar_names = [r'$\hat\lambda$',
                   r'$\rho$',
                   r'$\sigma$',
                   r'$\hat\lambda$', 
                   r'$\rho$',
                   r'$\sigma$']


ni_ar_names = [r'$\hat\sigma_{pb}$',
                   r'$\hat\sigma_{pr}$',
                   r'$\rho$',
                   r'$\sigma$',
                   r'$\hat\sigma_{pb}$',
                   r'$\hat\sigma_{pr}$',
                   r'NI: $\rho$',
                   r'NI: $\sigma$']


de_ar_names = [r'$\hat\theta$',
                   r'$\sigma_\theta$',
                   r'$\rho$',
                   r'$\sigma$',
                   r'$\hat\theta$',
                   r'$\sigma_\theta$',
                   r'$\rho$',
                   r'$\sigma$']


deni_ar_names = [r'$\hat\theta$',
                   r'$\hat\sigma_{pb}$',
                   r'$\hat\sigma_{pr}$',
                   r'$\rho$',
                   r'$\sigma$',
                   r'$\hat\theta$',
                   r'$\hat\sigma_{pb}$',
                   r'$\hat\sigma_{pr}$',
                   r'$\rho$',
                   r'$\sigma$']

se_sv_names = [r'$\hat\lambda$',
               r'$\gamma$']


ni_sv_names = [r'$\hat\sigma_{pb}$',
               r'$\hat\sigma_{pr}$',
               r'$\gamma$']


de_sv_names = [r'$\hat\theta$',
               r'$\sigma_\theta$',                   
               r'$\gamma$']


deni_sv_names = [r'$\hat\theta$',
                   r'$\hat\sigma_{pb}$',
                   r'$\hat\sigma_{pr}$',
                   r'$\gamma$']


names_list = [se_ar_names,
             ni_ar_names,
             de_ar_names,
             deni_ar_names,
             se_sv_names,
             ni_sv_names,
             de_sv_names,
             deni_sv_names]

################################################################################
## estimate the model for different agents, theory, inflation process and joint/theory 
#################################################################################

paras_list = []
paras_step2_list = []
paras_joint_list = []
paras_joint_step2_list = []


for agent_id,agent in enumerate(agents_list):
    print(agent)
    realized_this = realized_list[agent_id]
    data_mom_dict_this = data_mom_dict_list[agent_id]
    for pg_id,process in enumerate(process_list):
        print(process)
        ## history and real-time inflation that is fed in the model depends on agent type and process
        agent_process_id = agent_id*2+pg_id       
        process_paras_this = process_paras_list[agent_process_id]
        real_time_this = real_time_list[agent_process_id]
        history_this = history_list[agent_process_id] 
        
        for exp_id,ex_model in enumerate(ex_model_list):
            print(ex_model)
            model_idx  = pg_id*nb_ex_model+exp_id
            print(model_idx)
            model_instance = model_list[model_idx]
            alg_this= algorithm_list[model_idx]
            bounds_this = bns_list[model_idx]
            print('2-step estimation uses algorithm'+alg_this)
            print('bounds for parameters'+str(bounds_this))
            alg_joint_this= algorithm_joint_list[model_idx]
            bounds_joint_this = bns_joint_list[model_idx]
            names_this = names_list[model_idx]

            ## feed inputs to the instance 
            instance = model_instance

            print(instance)
            instance.GetRealization(realized_this)
            instance.real_time = real_time_this
            instance.history = history_this
            instance.process_para = process_paras_this 
            
            ## model-specific estimates holder 
            
            paras_list_this_model = []
            paras_step2_list_this_model = []
            paras_joint_list_this_model = []
            paras_joint_step2_list_this_model = []
            
            ## specific objetive function to minimize (only for expectation)

            for mom_id, moments_this in enumerate(moments_list):
                print(moments_this)
                print('Step 1')
                def Obj_this(paras_this):  
                    scalor = ObjGen(instance,
                                    paras = paras_this,
                                    data_mom_dict = data_mom_dict_this,
                                    moment_choice = moments_this,
                                    how ='expectation')
                    return scalor
                
                guess_this = guesses_list[exp_id]
                ## estimating
                para_est  = ParaEst(Obj_this,
                                 para_guess = guess_this,
                                 method= alg_this,
                                bounds = bounds_this)
                ## same para est
                para_est = np.round(para_est,2)
                print('Step 1:'+str(para_est))
                paras_list_this_model.append(para_est)
                paras_list.append(para_est)
                if len(para_est)>0:
                    ## compute the efficient weighting matrix 
                    instance.exp_para = para_est
                    smm_dict_this = instance.SMM()
                    distance = np.array([smm_dict_this[mom] - data_mom_dict_this[mom] for mom in moments_this]) 
                    distance_diag = np.diag(distance*distance.T)
                    wm1st = np.linalg.inv(distance_diag)

                    ## 2-step estimation using efficient matrix 
                    def Obj_this_step2(paras_this):
                        scalor = ObjWeight(instance,
                                           paras = paras_this,
                                           weight = wm1st,
                                           data_mom_dict = data_mom_dict_this,
                                           moment_choice = moments_this,
                                           how ='expectation')
                        return scalor
                    ## re-estimating
                    para_est_step2  = ParaEst(Obj_this_step2,
                                        para_guess = guess_this,
                                        method= alg_this,
                                        bounds = bounds_this)
                else:
                    para_est_step2 = np.array([])
                ## same para est
                para_est_step2 = np.round(para_est_step2,2)

                print('Step 2:'+str(para_est_step2))
                paras_step2_list_this_model.append(para_est_step2)
                paras_step2_list.append(para_est_step2)
                
                
                ##  joint estimation 
                n_exp_paras_this = n_exp_paras_list[exp_id]
                moments_this_ = moments_this+['InfAV','InfVar','InfATV'] ## added inflation moments 
                
                if pg_id <=0: ## no joint estimation for SV models 
                    def Obj_joint_this(paras):
                        scalor = ObjGen(instance,
                                        paras = paras,
                                        data_mom_dict = data_mom_dict_this,
                                        moment_choice = moments_this_,
                                        how = 'joint',
                                        n_exp_paras = n_exp_paras_this)
                        return scalor

                    guess_this_ = guesses_joint_list[model_idx]
                    ## estimating
                    paras_joint_est = ParaEst(Obj_joint_this,
                                              para_guess = guess_this_,
                                              method= alg_joint_this,
                                              bounds = bounds_joint_this) ##Nelder-Mead  
                    if len(paras_joint_est)>0:
                        ## compute the efficient weighting matrix 
                        instance.exp_para = paras_joint_est[0:n_exp_paras_this]     
                        instance.process_para = paras_joint_est[n_exp_paras_this:]
                        smm_dict_this = instance.SMM()
                        distance = np.array([smm_dict_this[mom] - data_mom_dict_this[mom] for mom in moments_this_]) 
                        distance_diag = np.diag(distance*distance.T)
                        wm1st = np.linalg.inv(distance_diag)

                        ## 2-step estimation using efficient matrix 
                        def Obj_joint_this_step2(paras_this):
                            scalor = ObjWeight(instance,
                                               paras = paras_this,
                                               weight = wm1st,
                                               data_mom_dict = data_mom_dict_this,
                                               moment_choice = moments_this_,
                                               how ='joint',
                                               n_exp_paras = n_exp_paras_this)
                            return scalor
                        ## re-estimating
                        para_est_joint_step2  = ParaEst(Obj_joint_this_step2,
                                                        para_guess = guess_this_,
                                                        method = alg_joint_this,
                                                        bounds = bounds_joint_this)     #  Nelder-Mead         
                    else:
                        para_est_joint_step2 = np.array([])
                
                else:
                    paras_joint_est = np.array([])
                    para_est_joint_step2 = np.array([])
                
                ## save in the list 
                paras_joint_est = np.round(paras_joint_est,2)
                print('Step 1:'+str(paras_joint_est))
                paras_joint_list_this_model.append(paras_joint_est)
                paras_joint_list.append(paras_joint_est)
                
                para_est_joint_step2 = np.round(para_est_joint_step2,2)
                print('Step 2:'+str(para_est_joint_step2))
                paras_joint_step2_list_this_model.append(para_est_joint_step2)
                paras_joint_step2_list.append(para_est_joint_step2)
            
            
            ### export model-specific estimates 
            names = names_list[model_idx]
            #print(paras_step2_list_this_model)
            #print(paras_joint_step2_list_this_model)
            para_est_tab_this_model = pd.DataFrame(paras_step2_list_this_model,
                                            index = moments_list_general)
            para_est_process_tab_this_model = pd.DataFrame([process_paras_this]*nb_moments,
                                            index = moments_list_general)
            para_joint_est_tab_this_model = pd.DataFrame(paras_joint_step2_list_this_model,
                                            index = moments_list_general)
            para_all_est_tab_this_model = pd.concat([para_est_tab_this_model,
                                                     para_est_process_tab_this_model,
                                                     para_joint_est_tab_this_model],
                                                    join = 'inner', axis=1)
            print(names_this)
            try:
                para_all_est_tab_this_model.columns = names_this 
            except:
                pass
            para_all_est_tab_this_model.to_excel('tables/'+agent+'_'+process+'_'+ex_model+'.xlsx',
                                       float_format='%.2f',
                                       index = True)


# + code_folding=[]
## create multiple index to store coefficients estimates 

iterables = [agents_list, process_list, ex_model_list,moments_list_general]
midx = pd.MultiIndex.from_product(iterables, names=['Agents', 'Process','Model','Moments'])
paras_table = pd.DataFrame(index = midx)

## 2-step table
paras_list = [tuple(paras) for paras in paras_list]
paras_table['ParaEst'] = paras_list

paras_step2_list = [tuple(paras) for paras in paras_step2_list]
paras_table['ParaEst2step'] = paras_step2_list


## joint table
paras_joint_table = pd.DataFrame(index = midx)

paras_joint_list = [tuple(paras) for paras in paras_joint_list]
paras_joint_table['ParaEst'] = paras_joint_list

paras_joint_step2_list = [tuple(paras) for paras in paras_joint_step2_list]
paras_joint_table['ParaEst2step'] = paras_joint_step2_list

# -

paras_table.columns = [['2-step Estimate','2-step Estimate 2nd step']]
paras_joint_table.columns = [['Joint Estimate','Joint Estimate 2nd step']]

# + code_folding=[0]
paras_combine_table = pd.merge(paras_table,
                               paras_joint_table,
                               how='outer',
                               left_index=True,
                               right_index=True)

# + code_folding=[]
## Flag those under-identified cases 

ui_list = [('SPF','AR','NI','FE'),
          ('SCE','AR','NI','FE'),
          ('SPF','AR','DENI','FE'),
          ('SCE','AR','DENI','FE'),
          ('SPF','SV','NI','FE'),
          ('SCE','SV','NI','FE'),
          ('SPF','SV','DENI','FE'),
          ('SCE','SV','DENI','FE')
          ]


for case in ui_list:
    for name in paras_combine_table.columns:
        paras_combine_table.loc[case,name]=tuple(np.array([]))
        #paras_combine_table.loc[case][name] = tuple(np.array([]))
# -

print(process_paraQ_est_ar)
print(process_paraM_est_ar)

paras_combine_table

# +
## only 2nd-step estimates 

paras_combine_table_2step = paras_combine_table[['2-step Estimate 2nd step','Joint Estimate 2nd step']]

paras_combine_table_2step

# + code_folding=[4]
## generate model moments

smm_list = []
smm_joint_list = []
for agent_id,agent in enumerate(agents_list):
    print(agent)
    realized_this = realized_list[agent_id]
    data_mom_dict_this = data_mom_dict_list[agent_id]
    for pg_id,process in enumerate(process_list):
        print(process)
        process_paras_this = process_paras_list[agent_process_id]
        ## history and real-time inflation that is fed in the model depends on agent type and process
        agent_process_id = agent_id*2+pg_id       
        process_paras_this = process_paras_list[agent_process_id]
        real_time_this = real_time_list[agent_process_id]
        history_this = history_list[agent_process_id] 
        
        for exp_id,ex_model in enumerate(ex_model_list):
            n_exp_model = len(ex_model_list)
            print(ex_model)
            model_idx  = pg_id*n_exp_model+exp_id
            #print(model_idx)
            model_instance = model_list[model_idx]

            ## feed inputs to the instance 
            instance = model_instance
            print(instance)
            instance.GetRealization(realized_this)
            instance.real_time = real_time_this
            instance.history = history_this
            instance.process_para = process_paras_this 
            
            ## specific objetive function to minimize (only for expectation)
            for moments in moments_list_general:
                ## 2-step moments 
                para_est_this = np.array(list(paras_combine_table.loc[agent,process,ex_model,moments]['2-step Estimate 2nd step'])).flatten()
                print(para_est_this)
                try:
                    instance.process_para = process_paras_this
                    instance.exp_para = para_est_this
                    smm_this = instance.SMM()
                except:
                    smm_this = {}
                smm_list.append(smm_this)

                ## joint moments
                n_exp_paras_this = n_exp_paras_list[exp_id]
                para_est_this_ = np.array(list(paras_combine_table.loc[agent,process,ex_model,moments]['Joint Estimate 2nd step'])).flatten()
                print(para_est_this_)
                try:
                    instance.exp_para = para_est_this_[0:n_exp_paras_this]     
                    instance.process_para = para_est_this_[n_exp_paras_this:]
                    try:
                        smm_this = instance.SMM()
                        smm_joint_list.append(smm_this)
                    except:
                        smm_this = {}
                        smm_joint_list.append(smm_this)
                except:
                    smm_this = {}
                    smm_joint_list.append(smm_this)

# + code_folding=[]
## model moments 
smm_model = pd.DataFrame(smm_list,
                         columns = list(smm_list[0].keys()),
                         index = midx)

smm_joint_model = pd.DataFrame(smm_joint_list,
                               columns = list(smm_joint_list[1].keys()),
                               index = midx)

# + code_folding=[]
## data moments

smm_data_spf =  pd.DataFrame(data_moms_dct_SPF.values()).T
smm_data_spf.columns = data_moms_dct_SPF.keys()

smm_data_sce =  pd.DataFrame(data_moms_dct_SCE.values()).T
smm_data_sce.columns = data_moms_dct_SCE.keys()

# +
mom_compare_spf = smm_data_spf.append(smm_model.loc['SPF'])
mom_compare_sce = smm_data_sce.append(smm_model.loc['SCE'])

mom_joint_compare_spf = smm_data_spf.append(smm_joint_model.loc['SPF','AR'])
mom_joint_compare_sce = smm_data_sce.append(smm_joint_model.loc['SCE','AR'])
# -

mom_compare_spf

mom_joint_compare_spf

mom_compare_sce

mom_joint_compare_sce


