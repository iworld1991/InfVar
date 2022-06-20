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

# ## Theories of Expectation Formation with Inflation Expectation
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

import numpy as np
from scipy.optimize import minimize
from numba import njit, float64, int64
from numba.experimental import jitclass
import pandas as pd
lw = 4
pd.options.display.float_format = '{:,.2f}'.format

from SMMEst import RationalExpectationAR


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
    if seed:
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
    model: a model class, i.e, sear representing sticky expectation and ar(1) 
    paras: an array vector of the parameters, which potentially includes both inflation process and expectation 
    data_mom_dic: a dictionary storing all data moments
    moment_choice: a list of moments, i.e. ['FE','FEATV','Var']
    how: string taking values of 'expectation','process','joint'
    n_exp_paras: nb of parameters for expectation model 
    
    outputs
    -------
    distance: the scalar of the moment distances to be minimized
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


# + code_folding=[0]
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


# -

if __name__ == "__main__":
       
    ## first, simulate some AR1 inflation with known parameters 
    ρ0,σ0 = 0.95,0.1
    history0 = SimAR1(ρ0,
                      σ0,
                      200)
    real_time0 = history0[11:-2] 
    realized0 = history0[12:-1]

# + code_folding=[0]
if __name__ == "__main__":
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
# -

# ###  Sticky Expectation and Noisy Information Hybrid(SENI) + AR1

model_data = [
    ('exp_para', float64[:]),             # parameters for expectation formation, empty for re
    ('process_para', float64[:]),         # parameters for inflation process, 2 entries for AR1 
    ('horizon', int64),                   # forecast horizons 
    ('real_time',float64[:]),             # real time data on inflation 
    ('history',float64[:]),               # a longer history of inflation 
    ('realized',float64[:])               # realized inflation 
]


# + code_folding=[1, 14, 18, 150]
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
        
        ## prepare matrices 
        nowcasts_to_burn = np.zeros((n_sim,n_history))  ### nowcasts matrix of which the initial simulation is to be burned 
        nowcasts_to_burn[:,0] = history[0]
        nowvars_to_burn = np.zeros((n_sim,n_history))   ### nowcasts uncertainty matrix
        nowvars_to_burn[:,0] = var_init
        Vars_to_burn = np.zeros((n_sim,n_history))      ### forecasting uncertainty matrix 
        
        ## fill the matrices for individual moments  
        for i in range(n_sim):
            signals_this_i = np.concatenate((nowsignals_pb_to_burn[i,:],nowsignals_pr_to_burn[i,:]),axis=0).reshape((2,-1))
            ## the histories signals specific to i: the first row is public signals and the second is private signals 
            Pkalman = np.zeros((n_history,nb_s))
            ## Kalman gains of this agent for respective signals 
            Pkalman[0,:] = 0  ## some initial values 
            
            for t in range(n_history-1):
                step1_vars_to_burn = ρ**2*nowvars_to_burn[i,t] + σ**2
                ## prior uncertainty 
                
                inv = np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v) 
                ## the inverse of the noisiness matrix  
                
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
        
        ## compute population moments
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

# + code_folding=[0]
if __name__ == "__main__":
    exp_paras_fake =  np.array([0.3,0.3,0.2])
    ## initialize the ar instance
    seniar0 = SENIHybridAR(exp_para = exp_paras_fake,
                           process_para = np.array([ρ0,σ0]),
                           real_time = real_time0,
                           history = history0,
                           horizon = 1)

    seniar0.GetRealization(realized0)
# -

# #### Estimate Hybrid using RE data 
# - This does NOT work correctly now! Hence, not used in the paper.

# + code_folding=[0]
if __name__ == "__main__":


    ## initialize an re instance 
    rear0 = RationalExpectationAR(exp_para = np.array([]),
                                  process_para = np.array([ρ0,σ0]),
                                  real_time = real_time0,
                                  history = history0,
                                  horizon = 1)
    
    rear0.GetRealization(realized0) 


    ## fake data moments dictionary 
    data_mom_dict_re = rear0.SMM()

# + code_folding=[0]
if __name__ == "__main__":

    ## only expectation estimation 

    moments0 = ['FE',
                'FEVar',
                'FEATV',
                'Disg',
                'DisgVar',
                'Var']

    def Objseniar_re(paras):
        scalar = ObjGen(seniar0,
                        paras = paras,
                        data_mom_dict = data_mom_dict_re,
                        moment_choice = moments0,
                        how = 'expectation')
        return scalar


    ## invoke estimation 
    
    Est = ParaEst(Objseniar_re,
            para_guess = np.array([0.5,0.5,0.5]),
            method='Nelder-Mead')
    
    print('True parameters: ',str(np.array([1.0,0.0,0.0]))) ## rational expectations 
    print('Estimates: ',str(Est))
    
# -

# #### Estimate Hybrid using Hybrid 
#
# - This does not work correctly till now. Not used in the paper.

# + code_folding=[0]
if __name__ == "__main__":

    ## get a fake data moment dictionary under a different parameter 
    exp_paras_fake = np.array([0.2,0.4,0.5])
    seniar1 = SENIHybridAR(exp_para = exp_paras_fake,
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
    Est = ParaEst(Objseniar_seni,
            para_guess = np.array([0.5,0.5,0.5]),
            method='Nelder-Mead')


    print('True parameters: ',str(exp_paras_fake)) ## rational expectations 
    print('Estimates: ',str(Est))
# -

# #### Joint Estimation 

# + code_folding=[0, 4, 15]
if __name__ == "__main__":

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
        scalar = ObjGen(seniar0,
                        paras = paras,
                        data_mom_dict = data_mom_dict_seni,
                        moment_choice = moments1,
                        how ='joint',
                        n_exp_paras = 3)
        return scalar

    ## invoek estimation 
    Est = ParaEst(Objseniar_joint,
            para_guess = np.array([0.4,0.2,0.3,0.8,0.2]),
            method='Nelder-Mead')
    
    print('True process parameters: ',str(np.array([ρ0,σ0])))
    print('Estimates: ',str(Est[3:]))
    print('True expectation parameter',str(exp_paras_fake))  
    print('Estimates: ',str(Est[0:3]))


# -

# ###  Sticky Expectation and Noisy Information Hybrid(SENI) + SV

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
        
        ## prepare matrices 
        nowcasts_to_burn = np.zeros((n_sim,n_history))  ### nowcasts matrix of which the initial simulation is to be burned 
        nowcasts_to_burn[:,0] = p_now[0]
        nowvars_to_burn = np.zeros((n_sim,n_history))   ### nowcasts uncertainty matrix
        nowvars_to_burn[:,0] = var_init
        Vars_to_burn = np.zeros((n_sim,n_history))      ### forecasting uncertainty matrix 
        
        
        ## fill the matrices for individual moments        
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
                ## prior uncertainty
                
                inv = np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v) 
                ## the inverse of the noisiness matrix
                
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

        
        ## compute population moments
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

# + code_folding=[]
if __name__ == "__main__":

    ## initial a sv instance
    senisv0 = SENIHybridSV(exp_para = np.array([0.5,0.23,0.32]),
                                   process_para = np.array([0.1]),
                                   real_time = xx_real_time,
                                   history = xx_real_time) ## history does not matter here, 


    senisv0.GetRealization(xx_realized)
