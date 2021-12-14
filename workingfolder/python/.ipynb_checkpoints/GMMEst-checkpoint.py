# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## GMM Estimation of Model Parameters of Expectation Formation
#
# - This notebook includes functions that estimate the parameters of different theories
# - It allows for flexible choices of moments to be used, forecast error, disagreement, and uncertainty, etc. 
# - It includes 
#   - A general function that implements the estimation using the minimum distance algorithm. 
#   - Model-specific functions that take real-time data and process parameters as inputs and produces forecasts and moments as outputs. It is model-specific because different models of expectation formation bring about different forecasts. 
#   - Auxiliary functions that compute the moments as well as the difference of data and model prediction, which will be used as inputs for GMM estimator. 

# ### 1. Estimation algorithms 

from scipy.optimize import minimize
from scipy.optimize import fixed_point as fp
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.api import AR
import copy as cp 
from scipy.stats import bernoulli
#from numpy import *
#from numba import jit
import time


# + {"code_folding": [1, 26]}
# a general-purpose estimating function of the parameter
def Estimator(obj_func,
              para_guess,
              method='CG',
              bounds = None,
              options = {'disp': True}):
    """
    Inputs
    ------
    - moments: a function of the rigidity model parameter  
    - method: method of optimization, i.e. 
         -  'L-BFGS-B', bounds=((1e-10, None), (1e-10, None))
    
    Outputs
    -------
    - parameter: an array of estimated parameter
    """
    
    parameter = minimize(obj_func,
                         x0 = para_guess,
                         method = method,
                         bounds = bounds,
                         options = options)['x']
    return parameter 

# a function that prepares moment conditions. So far the loss being simply the norm of the difference
def PrepMom(model_moments,
            data_moments):
    """
    Inputs
    -----
    model_moments: an array of moments from a certain model, i.e. forecast error, disagreement and uncertainty. 
    data_moments: an array of moments computed from the survey data
    
    Outputs
    ------
    diff: the Euclidean distance of two arrays of data and model 
    
    """
    distance = model_moments - data_moments
    diff = np.linalg.norm(distance)
    return diff



# + {"code_folding": [0, 4, 13]}
## auxiliary functions 
def hstepvar(h,sigma,rho):
    return np.sum([ rho**(2*i)*sigma**2 for i in range(h)])

def ForecastPlot(test):
    m_ct = len(test)
    x = plt.figure(figsize=([3,3*m_ct]))
    for i,val in enumerate(test):
        plt.subplot(m_ct,1,i+1)
        plt.plot(test[val],label=val)
        plt.legend(loc=1)
    return x
     
def ForecastPlotDiag(test,
                     data,
                     legends=['model','data'],
                     diff_scale = False):
    m_ct = len(test)
    x = plt.figure(figsize=([3,3*m_ct]))
    if diff_scale == False:
        for i,val in enumerate(test):
            plt.subplot(m_ct,1,i+1)
            plt.plot(test[val],'s-',label=legends[0]+ ': ' +val)
            plt.plot(np.array(data[val]),'o-',label=legends[1] +':'+ val)
            plt.legend(loc=1)
    if diff_scale == True:
        for i,val in enumerate(test):
            ax1 = plt.subplot(m_ct,1,i+1)
            ax1.plot(test[val],'s-',label=legends[0]+ ':' +val)
            ax1.legend(loc=0)
            ax2 = ax1.twinx()
            ax2.plot(np.array(data[val]),'o-',color='steelblue',label=legends[1] +':'+ val)
            ax2.legend(loc=3)
    return x
        
### AR1 simulator 
def AR1_simulator(rho,sigma,nobs):
    xxx = np.zeros(nobs+1)
    shocks = np.random.randn(nobs+1)*sigma
    xxx[0] = 0 
    for i in range(nobs):
        xxx[i+1] = rho*xxx[i] + shocks[i+1]
    return xxx[1:]


# + {"code_folding": [0]}
## some process parameters 
rho = 0.95
sigma = 0.1
process_para = {'rho':rho,
                'sigma':sigma}

# +
## Rational Expectation

# + {"code_folding": []}
## SE expectation parameters 
SE_para_default = {'lambda':0.4}

# + {"code_folding": [0, 4, 30, 43, 59, 110, 144, 180, 249, 319, 348, 377, 387, 395, 418, 441, 468, 506, 544, 576, 605, 630, 644, 671, 683, 695, 710, 725, 737, 813, 829]}
## Sticky Expectation(SE) class 


class StickyExpectation:
    def __init__(self,
                 real_time,
                 history,
                 horizon=1,
                 process_para = process_para,
                 exp_para = SE_para_default,
                 max_back =10,
                 moments = ['Forecast','Disg','Var']):
        self.history = history
        self.real_time = real_time
        self.n = len(real_time)
        self.horizon = horizon
        self.process_para = process_para
        self.exp_para = exp_para
        self.max_back = max_back
        self.data_moms_dct ={}
        self.para_est = {}
        self.moments = moments
        self.all_moments = ['Forecast','FE','Disg','Var']
        self.realized = None
        self.sim_realized = None
        
    def GetRealization(self,
                       realized_series):
        self.realized = realized_series 
    
    def SimulateRealization(self):
        n = self.n
        rho = self.process_para['rho']
        sigma = self.process_para['sigma']
        np.random.seed(12345)
        shocks = np.random.randn(n)*sigma
        sim_realized = np.zeros(n)
        for i in range(n):
            cum_shock = sum([rho**h*shocks[h] for h in range(self.horizon)])
            sim_realized[i] = rho**self.horizon*self.real_time[i] + cum_shock
        self.sim_realized = sim_realized
        return self.sim_realized
    
    def SimulateRealizationNoShock(self):
        n = self.n
        rho = self.process_para['rho']
        sigma = 0
        shocks = np.random.randn(n)*sigma
        sim_realized_noshock = np.zeros(n)
        for i in range(n):
            cum_shock = sum([rho**h*shocks[h] for h in range(self.horizon)])
            sim_realized_noshock[i] = rho**self.horizon*self.real_time[i] + cum_shock
        self.sim_realized_noshock = sim_realized_noshock
        return self.sim_realized_noshock
    
#################################
######## specific to new moments
################################

    def GMM(self):
        ## parameters and information set
        
        real_time = self.real_time
        history = self.history
        #realized = self.realized
        #sim_realized = self.sim_realized
        n = len(real_time)
        rho = self.process_para['rho']
        sigma =self.process_para['sigma']
        lbd = self.exp_para['lambda']
        #max_back = self.max_back
        horizon = self.horizon      
        n_burn = len(history) - n
        #n_history = n + n_burn  # of course equal to len(history)
        
        ## moments 
        
        # FE
        FE = 0  
        FEVar = ((1-lbd)*2*sigma**2+lbd**2*hstepvar(horizon,sigma,rho))/(1-(1-lbd)**2*rho**2)
        #FEVar = lbd**2*rho**(2*horizon)*sigma**2/(1-(1-lbd)**2)
        ## because FE_t' = (1-lbd)*rho*FE_t + (1-lbd) eps + lbd FE*
        FEATV = (1-lbd)*rho*FEVar
        
        # Disg
        Disg = rho**(2*horizon)*sigma**2/(1-rho**2*(1-lbd)**2)
        DisgVar = rho**(4*horizon)*sigma**4/(1-rho**(2*horizon)*(1-lbd)**2)
        DisgATV = rho**(2*horizon)*(1-lbd)**2*DisgVar
        #Disg = rho**(2*horizon)*(1-lbd)**2*lbd**2*sigma**2/(1- lbd**2*(1-lbd)**2) ## might be wrong. needs to check 
        
        # Var
        Var = sigma**2/(1-rho**2)*(1-lbd*rho**(2*horizon)/(1-(1-lbd)*rho*2))
        
        #ATV = (1-lbd)**Disg                               
        #Var = sum([(1-lbd)**tau*lbd*hstepvar(horizon+tau,sigma,rho) for tau in range(n + n_burn)])
        
        
        self.GMMMoments = {"FE":FE,
                           "FEVar":FEVar,
                           "FEATV":FEATV,
                           "Disg":Disg,
                           "DisgVar":DisgVar,
                           "DisgATV":DisgATV,
                           "Var":Var}
        return self.GMMMoments
    
#################################
######## specific to new moments
################################

    def SMM(self):
            
        ## simulate forecasts 
        
        self.ForecasterbySim(n_sim = 200)
        moms_sim = self.forecast_moments_sim
        
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
    
        self.SMMMoments = {"FE":FE_sim,
                           "FEVar":FEVar_sim,
                           "FEATV":FEATV_sim,
                           "Disg":Disg_sim,
                           "DisgVar":DisgVar_sim,
                           "DisgATV":DisgATV_sim,
                           "Var":Var_sim}
        return self.SMMMoments
    
#################################
######## New
################################

    def ProcessGMM(self):
        
        ## parameters and information set

        ## model 
        rho = self.process_para['rho']
        sigma = self.process_para['sigma']
        history = self.history
        
        resd = 0 
        Yresd = 0 
        resdVar = sigma**2
        YVar = sigma**2/(1-rho**2)
        YATV = rho*YVar
        
        
        self.ProcessMoments= {"Yresd":Yresd,
                              "YVar":YVar,
                              "resdVar":resdVar,
                              #"YATV":YATV
                             }
        ## data 
        resds = np.mean(history[1:]-rho*history[:-1])
        resd_data = np.mean(resds)
        resdVar_data = np.mean(resds**2)
        Yresd_data = np.mean(history[:-1]*resds)
        YVar_data = np.var(history)
        YATV_data = np.cov(np.stack((history[1:],history[:-1]),axis=0))[0,1]
        
        
        self.ProcessDataMoments = {"Yresd":Yresd_data,
                                  "YVar": YVar_data,
                                  "resdVar":resdVar_data,
                                   #"YATV":YATV_data
                                  }
    
    def Forecaster(self):
        ## inputs 
        real_time = self.real_time
        history = self.history
        realized = self.realized
        sim_realized = self.sim_realized
        n = len(real_time)
        rho = self.process_para['rho']
        sigma =self.process_para['sigma']
        lbd = self.exp_para['lambda']
        max_back = self.max_back
        horizon = self.horizon      
        n_burn = len(history) - n
        n_history = n + n_burn  # of course equal to len(history)
        
        
        ## forecast moments 
        Var_array = np.empty(n)
        for i in range(n):
            Var_this_i = sum([lbd*(1-lbd)**tau*hstepvar(tau+horizon,sigma,rho) for tau in range(i + n_burn)])
            Var_array[i] = Var_this_i
        Var = Var_array
        
        # average forecast 
        nowcast_array = np.empty(n)
        for i in range(n):
            nowcast_this_i = sum([lbd*(1-lbd)**tau*(rho**tau)*history[i+n_burn-tau] for tau in range(i+n_burn)])
            nowcast_array[i] = nowcast_this_i
        nowcast = nowcast_array
        forecast = rho**horizon*nowcast
        
        # forecast errors
        if realized is not None:
            FE = forecast - realized
        elif sim_realized is not None:
            FE = forecast - sim_realized
            
        ## diagreement 
        Disg_array = np.empty(n)
        for i in range(n):
            Disg_this_i = sum([lbd*(1-lbd)**tau*(rho**(tau+horizon)*history[i+n_burn-tau] - forecast[i])**2 for tau in range(i+n_burn)])
            Disg_array[i] = Disg_this_i
        Disg = Disg_array
        
        ## autocovariance of forecast
        ATV_array = np.empty(n)
        for i in range(n-1):
            ATV_array[i+1] = (1-lbd)*np.sqrt(Disg[i])*np.sqrt(Disg[i+1])
        ATV_array[0] = ATV_array[1]
        ATV = ATV_array 
        
        ## autocovariance of forecast errors
        FEVar = np.empty(n)
        FEATV = np.empty(n)
        for i in range(n):
            FEVar[i] = sum([lbd*(1-lbd)**tau*
                            (sum([rho**(x+horizon)*FE[i-x] for x in range(tau) if x<=i])
                             - FE[i])**2 
                            for tau in range(i+n_burn)])
        for i in range(n-1):
            FEATV[i+1] = (1-lbd)*np.sqrt(FEVar[i])*np.sqrt(FEVar[i+1])
        FEATV[0] = FEATV[1]
            
        self.forecast_moments = {"Forecast":forecast,
                                 "FE":FE,
                                 "Disg":Disg,
                                 "Var":Var}
        return self.forecast_moments
    
    def ForecasterbySim(self,
                        n_sim = 100):
        ## inputs 
        real_time = self.real_time
        history  = self.history
        realized = self.realized
        sim_realized = self.sim_realized
        n = len(real_time)
        rho = self.process_para['rho']
        sigma =self.process_para['sigma']
        lbd = self.exp_para['lambda']
        max_back = self.max_back
        horizon = self.horizon 
        n_burn = len(history) - n
        n_history = n + n_burn  # of course equal to len(history)
    
        
        ## simulation
        np.random.seed(12345)
        update_or_not = bernoulli.rvs(lbd,size = [n_sim,n_history])
        most_recent_when = np.empty([n_sim,n_history],dtype = int)
        nowcasts_to_burn = np.empty([n_sim,n_history])
        Vars_to_burn = np.empty([n_sim,n_history])
        
        ## look back for the most recent last update for each point of time  
        for i in range(n_sim):
            for j in range(n_history):
                if np.any([x for x in range(j) if update_or_not[i,j-x] == 1]):
                    most_recent_when[i,j] = np.min([x for x in range(j) if update_or_not[i,j-x] == 1])
                else:
                    most_recent_when[i,j] = j
                nowcasts_to_burn[i,j] = history[j - most_recent_when[i,j]]*rho**most_recent_when[i,j]
                Vars_to_burn[i,j]= hstepvar((most_recent_when[i,j]+horizon),sigma,rho)
        
        ## burn initial forecasts since history is too short 
        nowcasts = np.array( nowcasts_to_burn[:,n_burn:] )
        forecasts = rho**horizon*nowcasts
        Vars = np.array( Vars_to_burn[:,n_burn:])
        
        if realized is not None:
            FEs = forecasts - realized
        elif self.sim_realized is not None:
            FEs = forecasts - self.sim_realized
        
        ## compuate population moments
        forecasts_mean = np.mean(forecasts,axis = 0)
        forecasts_var = np.var(forecasts,axis = 0)
        
        if realized is not None:
            FEs_mean = forecasts_mean - realized
        elif self.sim_realized is not None:
            FEs_mean = forecasts_mean - self.sim_realized
            
        Vars_mean = np.mean(Vars,axis = 0) ## need to change 
        
        forecasts_vcv = np.cov(forecasts.T)
        forecasts_atv = np.array([forecasts_vcv[i+1,i] for i in range(n-1)])
        FEs_vcv = np.cov(FEs.T)
        FEs_atv = np.array([FEs_vcv[i+1,i] for i in range(n-1)]) ## this is no longer needed
        
        self.forecast_moments_sim = {"Forecast":forecasts_mean,
                                     "FE":FEs_mean,
                                     "Disg":forecasts_var,
                                     "Var":Vars_mean}
        return self.forecast_moments_sim
    
###################################
######## New
#####################################

    def SE_EstObjfuncGMM(self,
                         se_paras):
        """
        input
        -----
        lbd: the parameter of SE model to be estimated
        
        output
        -----
        the objective function to minmize
        """
        moments = self.moments
        data_moms_scalar_dct = self.data_moms_scalar_dct
        
        lbd = se_paras[0]
        SE_para = {"lambda":lbd}
        self.exp_para = SE_para  # give the new lambda
        
        SE_moms_scalar_dct = self.GMM().copy()
        SE_moms_scalar = np.array([SE_moms_scalar_dct[key] for key in moments] )
        data_moms_scalar = np.array([data_moms_scalar_dct[key] for key in moments] )
        distance = SE_moms_scalar - data_moms_scalar
        obj_func = np.linalg.norm(distance)
        return obj_func 
    
###################################
######## New
#####################################

    def SE_EstObjfuncSMM(self,
                         se_paras):
        """
        input
        -----
        lbd: the parameter of SE model to be estimated
        
        output
        -----
        the objective function to minmize
        """
        moments = self.moments
        data_moms_scalar_dct = self.data_moms_scalar_dct
        
        lbd = se_paras[0]
        SE_para = {"lambda":lbd}
        self.exp_para = SE_para  # give the new lambda
        
        SE_moms_scalar_dct = self.SMM().copy()
        SE_moms_scalar = np.array([SE_moms_scalar_dct[key] for key in moments] )
        data_moms_scalar = np.array([data_moms_scalar_dct[key] for key in moments] )
        distance = SE_moms_scalar - data_moms_scalar
        obj_func = np.linalg.norm(distance)
        return obj_func 
    
###################################
######## New
#####################################

    def PlotLossGMM(self,
                    lbds = np.linspace(0.01,0.99,20)):
        loss  = np.array([self.SE_EstObjfuncGMM(np.array([lbd])) for lbd in lbds])
        self.fig = plt.plot(lbds,loss,lw = 3)
        plt.title('Loss function of GMM')
        return self.fig  

###################################
######## New
#####################################
    def PlotLossSMM(self,
                    lbds = np.linspace(0.01,0.99,20)):
        loss  = np.array([self.SE_EstObjfuncSMM(np.array([lbd])) for lbd in lbds])
        self.fig = plt.plot(lbds,loss,lw = 3)
        plt.title('Loss function of SMM')
        return self.fig
    
    ## a function estimating SE model parameter only 
    def SE_EstObjfunc(self,
                      se_paras):
        """
        input
        -----
        lbd: the parameter of SE model to be estimated
        
        output
        -----
        the objective function to minmize
        """
        lbd = se_paras[0]
        moments = self.moments
        SE_para = {"lambda":lbd}
        self.exp_para = SE_para  # give the new lambda
        data_moms_dct = self.data_moms_dct
        SE_moms_dct = self.Forecaster().copy()
        SE_moms = np.array([SE_moms_dct[key] for key in moments] )
        data_moms = np.array([data_moms_dct[key] for key in moments] )
        distance = SE_moms - data_moms
        obj_func = np.linalg.norm(distance)
        return obj_func 
    
    def SE_EstObjfuncSim(self,
                         lbd):
        """
        input
        -----
        lbd: the parameter of SE model to be estimated
        
        output
        -----
        the objective function to minmize
        """
        moments = self.moments
        SE_para = {"lambda":lbd}
        self.exp_para = SE_para  # give the new lambda
        data_moms_dct = self.data_moms_dct
        np.random.seed(12345)
        SE_moms_dct = self.ForecasterbySim().copy()
        SE_moms = np.array([SE_moms_dct[key] for key in moments] )
        data_moms = np.array([data_moms_dct[key] for key in moments] )
        distance = SE_moms - data_moms
        obj_func = np.linalg.norm(distance)
        return obj_func
    
    def SE_EstObjfuncJoint(self,
                          paras):
        lbd,rho,sigma = paras
        moments = self.moments
        realized = self.realized
        
        process_para_joint = {'rho':rho,
                              'sigma':sigma}
        SE_para = {"lambda":lbd}
        self.exp_para = SE_para  # give the new lambda
        self.process_para = process_para_joint
        data_moms_dct = self.data_moms_dct
        sim_realized =  self.SimulateRealizationNoShock()
        SE_moms_dct = self.Forecaster().copy()
        SE_moms = np.array([SE_moms_dct[key] for key in moments] )
        data_moms = np.array([data_moms_dct[key] for key in moments] )
        n = len(sim_realized)
        SE_moms_stack = np.concatenate((SE_moms, sim_realized.reshape(1,n)), axis=0)
        data_moms_stack = np.concatenate((data_moms, realized.reshape(1,n)), axis=0)
        distance = SE_moms_stack - data_moms_stack
        obj_func = np.linalg.norm(distance)
        return obj_func
    
###################################
######## New
#####################################

    def SE_EstObjfuncGMMJoint(self,
                              paras):
        lbd,rho,sigma = paras
        moments = self.moments
        realized = self.realized
        
        process_para_joint = {'rho':rho,
                              'sigma':sigma}
        SE_para = {"lambda":lbd}
        self.exp_para = SE_para  # give the new lambda
        self.process_para = process_para_joint
        
        ## for the new parameters, update process GMM 
        self.ProcessGMM()
        ProcessDataMoments = self.ProcessDataMoments
        ProcessMoments = self.ProcessMoments
        
        ## get data and model moments conditions 
        
        data_moms_scalar_dct = self.data_moms_scalar_dct
        SE_moms_scalar_dct = self.GMM().copy()
        SE_moms_scalar = np.array([SE_moms_scalar_dct[key] for key in moments] )
        data_moms_scalar = np.array([data_moms_scalar_dct[key] for key in moments] )
        
        process_moms = np.array([ProcessMoments[key] for key in ProcessMoments.keys()])
        data_process_moms = np.array([ProcessDataMoments[key] for key in ProcessDataMoments.keys()])
        #print(ProcessMoments)
        #print(ProcessDataMoments)
        SE_moms_scalar_stack = np.concatenate((SE_moms_scalar,process_moms))
        data_moms_scalar_stack = np.concatenate((data_moms_scalar, data_process_moms))
        distance = SE_moms_scalar_stack - data_moms_scalar_stack
        obj_func = np.linalg.norm(distance)
        return obj_func
    
###################################
######## New
#####################################

    def SE_EstObjfuncSMMJoint(self,
                              paras):
        lbd,rho,sigma = paras
        moments = self.moments
        realized = self.realized
        
        process_para_joint = {'rho':rho,
                              'sigma':sigma}
        SE_para = {"lambda":lbd}
        self.exp_para = SE_para  # give the new lambda
        self.process_para = process_para_joint
        
        ## for the new parameters, update process GMM 
        self.ProcessGMM()
        ProcessDataMoments = self.ProcessDataMoments
        ProcessMoments = self.ProcessMoments
        
        ## get data and model moments conditions 
        
        data_moms_scalar_dct = self.data_moms_scalar_dct
        SE_moms_scalar_dct = self.SMM().copy()
        SE_moms_scalar = np.array([SE_moms_scalar_dct[key] for key in moments] )
        data_moms_scalar = np.array([data_moms_scalar_dct[key] for key in moments] )
        
        process_moms = np.array([ProcessMoments[key] for key in ProcessMoments.keys()])
        data_process_moms = np.array([ProcessDataMoments[key] for key in ProcessDataMoments.keys()])
        #print(ProcessMoments)
        #print(ProcessDataMoments)
        SE_moms_scalar_stack = np.concatenate((SE_moms_scalar,process_moms))
        data_moms_scalar_stack = np.concatenate((data_moms_scalar, data_process_moms))
        distance = SE_moms_scalar_stack - data_moms_scalar_stack
        obj_func = np.linalg.norm(distance)
        return obj_func
    
###################################
######## New
#####################################
    
    def SE_EstObjfuncSMMwm1st(self,
                              se_paras):
        
        # estimate first step 
        self.ParaEstimateSMM()
        self.WM1stSMM()
        
        ## data moments
        data_moms_scalar_dct = self.data_moms_scalar_dct
        
        ## 1-step weighting matrix 
        wm1st = self.wm1st
        
        ## parameters 
        lbd = se_paras[0]
        SE_para = {"lambda":lbd}
        self.exp_para = SE_para  # give the new lambda
        
        SE_moms_scalar_dct = self.SMM().copy()
        
        sim_moms = np.array([SE_moms_scalar_dct[key] for key in self.moments])
        data_moms = np.array([data_moms_scalar_dct[key] for key in self.moments])
        assert len(sim_moms) == len(data_moms), "not equal lenghth"
        distance = sim_moms - data_moms
        tmp = np.multiply(np.multiply(distance.T,wm1st),distance)  ## need to make sure it is right. 
        obj_func = np.sum(tmp)
        return obj_func
    
###################################
######## New
#####################################

    def SE_EstObjfuncSMMwmboot(self,
                                  se_paras):
        
        # estimate first step 
        self.ParaEstimateSMM()
        self.WMbootSMM()
        
        ## data moments
        data_moms_scalar_dct = self.data_moms_scalar_dct
        
        ## 1-step weighting matrix 
        wm1st = self.wm_boot
        
        ## parameters 
        lbd = se_paras[0]
        SE_para = {"lambda":lbd}
        self.exp_para = SE_para  # give the new lambda
        
        SE_moms_scalar_dct = self.SMM().copy()
        
        sim_moms = np.array([SE_moms_scalar_dct[key] for key in self.moments])
        data_moms = np.array([data_moms_scalar_dct[key] for key in self.moments])
        assert len(sim_moms) == len(data_moms), "not equal lenghth"
        distance = sim_moms - data_moms
        tmp = np.multiply(np.multiply(distance.T,wm1st),distance)  ## need to make sure it is right. 
        obj_func = np.sum(tmp)
        return obj_func
    
    ## feeds the instance with data moments dictionary 
    def GetDataMoments(self,
                       data_moms_dct):
        self.data_moms_dct = data_moms_dct
        
#################################
######## New
################################

        data_moms_scalar_dct = dict(zip(data_moms_dct.keys(),
                                        [np.mean(data_moms_dct[key]) for key in data_moms_dct.keys()]
                                       )
                                   )
        data_moms_scalar_dct['FEVar'] = data_moms_dct['FE'].var()
        FE_stack = np.stack((data_moms_dct['FE'][1:],data_moms_dct['FE'][:-1]),axis = 0)
        data_moms_scalar_dct['FEATV'] = np.cov(FE_stack)[0,1]
        data_moms_scalar_dct['DisgVar'] = data_moms_dct['Disg'].var()
        Disg_stack = np.stack((data_moms_dct['Disg'][1:],data_moms_dct['Disg'][:-1]),axis = 0)
        data_moms_scalar_dct['DisgATV'] = np.cov(Disg_stack)[0,1]
        
        self.data_moms_scalar_dct = data_moms_scalar_dct
        
###################################
######## New
####################################
        
    def ParaEstimateGMM(self,
                        para_guess = 0.2,
                        method='COBYLA',
                        bounds = None,
                        options = None):
        self.para_estGMM = Estimator(self.SE_EstObjfuncGMM,
                                     para_guess = para_guess,
                                     method = method,
                                     bounds = bounds,
                                     options = options)
###################################
######## New
####################################
        
    def ParaEstimateSMM(self,
                        wb = 'identity',
                        para_guess = 0.2,
                        method='Nelder-Mead',
                        bounds = None,
                        options = None):
        if wb == 'identity':
            self.para_estSMM = Estimator(self.SE_EstObjfuncSMM,
                                         para_guess = para_guess,
                                         method = method,
                                         bounds = bounds,
                                         options = options)
        elif wb =='2-step':
            self.para_estSMM = Estimator(self.SE_EstObjfuncSMMwm1st,
                                         para_guess = para_guess,
                                         method = method,
                                         bounds = bounds,
                                         options = options)
        elif wb =='bootstrap':
            self.para_estSMM = Estimator(self.SE_EstObjfuncSMMwmboot,
                                         para_guess = para_guess,
                                         method = method,
                                         bounds = bounds,
                                         options = options)
        return self.para_estSMM
    
    ## invoke the estimator 
    def ParaEstimate(self,
                     para_guess = 0.2,
                     method='CG',
                     bounds = None,
                     options = None):
        self.para_est = Estimator(self.SE_EstObjfunc,
                                  para_guess = para_guess,
                                  method = method,
                                  bounds = bounds,
                                  options = options)
        return self.para_est
        
    def ParaEstimateSim(self,
                        para_guess = 0.2,
                        method='BFGS',
                        bounds = None,
                        options = None):
        self.para_est_sim = Estimator(self.SE_EstObjfuncSim,
                                  para_guess=para_guess,
                                  method = method,
                                  bounds = bounds,
                                  options = options)
        return self.para_est_sim
    
    def ParaEstimateJoint(self,
                          para_guess = (0.5,0.1,0.2),
                          method='BFGS',
                          bounds = None,
                          options = None):
        self.para_est_joint = Estimator(self.SE_EstObjfuncJoint,
                                  para_guess = para_guess,
                                  method = method,
                                  bounds = bounds,
                                  options = options)
        
###################################
######## New
#####################################
    
    def ParaEstimateGMMJoint(self,
                             para_guess = (0.5,0.1,0.2),
                             method='BFGS',
                             bounds = None,
                             options = None):
        self.para_est_GMM_joint = Estimator(self.SE_EstObjfuncGMMJoint,
                                            para_guess = para_guess,
                                            method = method,
                                            bounds = bounds,
                                            options = options)
        
###################################
######## New
#####################################
    
    def ParaEstimateSMMJoint(self,
                             para_guess = (0.5,0.8,0.2),
                             method='Nelder-Mead',
                             bounds = None,
                             options = None):
        self.para_est_SMM_joint = Estimator(self.SE_EstObjfuncSMMJoint,
                                            para_guess = para_guess,
                                            method = method,
                                            bounds = bounds,
                                            options = options)
        return self.para_est_SMM_joint
        
    def ForecastPlot(self,
                     all_moms = False):
        plt.style.use('ggplot')
        if all_moms == False:
            m_ct = len(self.moments)
            x = plt.figure(figsize=([3,3*m_ct]))
            for i,val in enumerate(self.moments):
                plt.subplot(m_ct,1,i+1)
                plt.plot(self.forecast_moments[val],label=val)
                plt.legend(loc=1)
        else:
            m_ct = len(self.all_moments)
            x = plt.figure(figsize=([3,3*m_ct]))
            for i,val in enumerate(self.all_moments):
                plt.subplot(m_ct,1,i+1)
                plt.plot(self.forecast_moments[val],label=val)
                plt.legend(loc=1)
            
#################################
######## New
################################

    def ForecastPlotDiag(self,
                         all_moms = False,
                         diff_scale = False,
                         how = 'GMMs'):
        if how =='GMMs':
            exp_para_est_dct = {'lambda':self.para_est[0]}
        elif how == "GMM":
            exp_para_est_dct = {'lambda':self.para_estGMM}
        elif how == "SMM":
            exp_para_est_dct = {'lambda':self.para_estSMM}
        elif how =="SMMs":
            exp_para_est_dct = {'lambda':self.para_est_sim[0]}
        elif how =="SMMjoint":
            lbd,rho,sigma = self.para_est_SMM_joint
            exp_para_est_dct = {'lambda':lbd}
            process_para_est_dct = {'rho':rho,
                                   'sigma':sigma}
        elif how =="GMMsjoint":
            lbd,rho,sigma = self.para_est_joint
            exp_para_est_dct = {'lambda':lbd}
            process_para_est_dct = {'rho':rho,
                                   'sigma':sigma}
            
        ## plot 
        new_instance = cp.deepcopy(self)
        new_instance.exp_para = exp_para_est_dct
        self.forecast_moments_est = new_instance.ForecasterbySim()
        plt.style.use('ggplot')
        if all_moms == False:
            moments_to_plot = self.moments
        else:
            moments_to_plot = self.all_moments

        m_ct = len(moments_to_plot)
        x = plt.figure(figsize=([3,3*m_ct]))
        if diff_scale == False:
            for i,val in enumerate(moments_to_plot):
                plt.subplot(m_ct,1,i+1)
                plt.plot(self.forecast_moments_est[val],'s-',label='model:'+ val)
                plt.plot(np.array(self.data_moms_dct[val]),'o-',label='data:'+ val)
                plt.legend(loc=1)
        if diff_scale == True:
            for i,val in enumerate(moments_to_plot):
                ax1 = plt.subplot(m_ct,1,i+1)
                ax1.plot(self.forecast_moments_est[val],'rs-',label='model:'+ val)
                ax1.legend(loc=0)
                ax2 = ax1.twinx()
                ax2.plot(np.array(self.data_moms_dct[val]),'o-',color='steelblue',label='(RHS) data:'+ val)
                ax2.legend(loc=3)
        
#################################
######## New
################################
                
    def WM1stSMM(self):
        """
        - get the 1-st step variance and covariance matrix
        """
        exp_para_est_dct = {'lambda':self.para_estSMM}
        self.exp_para = exp_para_est_dct
        
        sim_moms_dct = self.SMM()
        data_moms_scalar_dct = self.data_moms_scalar_dct
        sim_moms = np.array([sim_moms_dct[key] for key in self.moments])
        data_moms = np.array([data_moms_scalar_dct[key] for key in self.moments])
        distance = sim_moms - data_moms
        distance_diag = np.diag(distance*distance.T)
        self.wm1st = np.linalg.inv(distance_diag)
        return self.wm1st
    
    def WMbootSMM(self,
                  n_boot = 100):
        # data moments 
        data_moms_scalar_dct = self.data_moms_scalar_dct
        
        # parameters 
        exp_para_est_dct = {'lambda':self.para_estSMM}
        self.exp_para = exp_para_est_dct
        
        distance_boot = []
        
        for i in range(n_boot): 
            self.SMM()
            sim_moms_dct = self.SMMMoments
            sim_moms = np.array([sim_moms_dct[key] for key in self.moments])
            data_moms = np.array([data_moms_scalar_dct[key] for key in self.moments])
            distance = sim_moms - data_moms
            distance_boot.append(np.array(distance))
        #print(np.array(distance_boot).shape)
        vcv_boot_ary = np.array(distance_boot, dtype=np.float64)
        self.vcv_boot = np.cov(vcv_boot_ary.T)
        #print(self.vcv_boot)
        self.wm_boot = np.linalg.inv(self.vcv_boot)               


# + {"code_folding": [3, 32, 45, 59, 129, 207, 222, 234, 286, 319, 356, 389, 401, 422, 455, 484, 495, 507, 522, 550, 587, 595, 610, 626, 657, 670, 682, 694, 774, 791]}
## Noisy Information(NI) class 

class NoisyInformation:
    def __init__(self,
                 real_time,
                 history,
                 horizon = 1,
                 process_para = process_para, 
                 exp_para = {'sigma_pb':0.2,
                             'sigma_pr':0.2,
                             #'var_init':1,
                             #'y_init':0.1,
                             #'disg_init':0.1,
                            },
                 moments = ['Forecast','FE','Disg']):
        self.real_time = real_time
        self.history = history
        self.n = len(real_time)
        self.horizon = horizon
        self.process_para = process_para
        self.exp_para = exp_para
        self.data_moms_dct ={}
        self.para_est = {}
        self.moments = moments
        self.all_moments = ['Forecast','FE','Disg','Var']
        self.realized = None
        self.sim_realized = None
    
    def GetRealization(self,
                       realized_series):
        self.realized = realized_series   
    
    def SimulateRealization(self):
        n = self.n
        rho = self.process_para['rho']
        sigma =self.process_para['sigma']
        np.random.seed(12345)
        shocks = np.random.randn(n)*sigma
        sim_realized = np.zeros(n)
        for i in range(n):
            cum_shock = sum([rho**h*shocks[h] for h in range(self.horizon)])
            sim_realized[i] = rho**self.horizon*self.real_time[i] + cum_shock
        self.sim_realized = sim_realized
        return self.sim_realized
        
    def SimulateSignals(self):
        n = self.n
        n_history = len(self.history)
        sigma_pb = self.exp_para['sigma_pb']
        sigma_pr =self.exp_para['sigma_pr']
        np.random.seed(1234)
        s_pb = self.history + sigma_pb*np.random.randn(n_history)
        np.random.seed(12343)
        s_pr = self.history + sigma_pr*np.random.randn(n_history)
        self.signals = np.asmatrix(np.array([s_pb,s_pr]))
        self.signals_pb = s_pb
        
    # a function that generates population moments according to NI     
    
    def Forecaster(self):
        ## inputs 
        real_time = self.real_time
        history = self.history
        realized = self.realized
        sim_realized = self.sim_realized
        n = self.n
        n_burn = len(history) - n
        n_history = n + n_burn  # of course equal to len(history)
        rho  = self.process_para['rho']
        sigma = self.process_para['sigma']
        sigma_pb = self.exp_para['sigma_pb']
        sigma_pr =self.exp_para['sigma_pr']
        #var_init = self.exp_para['var_init']
        #######################
        var_init = 1
        ##################
        #y_init = self.exp_para['y_init
        #######################
        y_init = 0
        ##################
        #disg_init = self.exp_para['disg_init']
        #######################
        disg_init = 1
        #######################
        
        sigma_v = np.asmatrix([[sigma_pb**2,0],[0,sigma_pr**2]])
        horizon = self.horizon      
        signals = self.signals
        nb_s = len(self.signals) ## # of signals 
        H = np.asmatrix ([[1,1]]).T
        Pkalman = np.zeros([n_history,nb_s])
        nowcast_to_burn = np.zeros(n_history)
        nowcast_to_burn[0] = y_init
        nowvar_to_burn = np.zeros(n_history)
        nowvar_to_burn[0] = var_init
        Var_to_burn = np.zeros(n_history)
        nowdisg_to_burn = np.zeros(n_history)
        nowdisg_to_burn[0] = disg_init 
     
        ## forecast moments        
        for t in range(n_history-1):
            step1_vars_to_burn = rho**2*nowvar_to_burn[t] + sigma**2
            nowvar_to_burn[t+1] = step1_vars_to_burn - step1_vars_to_burn*\
                                          H.T*np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v)*H*step1_vars_to_burn
            Pkalman[t+1,:] = step1_vars_to_burn*H.T*np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v)
            nowcast_to_burn[t+1] = (1-Pkalman[t+1,:]*H)*rho*nowcast_to_burn[t] + Pkalman[t+1,:]*signals[:,t+1]
            nowdisg_to_burn[t+1] = (1-Pkalman[t+1,:]*H)**2*rho**2*nowdisg_to_burn[t] + Pkalman[t+1,1]**2*sigma_pr**2
        nowcast = nowcast_to_burn[n_burn:]
        forecast = rho**horizon*nowcast
        if realized is not None:
            FE = forecast - realized
        elif sim_realized is not None:
            FE = forecast - sim_realized 

        for t in range(n_history):
            Var_to_burn[t] = rho**(2*horizon)*nowvar_to_burn[t] + hstepvar(horizon,sigma,rho)
        Var = Var_to_burn[n_burn:] 
        
        nowdisg = nowdisg_to_burn[n_burn:]
        Disg = rho**(2*horizon)*nowdisg
        
        self.Kalman = Pkalman
        self.forecast_moments = {"Forecast":forecast,
                                 "FE":FE,
                                 "Disg":Disg,
                                 "Var":Var}
        
        return self.forecast_moments
    
    def ForecasterbySim(self,
                       n_sim = 100):
        # parameters
        real_time = self.real_time
        history = self.history
        realized = self.realized
        sim_realized = self.sim_realized
        n = self.n
        n_burn = len(history) - n
        n_history = n + n_burn  # of course equal to len(history)
        n_sim = n_sim ## number of agents 
        n_history = len(self.history)
        sigma_pb = self.exp_para['sigma_pb']
        sigma_pr =self.exp_para['sigma_pr']
        #var_init = self.exp_para['var_init']
        #######################
        var_init = 10
        ##################
        sigma_v = np.asmatrix([[sigma_pb**2,0],[0,sigma_pr**2]])
        horizon = self.horizon      
        
        ## simulate signals 
        self.SimulateSignals()
        signals = self.signals
        nb_s = len(self.signals) ## # of signals 
        H = np.asmatrix ([[1,1]]).T
        
        # randomly simulated signals 
        signal_pb = self.signals_pb 
        np.random.seed(12434)
        signals_pr = np.array([self.history]) + sigma_pr*np.random.randn(n_sim*n_history).reshape([n_sim,n_history])
        
        ## prepare matricies 
        nowcasts_to_burn = np.zeros([n_sim,n_history])
        nowcasts_to_burn[:,0] = history[0]
        nowvars_to_burn = np.zeros([n_sim,n_history])
        nowvars_to_burn[:,0] = var_init
        Vars_to_burn = np.zeros([n_sim,n_history])
        
        
        ## fill the matricies for individual moments        
        for i in range(n_sim):
            signals_this_i = np.asmatrix(np.array([signal_pb,signals_pr[i,:]]))
            Pkalman = np.zeros([n_history,nb_s])
            Pkalman[0,:] = 0 
            for t in range(n_history-1):
                step1_vars_to_burn = rho**2*nowvars_to_burn[i,t] + sigma**2
                nowvars_to_burn[i,t+1] = step1_vars_to_burn - step1_vars_to_burn*\
                                          H.T*np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v)*H*step1_vars_to_burn
                Pkalman[t+1,:] = step1_vars_to_burn*H.T*np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v)
                nowcasts_to_burn[i,t+1] = (1-Pkalman[t+1,:]*H)*rho*nowcasts_to_burn[i,t]+ Pkalman[t+1,:]*signals_this_i[:,t+1]
            for t in range(n_history):
                Vars_to_burn[i,t] = rho**(2*horizon)*nowvars_to_burn[i,t] + hstepvar(horizon,sigma,rho)
                
        nowcasts = nowcasts_to_burn[:,n_burn:]
        forecasts = rho**horizon*nowcasts 
        Vars = Vars_to_burn[:,n_burn:]
        
        ## compuate population moments
        forecasts_mean = np.mean(forecasts,axis=0)
        forecasts_var = np.var(forecasts,axis=0)
        if realized is not None:
            FEs_mean = forecasts_mean - realized
        elif sim_realized is not None:
            FEs_mean = forecasts_mean - sim_realized
            
        Vars_mean = np.mean(Vars,axis=0) ## need to change for time-variant volatility
        
        self.forecast_moments_sim = {"Forecast":forecasts_mean,
                                     "FE":FEs_mean,
                                     "Disg":forecasts_var,
                                     "Var":Vars_mean}
        return self.forecast_moments_sim
    
###################################
######## new ######################
###################################

    def KalmanVar(self,
                  nowvar):
        rho  = self.process_para['rho']
        sigma = self.process_para['sigma']
        sigma_pb = self.exp_para['sigma_pb']
        sigma_pr =self.exp_para['sigma_pr']
        
        
        sigma_v = np.asmatrix([[sigma_pb**2,0],[0,sigma_pr**2]])
        H = np.asmatrix ([[1,1]]).T

        step1_vars = rho**2*nowvar + sigma**2
        nowvar_next= step1_vars - step1_vars*H.T*np.linalg.inv(H*step1_vars*H.T+sigma_v)*H*step1_vars
        return nowvar_next
    
    def SteadyState(self):

        VarSS = fp(self.KalmanVar,
                   x0 = 0.1)   # steady state nowcasting uncertainty 
        
        self.VarSS = VarSS
        return self.VarSS
    
#################################
######## New
################################

    def GMM(self):
        ## parameters
        now_var = self.VarSS  ## nowcasting variance in steady state 
        rho  = self.process_para['rho']
        sigma = self.process_para['sigma']
        sigma_pb = self.exp_para['sigma_pb']
        sigma_pr =self.exp_para['sigma_pr']
        #var_init = self.exp_para['var_init']
        #y_init = self.exp_para['y_init']
        #disg_init = self.exp_para['disg_init']
        
        sigma_v = np.asmatrix([[sigma_pb**2,0],[0,sigma_pr**2]])
        horizon = self.horizon      
        H = np.asmatrix ([[1,1]]).T
        
        ## compute steady state Pkalman 
        step1_vars_SS = rho**2*now_var + sigma**2
        PkalmanSS = step1_vars_SS*H.T*np.linalg.inv(H*step1_vars_SS*H.T+sigma_v)
        ## GMM unconditional moments
        
        ### FE
        FE = 0
        FEVar = np.asscalar((rho**(2*horizon)*sigma_pb**2+PkalmanSS[:,0]**2*(1-rho**(2*horizon))/
                             (1-rho)*sigma**2)/(1-(1-PkalmanSS*H)**2))
        FEATV = np.asscalar((1-PkalmanSS*H)*FEVar)
        
        ### Disg
        Disg_now = np.asscalar((PkalmanSS[:1]*sigma_pr**2/(1-(1-PkalmanSS*H)**2))[:,1])
        Disg = Disg_now*rho**(2*horizon)
        
        DisgVar_now = np.asscalar((PkalmanSS[:1]*sigma_pr**4/(1-(1-PkalmanSS*H)**4))[:,1])
        DisgVar = rho**(4*horizon)*DisgVar_now
        
        DisgATV_now =np.asscalar((1-PkalmanSS*H)**2*DisgVar_now) 
        DisgATV = rho**(4*horizon)*DisgATV_now
        
        ### Var
        Var = np.asscalar(rho**(2*horizon)*now_var + hstepvar(horizon,sigma,rho))
        
        self.GMMMoments = {"FE":FE,
                           "FEVar":FEVar,
                           "FEATV":FEATV,
                           "Disg":Disg,
                           "DisgVar":DisgVar,
                           "DisgATV":DisgATV,
                           "Var":Var}
        return self.GMMMoments
    
#################################
######## New
################################

    def SMM(self):
        
        ## simulate
        self.ForecasterbySim(n_sim = 200)
        moms_sim = self.forecast_moments_sim
        
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
        
        self.SMMMoments = {"FE":FE_sim,
                           "FEVar":FEVar_sim,
                           "FEATV":FEATV_sim,
                           "Disg":Disg_sim,
                           "DisgVar":DisgVar_sim,
                           "DisgATV":DisgATV_sim,
                           "Var":Var_sim}
        return self.SMMMoments
    
#################################
######## New
################################

    def ProcessGMM(self):
        
        ## parameters and information set
        
        rho = self.process_para['rho']
        sigma = self.process_para['sigma']
        history = self.history
        
        resd = 0 
        yresd = 0 
        YVar = sigma**2/(1-rho**2)
        resdVar = sigma**2
        YATV = rho*YVar
        
        self.ProcessMoments= {#"resd":resd,
                              "Yresd":yresd,
                              "YVar":YVar,
                              "resdVar":resdVar,
                              "YATV":YATV}
        resd_data = history[1:]-rho*history[:-1]
        resd_data = np.mean(resd_data)
        resdVar_data = np.mean(resd**2)
        Yresd_data = np.mean(history[1:]*(history[1:]-rho*history[:-1]))
        YATV_data = np.cov(np.stack((history[1:],history[:-1]),axis=0))[0,1]
        YVar_data = np.var(history)
        
        
        self.ProcessDataMoments = {#"resd":resd_data,
                                  "Yresd":Yresd_data,
                                  "YVar": YVar_data,
                                  "YATV":YATV_data,
                                  "resdVar_data":resdVar_data}
        
#################################
######## specific to new moments
################################

    def NI_EstObjfuncGMM(self,
                         ni_paras):
        """
        input
        -----
        lbd: the parameter of SE model to be estimated
        
        output
        -----
        the objective function to minmize
        """
        moments = self.moments
        NI_para = {"sigma_pb":ni_paras[0],
                  "sigma_pr":ni_paras[1],
                  #'var_init':ni_paras[2],
                  # 'y_init':ni_paras[3],
                  # 'disg_init':ni_paras[4]
                  }

        self.exp_para = NI_para  # give the new parameters 
        data_moms_scalar_dct = self.data_moms_scalar_dct
        self.SteadyState()
        NI_moms_scalar_dct = self.GMM().copy()
        NI_moms_scalar = np.array([NI_moms_scalar_dct[key] for key in moments] )
        data_moms_scalar = np.array([data_moms_scalar_dct[key] for key in moments] )
        distance = NI_moms_scalar - data_moms_scalar
        obj_func = np.linalg.norm(distance)
        return obj_func 
    
#################################
######## specific to new moments
################################

    def NI_EstObjfuncSMM(self,
                         ni_paras):
        """
        input
        -----
        ni_paras: the parameter of NI model to be estimated
        
        output
        -----
        the objective function to minmize
        """
        moments = self.moments
        NI_para = {"sigma_pb":ni_paras[0],
                  "sigma_pr":ni_paras[1],
                  #'var_init':ni_paras[2],
                  # 'y_init':ni_paras[3],
                  # 'disg_init':ni_paras[4]
                  }

        self.exp_para = NI_para  # give the new parameters 
        data_moms_scalar_dct = self.data_moms_scalar_dct
        #self.SteadyState()
        NI_moms_scalar_dct = self.SMM().copy()
        NI_moms_scalar = np.array([NI_moms_scalar_dct[key] for key in moments] )
        data_moms_scalar = np.array([data_moms_scalar_dct[key] for key in moments] )
        distance = NI_moms_scalar - data_moms_scalar
        obj_func = np.linalg.norm(distance)
        return obj_func 
    
###################################
######## New
#####################################
    
    def NI_EstObjfuncSMMwm1st(self,
                              ni_paras):
        
        # estimate first step 
        self.ParaEstimateSMM()
        self.WM1stSMM()
        
        ## data moments
        data_moms_scalar_dct = self.data_moms_scalar_dct
        
        ## 1-step weighting matrix 
        wm1st = self.wm1st
        
        ## parameters 
        sigma_pb,sigma_pr = ni_paras
        NI_para = {"sigma_pb":sigma_pb,
                  "sigma_pr":sigma_pr}
        self.exp_para = NI_para  # give the new lambda
        
        NI_moms_scalar_dct = self.SMM().copy()
        
        sim_moms = np.array([NI_moms_scalar_dct[key] for key in self.moments])
        data_moms = np.array([data_moms_scalar_dct[key] for key in self.moments])
        assert len(sim_moms) == len(data_moms), "not equal lenghth"
        distance = sim_moms - data_moms
        tmp = np.multiply(np.multiply(distance.T,wm1st),distance)  ## need to make sure it is right. 
        obj_func = np.sum(tmp)
        return obj_func
    
###################################
######## New
#####################################

    def NI_EstObjfuncSMMwmboot(self,
                               ni_paras):
        
        # estimate first step 
        self.ParaEstimateSMM()
        self.WMbootSMM()
        
        ## data moments
        data_moms_scalar_dct = self.data_moms_scalar_dct
        
        ## 1-step weighting matrix 
        wm1st = self.wm_boot
        
        ## parameters 
        sigma_pb,sigma_pr = ni_paras
        NI_para = {"sigma_pb":sigma_pb,
                  "sigma_pr":sigma_pr}
        self.exp_para = NI_para  # give the new lambda
        
        NI_moms_scalar_dct = self.SMM().copy()
        
        sim_moms = np.array([NI_moms_scalar_dct[key] for key in self.moments])
        data_moms = np.array([data_moms_scalar_dct[key] for key in self.moments])
        assert len(sim_moms) == len(data_moms), "not equal lenghth"
        distance = sim_moms - data_moms
        tmp = np.multiply(np.multiply(distance.T,wm1st),distance)  ## need to make sure it is right. 
        obj_func = np.sum(tmp)
        return obj_func
    
    def PlotLossGMM(self,
                    sigma_pbs = np.linspace(0.01,2,10),
                    sigma_prs = np.linspace(0.01,2,10)):
        xx,yy = np.meshgrid(sigma_pbs,sigma_prs)
        paras = np.array([[xx[i],yy[i]] for i in range(len(xx))])
        loss = self.NI_EstObjfuncGMM(paras)
        self.fig = plt.contourf(sigma_pbs,sigma_prs,loss)
        plt.title('Loss function of GMM')
        return self.fig
    
    ## a function estimating SE model parameter only 
    def NI_EstObjfunc(self,
                      ni_paras):
        """
        input
        -----
        sigma: the parameters of NI model to be estimated. A vector of sigma_pb and sigma_pr
        
        output
        -----
        the objective function to minmize
        """
        moments = self.moments
        NI_para = {"sigma_pb":ni_paras[0],
                  "sigma_pr":ni_paras[1],
                  #'var_init':ni_paras[2],
                   #'y_init':ni_paras[3],
                   #'disg_init':ni_paras[4]
                  }
        self.exp_para = NI_para  # give the new parameters 
        data_moms_dct = self.data_moms_dct
        NI_moms_dct = self.Forecaster()
        NI_moms = np.array([NI_moms_dct[key] for key in moments] )
        data_moms = np.array([data_moms_dct[key] for key in moments] )
        distance = NI_moms - data_moms
        obj_func = np.linalg.norm(distance)
        return obj_func 
    
    def NI_EstObjfuncJoint(self,
                          paras):
        sigma_pb,sigma_pr,var_init,rho,sigma = paras
        moments = self.moments
        realized = self.realized

        process_para_joint = {'rho':rho,
                              'sigma':sigma}
        
        NI_para = {"sigma_pb":sigma_pb,
                  "sigma_pr":sigma_pr,
                  #'var_init':var_init
                  }
        
        self.exp_para = NI_para  # give the new lambda
        self.process_para = process_para_joint
        data_moms_dct = self.data_moms_dct
        sim_realized =  self.SimulateRealization()
        NI_moms_dct = self.Forecaster().copy()
        NI_moms = np.array([NI_moms_dct[key] for key in moments] )
        data_moms = np.array([data_moms_dct[key] for key in moments] )
        n = len(sim_realized)
        NI_moms_stack = np.concatenate((NI_moms, sim_realized.reshape(1,n)), axis=0)
        data_moms_stack = np.concatenate((data_moms, realized.reshape(1,n)), axis=0)
        distance = NI_moms_stack - data_moms_stack
        obj_func = np.linalg.norm(distance)
        return obj_func
    
    def NI_EstObjfuncSMMJoint(self,
                              paras):
        sigma_pb,sigma_pr,rho,sigma = paras
        moments = self.moments
        realized = self.realized
        
        process_para_joint = {'rho':rho,
                              'sigma':sigma}
        NI_para = {"sigma_pb":sigma_pb,
                  "sigma_pr":sigma_pr}
        
        self.exp_para = NI_para  # give the new lambda
        self.process_para = process_para_joint
        
        ## for the new parameters, update process GMM 
        self.ProcessGMM()
        ProcessDataMoments = self.ProcessDataMoments
        ProcessMoments = self.ProcessMoments
        
        ## get data and model moments conditions 
        
        data_moms_scalar_dct = self.data_moms_scalar_dct
        NI_moms_scalar_dct = self.SMM().copy()
        NI_moms_scalar = np.array([NI_moms_scalar_dct[key] for key in moments] )
        data_moms_scalar = np.array([data_moms_scalar_dct[key] for key in moments] )
        
        process_moms = np.array([ProcessMoments[key] for key in ProcessMoments.keys()])
        data_process_moms = np.array([ProcessDataMoments[key] for key in ProcessDataMoments.keys()])
        #print(ProcessMoments)
        #print(ProcessDataMoments)
        NI_moms_scalar_stack = np.concatenate((NI_moms_scalar,process_moms))
        data_moms_scalar_stack = np.concatenate((data_moms_scalar, data_process_moms))
        distance = NI_moms_scalar_stack - data_moms_scalar_stack
        obj_func = np.linalg.norm(distance)
        return obj_func
    
    ## feeds the instance with data moments dictionary 
    def GetDataMoments(self,
                       data_moms_dct):
        self.data_moms_dct = data_moms_dct
        
#################################
######## specific to new moments
################################

        data_moms_scalar_dct = dict(zip(data_moms_dct.keys(),
                                        [np.mean(data_moms_dct[key]) for key in data_moms_dct.keys()]
                                       )
                                   )
        data_moms_scalar_dct['FEVar'] = data_moms_dct['FE'].var()
        data_moms_scalar_dct['FEATV'] = np.cov(data_moms_dct['FE'][1:],data_moms_dct['FE'][:-1])[1,1]
        data_moms_scalar_dct['DisgVar'] = data_moms_dct['Disg'].var()
        data_moms_scalar_dct['DisgATV'] =np.cov(data_moms_dct['Disg'][1:],data_moms_dct['Disg'][:-1])[1,1]
        
        self.data_moms_scalar_dct = data_moms_scalar_dct
        
#################################
######## New
################################

    def ParaEstimateGMM(self,
                        para_guess=np.array([0.2,0.2]),
                        method='CG',
                        bounds = None,
                        options = None):
        self.para_estGMM = Estimator(self.NI_EstObjfuncGMM,
                                     para_guess = para_guess,
                                     method = method,
                                     bounds = bounds,
                                     options = options)
        return self.para_estGMM
        
#################################
######## New
################################

    def ParaEstimateSMM(self,
                        wb = 'identity',
                        para_guess = np.array([0.2,0.2]),
                        method='Nelder-Mead',
                        bounds = None,
                        options = None):
        if wb =='identity':
            self.para_estSMM = Estimator(self.NI_EstObjfuncSMM,
                                         para_guess = para_guess,
                                         method = method,
                                         bounds = bounds,
                                         options = options)
        elif wb =='2-step':
            self.para_estSMM = Estimator(self.NI_EstObjfuncSMMwm1st,
                                         para_guess = para_guess,
                                         method = method,
                                         bounds = bounds,
                                         options = options)
        elif wb =='bootstrap':
            self.para_estSMM = Estimator(self.NI_EstObjfuncSMMwmboot,
                                         para_guess = para_guess,
                                         method = method,
                                         bounds = bounds,
                                         options = options)
            
        return self.para_estSMM
        
#################################
######## New
################################

    def ParaEstimateSMMJoint(self,
                            para_guess=np.array([0.2,0.2,0.8,0.1]),
                            method='Nelder-Mead',
                            bounds = None,
                            options = None):
        self.para_est_SMM_joint = Estimator(self.NI_EstObjfuncSMMJoint,
                                             para_guess = para_guess,
                                             method = method,
                                             bounds = bounds,
                                             options = options)
        return self.para_est_SMM_joint
            
    ## invoke the estimator 
    def ParaEstimate(self,
                     para_guess=np.array([0.2,0.2]),
                     method='CG',
                     bounds = None,
                     options = None):
        self.para_est = Estimator(self.NI_EstObjfunc,
                                  para_guess = para_guess,
                                  method = method,
                                  bounds = bounds,
                                  options = options)
        return self.para_est
    
    def ParaEstimateJoint(self,
                          para_guess = (0.5,0.1),
                          method='BFGS',
                          bounds = None,
                          options = None):
        self.para_est_joint = Estimator(self.NI_EstObjfuncJoint,
                                  para_guess = para_guess,
                                  method = method,
                                  bounds = bounds,
                                  options = options)
    
    ## plot functions
    def ForecastPlot(self,
                     all_moms = False):
        plt.style.use('ggplot')
        if all_moms == False:
            moments_to_plot = self.moments
        else:
            moments_to_plot = self.all_moments
        m_ct = len(moments_to_plot)
        x = plt.figure(figsize=([3,3*m_ct]))
        for i,val in enumerate(moments_to_plot):
            plt.subplot(m_ct,1,i+1)
            plt.plot(self.forecast_moments[val],label=val)
            plt.legend(loc=1)
    
########################
###### New
########################

    ## diagostic plots 
    def ForecastPlotDiag(self,
                         all_moms = False,
                         diff_scale = False,
                         how ="GMMs"):
        if how =="GMMs":
            exp_para_est_dct = {'sigma_pb':self.para_est[0],
                               'sigma_pr':self.para_est[1]}
        elif how == "GMM":
            exp_para_est_dct = {'sigma_pb':self.para_estGMM[0],
                               'sigma_pr':self.para_estGMM[1]}
        elif how == "SMM":
            exp_para_est_dct = {'sigma_pb':self.para_estSMM[0],
                               'sigma_pr':self.para_estSMM[1]}
        elif how =="SMMs":
            exp_para_est_dct = {'sigma_pb':self.para_est_sim[0],
                               'sigma_pr':self.para_est_sim[1]}
        elif how =="SMMjoint":
            sigma_pb,sigma_pr,rho,sigma = self.para_est_SMM_joint
            exp_para_est_dct = {'sigma_pb':sigma_pb,
                               'sigma_pr':sigma_pr}
            process_para_est_dct = {'rho':rho,
                                   'sigma':sigma}
        elif how =="GMMsjoint":
            sigma_pb,sigma_pr,rho,sigma = self.para_est_joint
            exp_para_est_dct = {'sigma_pb':sigma_pb,
                               'sigma_pr':sigma_pr}
            process_para_est_dct = {'rho':rho,
                                   'sigma':sigma} 
        
        new_instance = cp.deepcopy(self)
        new_instance.exp_para = exp_para_est_dct
        self.forecast_moments_est = new_instance.ForecasterbySim()
        plt.style.use('ggplot')
        if all_moms == False:
            moments_to_plot = self.moments
        else:
            moments_to_plot = self.all_moments
            
        m_ct = len(moments_to_plot)
        
        x = plt.figure(figsize=([3,3*m_ct]))
        if diff_scale == False:
            for i,val in enumerate(moments_to_plot):
                plt.subplot(m_ct,1,i+1)
                plt.plot(self.forecast_moments_est[val],'s-',label='model:'+ val)
                plt.plot(np.array(self.data_moms_dct[val]),'o-',label='data:'+ val)
                plt.legend(loc=1)
        if diff_scale == True:
            for i,val in enumerate(moments_to_plot):
                ax1 = plt.subplot(m_ct,1,i+1)
                ax1.plot(self.forecast_moments_est[val],'rs-',label='model:'+ val)
                ax1.legend(loc=0)
                ax2 = ax1.twinx()
                ax2.plot(np.array(self.data_moms_dct[val]),'o-',color='steelblue',label='(RHS) data:'+ val)
                ax2.legend(loc=3)
                
                
#################################
######## New
################################
                
    def WM1stSMM(self):
        """
        - get the 1-st step variance and covariance matrix
        """
        exp_para_est_dct = {'sigma_pb':self.para_estSMM[0],
                           'sigma_pr':self.para_estSMM[1]}
        self.exp_para = exp_para_est_dct
        
        sim_moms_dct = self.SMM()
        data_moms_scalar = self.data_moms_scalar_dct
        sim_moms = np.array([sim_moms_dct[key] for key in self.moments])
        data_moms = np.array([data_moms_scalar[key] for key in self.moments])
        distance = sim_moms - data_moms
        distance_diag = np.diag(distance*distance.T)
        self.wm1st = np.linalg.inv(distance_diag)
        return self.wm1st
    
    def WMbootSMM(self,
                  n_boot = 100):
        # data moments 
        data_moms_scalar = self.data_moms_scalar_dct
        
        # parameters 
        exp_para_est_dct = {'sigma_pb':self.para_estSMM[0],
                           'sigma_pr':self.para_estSMM[1]}
        self.exp_para = exp_para_est_dct
        
        distance_boot = []
        
        for i in range(n_boot): 
            self.SMM()
            sim_moms_dct = self.SMMMoments
            sim_moms = np.array([sim_moms_dct[key] for key in self.moments])
            data_moms = np.array([data_moms_scalar[key] for key in self.moments])
            distance = sim_moms - data_moms
            distance_boot.append(np.array(distance))
        #print(np.array(distance_boot).shape)
        vcv_boot_ary = np.array(distance_boot, dtype=np.float64)
        self.vcv_boot = np.cov(vcv_boot_ary.T)
        #print(self.vcv_boot)
        self.wm_boot = np.linalg.inv(self.vcv_boot) 
# -
DE_para_default = {'theta':0.1,
                  'theta_sigma':0.1}

# + {"code_folding": [4, 24, 28, 45, 79, 120, 188, 218, 259, 289, 314, 344, 362, 446, 463]}
## Diagnostic Expectation(DE) class


class DiagnosticExpectation:
    def __init__(self,
                 real_time,
                 history,
                 horizon = 1,
                 process_para = process_para,
                 exp_para = DE_para_default,
                 moments = ['Forecast','Disg','Var']):
        self.history = history
        self.real_time = real_time
        self.n = len(real_time)
        self.horizon = horizon
        self.process_para = process_para
        self.exp_para = exp_para
        self.data_moms_dct ={}
        self.para_est = {}
        self.moments = moments
        self.all_moments = ['Forecast','FE','Disg','Var']
        self.realized = None
        self.sim_realized = None
        
    def GetRealization(self,
                       realized_series):
        self.realized = realized_series 
    
    def SimulateRealization(self):
        n = self.n
        rho = self.process_para['rho']
        sigma = self.process_para['sigma']
        np.random.seed(12345)
        shocks = np.random.randn(n)*sigma
        sim_realized = np.zeros(n)
        for i in range(n):
            cum_shock = sum([rho**h*shocks[h] for h in range(self.horizon)])
            sim_realized[i] = rho**self.horizon*self.real_time[i] + cum_shock
        self.sim_realized = sim_realized
        return self.sim_realized 
    
#################################
######## specific to new moments
################################

    def SMM(self):
            
        ## simulate forecasts 
        
        self.ForecasterbySim(n_sim = 200)
        moms_sim = self.forecast_moments_sim
        
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
    
        self.SMMMoments = {"FE":FE_sim,
                           "FEVar":FEVar_sim,
                           "FEATV":FEATV_sim,
                           "Disg":Disg_sim,
                           "DisgVar":DisgVar_sim,
                           "DisgATV":DisgATV_sim,
                           "Var":Var_sim}
        return self.SMMMoments
    
#################################
######## New
################################

    def ProcessGMM(self):
        
        ## parameters and information set

        ## model 
        rho = self.process_para['rho']
        sigma = self.process_para['sigma']
        history = self.history
        
        resd = 0 
        Yresd = 0 
        resdVar = sigma**2
        YVar = sigma**2/(1-rho**2)
        YATV = rho*YVar
        
        
        self.ProcessMoments= {"Yresd":Yresd,
                              "YVar":YVar,
                              "resdVar":resdVar,
                              #"YATV":YATV
                             }
        ## data 
        resds = np.mean(history[1:]-rho*history[:-1])
        resd_data = np.mean(resds)
        resdVar_data = np.mean(resds**2)
        Yresd_data = np.mean(history[:-1]*resds)
        YVar_data = np.var(history)
        YATV_data = np.cov(np.stack((history[1:],history[:-1]),axis=0))[0,1]
        
        
        self.ProcessDataMoments = {"Yresd":Yresd_data,
                                  "YVar": YVar_data,
                                  "resdVar":resdVar_data,
                                   #"YATV":YATV_data
                                  }
        
        
###############
### need to change 
########################

    def ForecasterbySim(self,
                        n_sim = 100):
        ## inputs 
        real_time = self.real_time
        history  = self.history
        realized = self.realized
        sim_realized = self.sim_realized
        n = len(real_time)
        rho = self.process_para['rho']
        sigma =self.process_para['sigma']
        theta = self.exp_para['theta']
        theta_sigma = self.exp_para['theta_sigma']
        horizon = self.horizon 
        n_burn = len(history) - n
        n_history = n + n_burn  # of course equal to len(history)
    
        
        ## simulation
        np.random.seed(12345)
        thetas = theta_sigma*np.random.randn(n_sim) + theta  ## randomly drawn representativeness parameters
        
        nowcasts_to_burn = np.empty([n_sim,n_history])
        Vars_to_burn = np.empty([n_sim,n_history])
        nowcasts_to_burn[:,0] = history[0]
        Vars_to_burn[:,:] = hstepvar(horizon,sigma,rho)
        
        ## diagnostic and extrapolative expectations 
        for i in range(n_sim):
            this_theta = thetas[i]
            for j in range(n_history-1):
                nowcasts_to_burn[i,j+1] = history[j+1]+ this_theta*(history[j+1]-rho*history[j])  # can be nowcasting[j-1] instead
        
        ## burn initial forecasts since history is too short 
        nowcasts = np.array( nowcasts_to_burn[:,n_burn:] )
        forecasts = rho**horizon*nowcasts
        Vars = np.array( Vars_to_burn[:,n_burn:])
        
        if realized is not None:
            FEs = forecasts - realized
        elif self.sim_realized is not None:
            FEs = forecasts - self.sim_realized
        
        ## compuate population moments
        forecasts_mean = np.mean(forecasts,axis = 0)
        forecasts_var = np.var(forecasts,axis = 0)
        
        if realized is not None:
            FEs_mean = forecasts_mean - realized
        elif self.sim_realized is not None:
            FEs_mean = forecasts_mean - self.sim_realized
            
        Vars_mean = np.mean(Vars,axis = 0) ## need to change 
        
        forecasts_vcv = np.cov(forecasts.T)
        forecasts_atv = np.array([forecasts_vcv[i+1,i] for i in range(n-1)])
        FEs_vcv = np.cov(FEs.T)
        FEs_atv = np.array([FEs_vcv[i+1,i] for i in range(n-1)]) ## this is no longer needed
        
        self.forecast_moments_sim = {"Forecast":forecasts_mean,
                                     "FE":FEs_mean,
                                     "Disg":forecasts_var,
                                     "Var":Vars_mean}
        return self.forecast_moments_sim
      
###################################
######## New
#####################################

    def DE_EstObjfuncSMM(self,
                         paras):
        """
        input
        -----
        theta: the parameter of DE model to be estimated
        theta_sigma: the dispersion of representation parameter
        output
        -----
        the objective function to minmize
        """
        moments = self.moments
        data_moms_scalar_dct = self.data_moms_scalar_dct
        
        DE_para = {"theta":paras[0],
                  "theta_sigma":paras[1]}
        self.exp_para = DE_para  
        
        DE_moms_scalar_dct = self.SMM().copy()
        DE_moms_scalar = np.array([DE_moms_scalar_dct[key] for key in moments] )
        data_moms_scalar = np.array([data_moms_scalar_dct[key] for key in moments] )
        distance = DE_moms_scalar - data_moms_scalar
        obj_func = np.linalg.norm(distance)
        return obj_func 
    
    
###################################
######## New
#####################################

    def DE_EstObjfuncSMMJoint(self,
                              paras):
        theta,theta_sigma,rho,sigma = paras
        moments = self.moments
        realized = self.realized
        
        process_para_joint = {'rho':rho,
                              'sigma':sigma}
        DE_para = {"theta":theta,
                  "theta_sigma":theta_sigma}
        
        self.exp_para = DE_para  # give the new thetas
        self.process_para = process_para_joint
        
        ## for the new parameters, update process GMM 
        self.ProcessGMM()
        ProcessDataMoments = self.ProcessDataMoments
        ProcessMoments = self.ProcessMoments
        
        ## get data and model moments conditions 
        
        data_moms_scalar_dct = self.data_moms_scalar_dct
        DE_moms_scalar_dct = self.SMM().copy()
        DE_moms_scalar = np.array([DE_moms_scalar_dct[key] for key in moments] )
        data_moms_scalar = np.array([data_moms_scalar_dct[key] for key in moments] )
        
        process_moms = np.array([ProcessMoments[key] for key in ProcessMoments.keys()])
        data_process_moms = np.array([ProcessDataMoments[key] for key in ProcessDataMoments.keys()])
        #print(ProcessMoments)
        #print(ProcessDataMoments)
        DE_moms_scalar_stack = np.concatenate((DE_moms_scalar,process_moms))
        data_moms_scalar_stack = np.concatenate((data_moms_scalar, data_process_moms))
        distance = DE_moms_scalar_stack - data_moms_scalar_stack
        obj_func = np.linalg.norm(distance)
        return obj_func
    
    
###################################
######## New
###################################
    
    def DE_EstObjfuncSMMwm1st(self,
                              paras):
        
        # estimate first step 
        self.ParaEstimateSMM()
        self.WM1stSMM()
        
        ## data moments
        data_moms_scalar_dct = self.data_moms_scalar_dct
        
        ## 1-step weighting matrix 
        wm1st = self.wm1st
        
        ## parameters 
        DE_para = {"theta":paras[0],
                  "theta_sigma":paras[1]}
        self.exp_para = SE_para  # give the new lambda
        
        SE_moms_scalar_dct = self.SMM().copy()
        
        sim_moms = np.array([SE_moms_scalar_dct[key] for key in self.moments])
        data_moms = np.array([data_moms_scalar_dct[key] for key in self.moments])
        assert len(sim_moms) == len(data_moms), "not equal lenghth"
        distance = sim_moms - data_moms
        tmp = np.multiply(np.multiply(distance.T,wm1st),distance)  ## need to make sure it is right. 
        obj_func = np.sum(tmp)
        return obj_func
    
    
    ## feeds the instance with data moments dictionary 
    def GetDataMoments(self,
                       data_moms_dct):
        self.data_moms_dct = data_moms_dct
        
#################################
######## New
################################

        data_moms_scalar_dct = dict(zip(data_moms_dct.keys(),
                                        [np.mean(data_moms_dct[key]) for key in data_moms_dct.keys()]
                                       )
                                   )
        data_moms_scalar_dct['FEVar'] = data_moms_dct['FE'].var()
        FE_stack = np.stack((data_moms_dct['FE'][1:],data_moms_dct['FE'][:-1]),axis = 0)
        data_moms_scalar_dct['FEATV'] = np.cov(FE_stack)[0,1]
        data_moms_scalar_dct['DisgVar'] = data_moms_dct['Disg'].var()
        Disg_stack = np.stack((data_moms_dct['Disg'][1:],data_moms_dct['Disg'][:-1]),axis = 0)
        data_moms_scalar_dct['DisgATV'] = np.cov(Disg_stack)[0,1]
        
        self.data_moms_scalar_dct = data_moms_scalar_dct
        
###################################
######## New
####################################
        
    def ParaEstimateSMM(self,
                        wb = 'identity',
                        para_guess = 0.2,
                        method='Nelder-Mead',
                        bounds = None,
                        options = None):
        if wb == 'identity':
            self.para_estSMM = Estimator(self.DE_EstObjfuncSMM,
                                         para_guess = para_guess,
                                         method = method,
                                         bounds = bounds,
                                         options = options)
        elif wb =='2-step':
            self.para_estSMM = Estimator(self.DE_EstObjfuncSMMwm1st,
                                         para_guess = para_guess,
                                         method = method,
                                         bounds = bounds,
                                         options = options)
        elif wb =='bootstrap':
            self.para_estSMM = Estimator(self.DE_EstObjfuncSMMwmboot,
                                         para_guess = para_guess,
                                         method = method,
                                         bounds = bounds,
                                         options = options)
        return self.para_estSMM
    
###################################
######## New
#####################################
    
    def ParaEstimateSMMJoint(self,
                             para_guess = (0.5,0.8,0.2),
                             method='Nelder-Mead',
                             bounds = None,
                             options = None):
        self.para_est_SMM_joint = Estimator(self.DE_EstObjfuncSMMJoint,
                                            para_guess = para_guess,
                                            method = method,
                                            bounds = bounds,
                                            options = options)
        return self.para_est_SMM_joint
    
    
###################################
######## New
#####################################


    def ForecastPlot(self,
                     all_moms = False):
        plt.style.use('ggplot')
        if all_moms == False:
            m_ct = len(self.moments)
            x = plt.figure(figsize=([3,3*m_ct]))
            for i,val in enumerate(self.moments):
                plt.subplot(m_ct,1,i+1)
                plt.plot(self.forecast_moments[val],label=val)
                plt.legend(loc=1)
        else:
            m_ct = len(self.all_moments)
            x = plt.figure(figsize=([3,3*m_ct]))
            for i,val in enumerate(self.all_moments):
                plt.subplot(m_ct,1,i+1)
                plt.plot(self.forecast_moments[val],label=val)
                plt.legend(loc=1)
                
                
#################################
######## New
################################

    def ForecastPlotDiag(self,
                         all_moms = False,
                         diff_scale = False,
                         how = 'GMMs'):
        if how =='GMMs':
            exp_para_est_dct = {'theta':self.para_est[0],
                               'theta_sigma': self.para_est[1]}
        elif how == "GMM":
            exp_para_est_dct = {'theta':self.para_estGMM[0],
                               'theta_sigma': self.para_estGMM[1]}
        elif how == "SMM":
            exp_para_est_dct = {'theta':self.para_estSMM[0],
                               'theta_sigma': self.para_estSMM[1]}
        elif how =="SMMs":
            exp_para_est_dct = {'theta':self.para_est_sim[0],
                               'theta_sigma': self.para_est_sim[1]}
        elif how =="SMMjoint":
            theta,theta_sigma,rho,sigma = self.para_est_SMM_joint
            exp_para_est_dct = {'theta':theta,
                               'theta_sigma':theta_sigma}
            process_para_est_dct = {'rho':rho,
                                   'sigma':sigma}
        elif how =="GMMsjoint":
            theta,theta_sigma,rho,sigma = self.para_est_joint
            exp_para_est_dct = {'theta':theta,
                               'theta_sigma':theta_sigma}
            process_para_est_dct = {'rho':rho,
                                   'sigma':sigma}
            
        ## plot 
        new_instance = cp.deepcopy(self)
        new_instance.exp_para = exp_para_est_dct
        self.forecast_moments_est = new_instance.ForecasterbySim()
        plt.style.use('ggplot')
        if all_moms == False:
            moments_to_plot = self.moments
        else:
            moments_to_plot = self.all_moments

        m_ct = len(moments_to_plot)
        x = plt.figure(figsize=([3,3*m_ct]))
        if diff_scale == False:
            for i,val in enumerate(moments_to_plot):
                plt.subplot(m_ct,1,i+1)
                plt.plot(self.forecast_moments_est[val][:-1],'s-',label='model:'+ val)
                plt.plot(np.array(self.data_moms_dct[val][:-1]),'o-',label='data:'+ val)
                plt.legend(loc = 1)
        if diff_scale == True:
            for i,val in enumerate(moments_to_plot):
                ax1 = plt.subplot(m_ct,1,i+1)
                ax1.plot(self.forecast_moments_est[val][:-1],'rs-',label='model:'+ val)
                ax1.legend(loc=0)
                ax2 = ax1.twinx()
                ax2.plot(np.array(self.data_moms_dct[val][:-1]),'o-',color='steelblue',label='(RHS) data:'+ val)
                ax2.legend(loc=3)
                
                
#################################
######## New
################################
                
    def WM1stSMM(self):
        """
        - get the 1-st step variance and covariance matrix
        """
        exp_para_est_dct = {'theta':self.para_estSMM[0],
                           'theta_sigma':self.para_estSMM[1]}
        self.exp_para = exp_para_est_dct
        
        sim_moms_dct = self.SMM()
        data_moms_scalar_dct = self.data_moms_scalar_dct
        sim_moms = np.array([sim_moms_dct[key] for key in self.moments])
        data_moms = np.array([data_moms_scalar_dct[key] for key in self.moments])
        distance = sim_moms - data_moms
        distance_diag = np.diag(distance*distance.T)
        self.wm1st = np.linalg.inv(distance_diag)
        return self.wm1st
    
    def WMbootSMM(self,
                  n_boot = 100):
        # data moments 
        data_moms_scalar_dct = self.data_moms_scalar_dct
        
        # parameters 
        exp_para_est_dct = {'theta':self.para_estSMM[0],
                           'theta_sigma':self.para_estSMM[1]}
        self.exp_para = exp_para_est_dct
        
        distance_boot = []
        
        for i in range(n_boot): 
            self.SMM()
            sim_moms_dct = self.SMMMoments
            sim_moms = np.array([sim_moms_dct[key] for key in self.moments])
            data_moms = np.array([data_moms_scalar_dct[key] for key in self.moments])
            distance = sim_moms - data_moms
            distance_boot.append(np.array(distance))
        #print(np.array(distance_boot).shape)
        vcv_boot_ary = np.array(distance_boot, dtype=np.float64)
        self.vcv_boot = np.cov(vcv_boot_ary.T)
        #print(self.vcv_boot)
        self.wm_boot = np.linalg.inv(self.vcv_boot)               

# -

SENI_para_default = {'lambda':0.1,
                     'sigma_pb':0.1,
                     'sigma_pr':0.1}

# + {"code_folding": [30, 44, 61, 95, 245, 276, 316, 350, 382, 407, 438, 522, 540]}
## Hybrid SENI Class 

## Diagnostic Expectation(DE) class


class StickyNoisyHybrid:
    def __init__(self,
                 real_time,
                 history,
                 horizon = 1,
                 process_para = process_para,
                 exp_para = SENI_para_default,
                 moments = ['Forecast','Disg','Var']):
        self.history = history
        self.real_time = real_time
        self.n = len(real_time)
        self.horizon = horizon
        self.process_para = process_para
        self.exp_para = exp_para
        self.data_moms_dct ={}
        self.para_est = {}
        self.moments = moments
        self.all_moments = ['Forecast','FE','Disg','Var']
        self.realized = None
        self.sim_realized = None
        
    def GetRealization(self,
                       realized_series):
        self.realized = realized_series 
    
    def SimulateRealization(self):
        n = self.n
        rho = self.process_para['rho']
        sigma = self.process_para['sigma']
        np.random.seed(12345)
        shocks = np.random.randn(n)*sigma
        sim_realized = np.zeros(n)
        for i in range(n):
            cum_shock = sum([rho**h*shocks[h] for h in range(self.horizon)])
            sim_realized[i] = rho**self.horizon*self.real_time[i] + cum_shock
        self.sim_realized = sim_realized
        return self.sim_realized 
    
    
    def SimulateSignals(self):
        n = self.n
        n_history = len(self.history)
        sigma_pb = self.exp_para['sigma_pb']
        sigma_pr =self.exp_para['sigma_pr']
        np.random.seed(1234)
        s_pb = self.history + sigma_pb*np.random.randn(n_history)
        np.random.seed(12343)
        s_pr = self.history + sigma_pr*np.random.randn(n_history)
        self.signals = np.asmatrix(np.array([s_pb,s_pr]))
        self.signals_pb = s_pb
      
    
#################################
######## specific to new moments
################################

    def SMM(self):
            
        ## simulate forecasts 
        
        self.ForecasterbySim(n_sim = 200)
        moms_sim = self.forecast_moments_sim
        
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
    
        self.SMMMoments = {"FE":FE_sim,
                           "FEVar":FEVar_sim,
                           "FEATV":FEATV_sim,
                           "Disg":Disg_sim,
                           "DisgVar":DisgVar_sim,
                           "DisgATV":DisgATV_sim,
                           "Var":Var_sim}
        return self.SMMMoments
    
#################################
######## New
################################

    def ProcessGMM(self):
        
        ## parameters and information set

        ## model 
        rho = self.process_para['rho']
        sigma = self.process_para['sigma']
        history = self.history
        
        resd = 0 
        Yresd = 0 
        resdVar = sigma**2
        YVar = sigma**2/(1-rho**2)
        YATV = rho*YVar
        
        
        self.ProcessMoments= {"Yresd":Yresd,
                              "YVar":YVar,
                              "resdVar":resdVar,
                              #"YATV":YATV
                             }
        ## data 
        resds = np.mean(history[1:]-rho*history[:-1])
        resd_data = np.mean(resds)
        resdVar_data = np.mean(resds**2)
        Yresd_data = np.mean(history[:-1]*resds)
        YVar_data = np.var(history)
        YATV_data = np.cov(np.stack((history[1:],history[:-1]),axis=0))[0,1]
        
        
        self.ProcessDataMoments = {"Yresd":Yresd_data,
                                  "YVar": YVar_data,
                                  "resdVar":resdVar_data,
                                   #"YATV":YATV_data
                                  }
        

#########################################################################
######## New. Need to change
###########################################################################

    def ForecasterbySim(self,
                        n_sim = 100):
        ## inputs 
        real_time = self.real_time
        history  = self.history
        realized = self.realized
        sim_realized = self.sim_realized
        n = len(real_time)
        rho = self.process_para['rho']
        sigma = self.process_para['sigma']
        
        #############################################
        lbd = self.exp_para['lambda']
        sigma_pb = self.exp_para['sigma_pb']
        sigma_pr = self.exp_para['sigma_pr']
        ############################################
        
        horizon = self.horizon 
        n_burn = len(history) - n
        n_history = n + n_burn  # of course equal to len(history)
        
        #######################
        var_init = 10
        ##################
        sigma_v = np.asmatrix([[sigma_pb**2,0],[0,sigma_pr**2]])
        horizon = self.horizon 
        
        
        ## simulate signals 
        self.SimulateSignals()
        signals = self.signals
        nb_s = len(self.signals) ## # of signals 
        H = np.asmatrix ([[1,1]]).T
        
        # randomly simulated signals 
        signal_pb = self.signals_pb 
        np.random.seed(12434)
        signals_pr = np.array([self.history]) + sigma_pr*np.random.randn(n_sim*n_history).reshape([n_sim,n_history])
        
        ## SE governs if updating for each agent at each point of time 
        ## simulation of updating profile
        np.random.seed(12345)
        update_or_not = bernoulli.rvs(lbd,size = [n_sim,n_history])
        most_recent_when = np.empty([n_sim,n_history],dtype = int)
        nowsignals_pb_to_burn = np.empty([n_sim,n_history])
        nowsignals_pr_to_burn = np.empty([n_sim,n_history])
        #nowcasts_to_burn = np.empty([n_sim,n_history])
        #Vars_to_burn = np.empty([n_sim,n_history])
        
        ## look back for the most recent last update for each point of time  
        for i in range(n_sim):
            for j in range(n_history):
                if np.any([x for x in range(j) if update_or_not[i,j-x] == 1]):
                    most_recent_when[i,j] = np.min([x for x in range(j) if update_or_not[i,j-x] == 1])
                else:
                    most_recent_when[i,j] = j
                ################################################################################
                nowsignals_pr_to_burn[i,j] = signal_pb[j - most_recent_when[i,j]]
                nowsignals_pr_to_burn[i,j] = signals_pr[i,j - most_recent_when[i,j]]
                ## both above are the matrices of signals available to each agent depending on if updating
                #####################################################################################
        
        ## Once sticky signals are prepared. Then agent filter as NI
        ## prepare matricies 
        nowcasts_to_burn = np.zeros([n_sim,n_history])
        nowcasts_to_burn[:,0] = history[0]
        nowvars_to_burn = np.zeros([n_sim,n_history])
        nowvars_to_burn[:,0] = var_init
        Vars_to_burn = np.zeros([n_sim,n_history])
        
        
        ## fill the matricies for individual moments        
        for i in range(n_sim):
            signals_this_i = np.asmatrix(np.array([nowsignals_pb_to_burn[i,:],nowsignals_pr_to_burn[i,:]]))
            Pkalman = np.zeros([n_history,nb_s])
            Pkalman[0,:] = 0 
            for t in range(n_history-1):
                step1_vars_to_burn = rho**2*nowvars_to_burn[i,t] + sigma**2
                nowvars_to_burn[i,t+1] = step1_vars_to_burn - step1_vars_to_burn*\
                                          H.T*np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v)*H*step1_vars_to_burn
                Pkalman[t+1,:] = step1_vars_to_burn*H.T*np.linalg.inv(H*step1_vars_to_burn*H.T+sigma_v)
                nowcasts_to_burn[i,t+1] = (1-Pkalman[t+1,:]*H)*rho*nowcasts_to_burn[i,t]+ Pkalman[t+1,:]*signals_this_i[:,t+1]
            for t in range(n_history):
                Vars_to_burn[i,t] = rho**(2*horizon)*nowvars_to_burn[i,t] + hstepvar(horizon,sigma,rho)
                
        nowcasts = nowcasts_to_burn[:,n_burn:]
        forecasts = rho**horizon*nowcasts 
        Vars = Vars_to_burn[:,n_burn:]
        
        ## compuate population moments
        forecasts_mean = np.mean(forecasts,axis=0)
        forecasts_var = np.var(forecasts,axis=0)
        if realized is not None:
            FEs_mean = forecasts_mean - realized
        elif sim_realized is not None:
            FEs_mean = forecasts_mean - sim_realized
            
        Vars_mean = np.mean(Vars,axis=0) ## need to change for time-variant volatility
        
        self.forecast_moments_sim = {"Forecast":forecasts_mean,
                                     "FE":FEs_mean,
                                     "Disg":forecasts_var,
                                     "Var":Vars_mean}
        return self.forecast_moments_sim
    
###################################
######## New
#####################################

    def SENI_EstObjfuncSMM(self,
                         paras):
        """
        input
        -----
        lbd: the parameter of SE model to be estimated
        
        output
        -----
        the objective function to minmize
        """
        moments = self.moments
        data_moms_scalar_dct = self.data_moms_scalar_dct
        
        lbd,sigma_pb,sigma_pr = paras
        paras = {"lambda":lbd,
                  'sigma_pb':sigma_pb,
                  'sigma_pr':sigma_pr}
        self.exp_para = paras  # give the new lambda
        
        SENI_moms_scalar_dct = self.SMM().copy()
        SENI_moms_scalar = np.array([SENI_moms_scalar_dct[key] for key in moments] )
        data_moms_scalar = np.array([data_moms_scalar_dct[key] for key in moments] )
        distance = SENI_moms_scalar - data_moms_scalar
        obj_func = np.linalg.norm(distance)
        return obj_func 
    
###################################
######## New
#####################################

    def SENI_EstObjfuncSMMJoint(self,
                              paras):
        lbd,sigma_pb,sigma_pr,rho,sigma = paras
        moments = self.moments
        realized = self.realized
        
        process_para_joint = {'rho':rho,
                              'sigma':sigma}
        paras = {'lambda':lbd,
                'sigma_pb':sigma_pb,
                'sigma_pr':sigma_pr}
        self.exp_para = paras  # give the new lambda
        self.process_para = process_para_joint
        
        ## for the new parameters, update process GMM 
        self.ProcessGMM()
        ProcessDataMoments = self.ProcessDataMoments
        ProcessMoments = self.ProcessMoments
        
        ## get data and model moments conditions 
        
        data_moms_scalar_dct = self.data_moms_scalar_dct
        SENI_moms_scalar_dct = self.SMM().copy()
        SENI_moms_scalar = np.array([SENI_moms_scalar_dct[key] for key in moments] )
        data_moms_scalar = np.array([data_moms_scalar_dct[key] for key in moments] )
        
        process_moms = np.array([ProcessMoments[key] for key in ProcessMoments.keys()])
        data_process_moms = np.array([ProcessDataMoments[key] for key in ProcessDataMoments.keys()])
        #print(ProcessMoments)
        #print(ProcessDataMoments)
        SENI_moms_scalar_stack = np.concatenate((SENI_moms_scalar,process_moms))
        data_moms_scalar_stack = np.concatenate((data_moms_scalar, data_process_moms))
        distance = SENI_moms_scalar_stack - data_moms_scalar_stack
        obj_func = np.linalg.norm(distance)
        return obj_func
    
###################################
######## New
####################################
    
    def SENI_EstObjfuncSMMwm1st(self,
                                paras):
        
        # estimate first step 
        self.ParaEstimateSMM()
        self.WM1stSMM()
        
        ## data moments
        data_moms_scalar_dct = self.data_moms_scalar_dct
        
        ## 1-step weighting matrix 
        wm1st = self.wm1st
        
        ## parameters 
        lbd,sigma_pb,sigma_pr = paras
        paras = {'lambda':lbd,
                'sigma_pb':sigma_pb,
                'sigma_pr':sigma_pr}
        self.exp_para = paras  # give the new lambda
        
        SE_moms_scalar_dct = self.SMM().copy()
        
        sim_moms = np.array([SENI_moms_scalar_dct[key] for key in self.moments])
        data_moms = np.array([data_moms_scalar_dct[key] for key in self.moments])
        assert len(sim_moms) == len(data_moms), "not equal lenghth"
        distance = sim_moms - data_moms
        tmp = np.multiply(np.multiply(distance.T,wm1st),distance)  ## need to make sure it is right. 
        obj_func = np.sum(tmp)
        return obj_func
    
###################################
######## New
#####################################

    def SENI_EstObjfuncSMMwmboot(self,
                                 paras):
        
        # estimate first step 
        self.ParaEstimateSMM()
        self.WMbootSMM()
        
        ## data moments
        data_moms_scalar_dct = self.data_moms_scalar_dct
        
        ## 1-step weighting matrix 
        wm1st = self.wm_boot
        
        ## parameters 
        lbd,sigma_pb,sigma_pr = paras
        paras = {'lambda':lbd,
                'sigma_pb':sigma_pb,
                'sigma_pr':sigma_pr}
        self.exp_para = SE_para  # give the new lambda
        
        SENI_moms_scalar_dct = self.SMM().copy()
        
        sim_moms = np.array([SENI_moms_scalar_dct[key] for key in self.moments])
        data_moms = np.array([data_moms_scalar_dct[key] for key in self.moments])
        assert len(sim_moms) == len(data_moms), "not equal lenghth"
        distance = sim_moms - data_moms
        tmp = np.multiply(np.multiply(distance.T,wm1st),distance)  ## need to make sure it is right. 
        obj_func = np.sum(tmp)
        return obj_func
    
## feeds the instance with data moments dictionary 
    
    def GetDataMoments(self,
                       data_moms_dct):
        self.data_moms_dct = data_moms_dct
        
#################################
######## New
################################

        data_moms_scalar_dct = dict(zip(data_moms_dct.keys(),
                                        [np.mean(data_moms_dct[key]) for key in data_moms_dct.keys()]
                                       )
                                   )
        data_moms_scalar_dct['FEVar'] = data_moms_dct['FE'].var()
        FE_stack = np.stack((data_moms_dct['FE'][1:],data_moms_dct['FE'][:-1]),axis = 0)
        data_moms_scalar_dct['FEATV'] = np.cov(FE_stack)[0,1]
        data_moms_scalar_dct['DisgVar'] = data_moms_dct['Disg'].var()
        Disg_stack = np.stack((data_moms_dct['Disg'][1:],data_moms_dct['Disg'][:-1]),axis = 0)
        data_moms_scalar_dct['DisgATV'] = np.cov(Disg_stack)[0,1]
        
        self.data_moms_scalar_dct = data_moms_scalar_dct
        
#################################
######## New
################################

    def ParaEstimateSMM(self,
                        wb = 'identity',
                        para_guess = np.array([0.2,0.2]),
                        method='Nelder-Mead',
                        bounds = None,
                        options = None):
        if wb =='identity':
            self.para_estSMM = Estimator(self.SENI_EstObjfuncSMM,
                                         para_guess = para_guess,
                                         method = method,
                                         bounds = bounds,
                                         options = options)
        elif wb =='2-step':
            self.para_estSMM = Estimator(self.SENI_EstObjfuncSMMwm1st,
                                         para_guess = para_guess,
                                         method = method,
                                         bounds = bounds,
                                         options = options)
        elif wb =='bootstrap':
            self.para_estSMM = Estimator(self.SENI_EstObjfuncSMMwmboot,
                                         para_guess = para_guess,
                                         method = method,
                                         bounds = bounds,
                                         options = options)
            
        return self.para_estSMM
        
#################################
######## New
################################

    def ParaEstimateSMMJoint(self,
                            para_guess=np.array([0.2,0.2,0.8,0.1]),
                            method='Nelder-Mead',
                            bounds = None,
                            options = None):
        self.para_est_SMM_joint = Estimator(self.SENI_EstObjfuncSMMJoint,
                                             para_guess = para_guess,
                                             method = method,
                                             bounds = bounds,
                                             options = options)
        return self.para_est_SMM_joint
    
    
########################
###### New
########################

    ## diagostic plots 
    def ForecastPlotDiag(self,
                         all_moms = False,
                         diff_scale = False,
                         how ="GMMs"):
        if how =="GMMs":
            exp_para_est_dct = {'lambda':self.para_est[0],
                                'sigma_pb':self.para_est[1],
                               'sigma_pr':self.para_est[2]}
        elif how == "GMM":
            exp_para_est_dct = {'lambda':self.para_estGMM[0],
                                'sigma_pb':self.para_estGMM[1],
                               'sigma_pr':self.para_estGMM[2]}
        elif how == "SMM":
            exp_para_est_dct = {'lambda':self.para_estSMM[0],
                                'sigma_pb':self.para_estSMM[1],
                               'sigma_pr':self.para_estSMM[2]}
        elif how =="SMMs":
            exp_para_est_dct = {'lambda':self.para_est_sim[0],
                                'sigma_pb':self.para_est_sim[1],
                               'sigma_pr':self.para_est_sim[2]}
        elif how =="SMMjoint":
            lbd,sigma_pb,sigma_pr,rho,sigma = self.para_est_SMM_joint
            exp_para_est_dct = {'lambda':lbd,
                                'sigma_pb':sigma_pb,
                               'sigma_pr':sigma_pr}
            process_para_est_dct = {'rho':rho,
                                   'sigma':sigma}
        elif how =="GMMsjoint":
            lbd,sigma_pb,sigma_pr,rho,sigma = self.para_est_joint
            exp_para_est_dct = {'lambda':lbd,
                                'sigma_pb':sigma_pb,
                               'sigma_pr':sigma_pr}
            process_para_est_dct = {'rho':rho,
                                   'sigma':sigma} 
        
        new_instance = cp.deepcopy(self)
        new_instance.exp_para = exp_para_est_dct
        self.forecast_moments_est = new_instance.ForecasterbySim()
        plt.style.use('ggplot')
        if all_moms == False:
            moments_to_plot = self.moments
        else:
            moments_to_plot = self.all_moments
            
        m_ct = len(moments_to_plot)
        
        x = plt.figure(figsize=([3,3*m_ct]))
        if diff_scale == False:
            for i,val in enumerate(moments_to_plot):
                plt.subplot(m_ct,1,i+1)
                plt.plot(self.forecast_moments_est[val],'s-',label='model:'+ val)
                plt.plot(np.array(self.data_moms_dct[val]),'o-',label='data:'+ val)
                plt.legend(loc=1)
        if diff_scale == True:
            for i,val in enumerate(moments_to_plot):
                ax1 = plt.subplot(m_ct,1,i+1)
                ax1.plot(self.forecast_moments_est[val],'rs-',label='model:'+ val)
                ax1.legend(loc=0)
                ax2 = ax1.twinx()
                ax2.plot(np.array(self.data_moms_dct[val]),'o-',color='steelblue',label='(RHS) data:'+ val)
                ax2.legend(loc=3)
                
#################################
######## New
################################
                
    def WM1stSMM(self):
        """
        - get the 1-st step variance and covariance matrix
        """
        exp_para_est_dct = {'lambda':self.para_estSMM[0],
                            'sigma_pb':self.para_estSMM[1],
                           'sigma_pr':self.para_estSMM[2]}
        self.exp_para = exp_para_est_dct
        
        sim_moms_dct = self.SMM()
        data_moms_scalar = self.data_moms_scalar_dct
        sim_moms = np.array([sim_moms_dct[key] for key in self.moments])
        data_moms = np.array([data_moms_scalar[key] for key in self.moments])
        distance = sim_moms - data_moms
        distance_diag = np.diag(distance*distance.T)
        self.wm1st = np.linalg.inv(distance_diag)
        return self.wm1st
    
    def WMbootSMM(self,
                  n_boot = 100):
        # data moments 
        data_moms_scalar = self.data_moms_scalar_dct
        
        # parameters 
        exp_para_est_dct = {'lambda':self.para_estSMM[0],
                            'sigma_pb':self.para_estSMM[1],
                           'sigma_pr':self.para_estSMM[2]}
        self.exp_para = exp_para_est_dct
        
        distance_boot = []
        
        for i in range(n_boot): 
            self.SMM()
            sim_moms_dct = self.SMMMoments
            sim_moms = np.array([sim_moms_dct[key] for key in self.moments])
            data_moms = np.array([data_moms_scalar[key] for key in self.moments])
            distance = sim_moms - data_moms
            distance_boot.append(np.array(distance))
        #print(np.array(distance_boot).shape)
        vcv_boot_ary = np.array(distance_boot, dtype=np.float64)
        self.vcv_boot = np.cov(vcv_boot_ary.T)
        #print(self.vcv_boot)
        self.wm_boot = np.linalg.inv(self.vcv_boot) 
# -

## parameter learning estimator 
PL_para_default = SE_para_default


# + {"code_folding": [0, 3, 24, 38, 64, 89, 112, 117, 124]}
### Paramter Learning(PL) class 

class ParameterLearning:
    def __init__(self,
                 real_time,
                 history,
                 horizon = 1,
                 process_para = process_para,
                 exp_para = {},
                 moments=['Forecast','Disg','Var']):
        self.real_time = real_time
        self.history = history 
        self.n = len(real_time)
        self.horizon = horizon
        self.process_para = process_para
        self.exp_para = exp_para
        self.data_moms_dct ={}
        self.para_est = {}
        self.moments = moments
        self.all_moments = ['Forecast','FE','Disg','Var']
        
    def GetRealization(self,realized_series):
        self.realized = realized_series   
    
    def SimulateRealization(self):
        n = self.n
        rho = self.process_para['rho']
        sigma =self.process_para['sigma']
        
        
        np.random.seed(12345)
        shocks = np.random.randn(n)*sigma
        realized = np.zeros(n)
        for i in range(n):
            cum_shock = sum([rho**h*shocks[h] for h in range(self.horizon)])
            realized[i] = rho**self.horizon*self.real_time[i] +cum_shock
        self.realized = realized
    
    def LearnParameters(self):
        n = self.n
        history = self.history
        n_burn = len(history) - n
        n_history = n + n_burn  # of course equal to len(history)
        real_time = self.real_time
        history = self.history
        rhos_to_burn = np.zeros(n_history)
        sigmas_to_burn = np.zeros(n_history)
        
        
        for i in range(n_history):
            ## OLS parameter learning here
            if i >=2:
                x = history[0:i]
                model = AR(x)
                ar_rs = model.fit(1,trend='nc')
                rhos_to_burn[i] = ar_rs.params[0]
                sigmas_to_burn[i] = np.sqrt(sum(ar_rs.resid**2)/(len(x)-1))
            else:
                pass 
        self.rhos = rhos_to_burn[n_burn:]
        self.sigmas = sigmas_to_burn[n_burn:]
        self.process_para_learned = {'rho':self.rhos,
                                    'sigma':self.sigmas}
    
    def Forecaster(self):
        ## parameters
        n = len(self.real_time)
        rhos = self.process_para_learned['rho']
        sigmas =self.process_para_learned['sigma']
        
        ## parameters
        max_back = self.max_back
        real_time = self.real_time
        horizon = self.horizon
        
        ## forecast moments 
        Disg = np.zeros(n)
        infoset = real_time
        nowcast = infoset
        forecast = np.multiply(rhos**horizon,nowcast)
        Var = [hstepvar(horizon,sigmas[i],rhos[i]) for i in range(n)] # this does not include var parameter
        FE = forecast - self.realized ## forecast errors depend on realized shocks 
        self.forecast_moments = {"Forecast":forecast, 
                "FE":FE,
                "Disg":Disg,
                "Var":Var}
        return self.forecast_moments
    
    ## a function estimating SE model parameter only 
    def PL_EstObjfunc(self,
                      lbd):
        """
        input
        -----
        lbd: the parameter of PL model to be estimated
        
        output
        -----
        the objective function to minmize
        """
        moments = self.moments
        PL_para = {"lambda":lbd}
        self.exp_para = PL_para  # give the new lambda
        data_moms_dct = self.data_moms_dct
        
        PL_moms_dct = self.Forecaster()
        PL_moms = np.array([PL_moms_dct[key] for key in moments] )
        data_moms = np.array([data_moms_dct[key] for key in moments] )
        obj_func = PrepMom(PL_moms,data_moms)
        return obj_func 
    
    ## feeds the instance with data moments dictionary 
    def GetDataMoments(self,
                       data_moms_dct):
        self.data_moms_dct = data_moms_dct
        
    ## invoke the estimator 
    def ParaEstimate(self,
                     para_guess=0.2,
                     method='CG'):
        self.para_est = Estimator(self.PL_EstObjfunc,
                                  para_guess=para_guess,
                                  method='CG')
        
    def ForecastPlot(self,
                     all_moms = False):
        plt.style.use('ggplot')
        if all_moms == False:
            m_ct = len(self.moments)
            x = plt.figure(figsize=([3,3*m_ct]))
            for i,val in enumerate(self.moments):
                plt.subplot(m_ct,1,i+1)
                plt.plot(self.forecast_moments[val],label=val)
                plt.legend(loc=1)
        if all_moms == False:
            moments_to_plot = self.moments
        else:
            moments_to_plot = self.all_moments
        
        m_ct = len(moments_to_plot)
        x = plt.figure(figsize=([3,3*m_ct]))
        for i,val in enumerate(moments_to_plot):
            plt.subplot(m_ct,1,i+1)
            plt.plot(self.forecast_moments[val],label=val)
            plt.legend(loc=1)
