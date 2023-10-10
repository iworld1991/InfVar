# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## SMM Estimation of Theories of Expectation Formation with Inflation Expectation
#

# +
import numpy as np
import matplotlib.pyplot as plt
from numba import types
from numba.typed import Dict,List
import pandas as pd
from statsmodels.tsa.api import AutoReg as AR
pd.options.display.float_format = '{:,.2f}'.format
plt.style.use('ggplot')

## figure config

lw = 4  #line width

# -

# ## Model

# +
from SMMEst import SimAR1, SimUCSV,ObjGen, ObjWeight,ParaEst,type_list

from SMMEst import StickyExpectationAR, StickyExpectationSV
from SMMEst import NoisyInformationAR, NoisyInformationSV
from SMMEst import DiagnosticExpectationAR, DiagnosticExpectationSV
from SMMEst import DENIHybridAR, DENIHybridSV



# + pycharm={"name": "#%%\n"} code_folding=[0]
## create some fake parameters and data to initialize model class
### not used for estimation

## AR1 parameters
ρ0,σ0 = 0.98, 0.10

history0 = SimAR1(ρ0,
                  σ0,
                  200)
real_time0 = history0[11:-2]

realized0 = history0[12:-1]

### UCSV inflation

p0_fake = 0 ## initial permanent component
γ_fake = 0.1 ## size of shock to the volatility
σs_now_fake = [0.2,0.3] ## volatility of permanent and transitory component

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

# ### Sticky Expectation (SE) + AR1

# ### Sticky Expectation (SE) + SV

# + code_folding=[1]
## initialize the ar instance
sear0 = StickyExpectationAR(exp_para = np.array([0.2]),
                            process_para = np.array([ρ0,σ0]),
                            real_time = real_time0,
                            history = history0,
                            horizon = 1)

sear0.GetRealization(realized0)

# + code_folding=[]
## initialize the sv instance
sesv0 = StickyExpectationSV(exp_para = np.array([0.3]),
                           process_para = np.array([0.1]),
                           real_time = xx_real_time,
                           history = xx_real_time) ## history does not matter here, 

## get the realization 

sesv0.GetRealization(xx_realized)
# -

# ### Noisy Information (NI) + AR1
#

# ### Noisy Information (NI) + SV
#
#

# + code_folding=[0, 1]
## initialize the ar instance
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

# ###  Diagnostic Expectation(DE) + AR1

# ###  Diagnostic Expectation(DE) + SV

# + code_folding=[0]
## initialize the ar instance
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

# ###  Diagnostic Expectation and Noisy Information Hybrid(DENI) + AR1

# ###  Diagnostic Expectation and Noisy Information Hybrid(DENI) + SV
#
#

# + code_folding=[1]
## initialize the ar instance
deniar0 = DENIHybridAR(exp_para = np.array([0.1,0.3]),
                       process_para = np.array([ρ0,σ0]),
                            real_time = real_time0,
                            history = history0,
                            horizon = 1)

deniar0.GetRealization(realized0)

# + code_folding=[0]
## initial a sv instance
denisv0 = DENIHybridSV(exp_para = np.array([0.1,0.2]),
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
real_time_inf = real_time_index.pct_change(periods=12)*100  ## year-over-year inflation, standard 
real_time_inf = real_time_inf.dropna()
# -

# #### Inflation data 

# + code_folding=[0]
###############
## monthly ### 
##############

InfM = pd.read_stata('../OtherData/InfM.dta')
InfM = InfM[-InfM.date.isnull()]
dateM = pd.to_datetime(InfM['date'],format='%Y%m%d')

InfM.index = pd.DatetimeIndex(dateM,
                              freq='infer')

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

# + code_folding=[0]
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
## plot times eries of 

SPFCPI.plot(title='SPF Expectations')

SCECPI.plot(title='SCE Expectations')

# + code_folding=[]
import datetime

## tempoary date used to check the data moments 

end_date_late = datetime.datetime(2023, 3, 30)
end_date_early = datetime.datetime(2020, 3, 30)

## we only focus on 4th quarter observations from SPF 
SPFCPI = SPFCPI[SPFCPI.index.quarter==4]


# +
print('SCE\n')
print(SCECPI.mean())
print('sample period, begin at '+str(SCECPI.index[0])+', and end at '+str(SCECPI.index[-1]))
print('\n')
print('before 2020\n')
print(SCECPI[SCECPI.index<end_date_early].mean())

print('\n')

print('SPF (only in the forth quarter)\n')
print(SPFCPI.mean())
print('sample period, begin at '+str(SPFCPI.index[0])+', and end at '+str(SPFCPI.index[-1]))
print('\n')

print('before 2020\n')
print(SPFCPI[SPFCPI.index<end_date_early].mean())

# +
## filter sample period 

## SPF
SPFCPI = SPFCPI[SPFCPI.index<end_date_early]

## SCE
SCECPI = SCECPI[SCECPI.index<end_date_early]

# + code_folding=[]
## Combine expectation data and real-time data 

SPF_est= pd.concat([SPFCPI,
                    real_time_inf,
                    InfQ], 
                   join='inner', 
                   axis=1)

SCE_est = pd.concat([SCECPI,
                     real_time_inf,
                     InfM], 
                    join='inner', 
                    axis=1)
# -

# #### History data 

# + code_folding=[]
## process parameters estimation AR1 
# period filter 
start_t='2000-01-01'
end_t = '2023-3-30'   ## 

######################
### quarterly data ##
#####################

CPICQ = InfQ['Inf1y_CPICore'].copy().loc[start_t:end_t]

print('for SPF moments estimation, the sample is between '+str(start_t)+' and '+str(end_t))

###################
### monthly data ##
###################

CPIM = InfM['Inf1y_CPIAU'].copy().loc[start_t:end_t]

print('for SPF moments estimation, the sample is between '+str(start_t)+' and '+str(end_t))

# + code_folding=[0]
## history data, the series ends at the same dates with real-time data but startes earlier

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

# + code_folding=[0]
######################
### quarterly data ##
#####################

CPICQ_demean = CPICQ

ARmodel = AR(CPICQ_demean,lags=[1],trend='n') 
ar_rs = ARmodel.fit()
rhoQ_est = ar_rs.params[0]
if rhoQ_est>1.0:
    rhoQ_est = 1.0
sigmaQ_est = np.sqrt(ar_rs.sigma2) #np.sqrt(sum(ar_rs.resid**2)/(len(CPIM)-1))

###################
### monthly data ##
###################

CPIM_demean = CPIM
ARmodel2 = AR(CPIM_demean,lags=[1],trend='n') ## 12 months of lags!
ar_rs2 = ARmodel2.fit()
rhoM_est = ar_rs2.params[0]
if rhoM_est>1.0:
    rhoM_est = 1.0
sigmaM_est = np.sqrt(ar_rs2.sigma2)  # or np.sqrt(sum(ar_rs2.resid**2)/(len(CPIM)-1))

# + code_folding=[]
print('For the sample before', str(end_t))

print('quarterly AR(1) estimates for CPI core:')
print(rhoQ_est)
print(sigmaQ_est)
print('monthly AR(1) estimates for CPI headline:')
print(rhoM_est)
print(sigmaM_est)
# -

# #### Data moments 

# + code_folding=[80, 108]
#####################################
## preparing data moments
#####################################

## Be careful with the frequency here when computing auto-correlation!!!!!
### SPF: quarters 
### SCE: month

#####################
## Professionals ####
#####################

### inflation moments 

realized_CPIC = CPICQ
InfAV_data = np.mean(realized_CPIC-np.mean(realized_CPIC))
InfVar_data = np.var(realized_CPIC)
InfATV_data = np.cov(np.stack( (realized_CPIC[1:],realized_CPIC[:-1]),axis = 0 ))[0,1]
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
FEATV_data = np.cov(np.stack( (FEs_data[1:],FEs_data[:-1]),axis = 0))[0,1]  ## one quarter apart 
## annual autocovariance


Disg_data = np.mean(Disgs_data)
DisgVar_data = np.var(Disgs_data)
DisgATV_data = np.cov(np.stack( (Disgs_data[1:],Disgs_data[:-1]),axis = 0))[0,1]
Var_data = np.mean(Vars_data)
VarVar_data = np.var(Vars_data)
VarATV_data = np.cov(np.stack( (Vars_data[1:],Vars_data[:-1]),axis = 0))[0,1]
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
############!!!!! using "xx_rd" moments only if one wants to control for individual fixed effects
###########################################################################################

control_ind_fe = True

if control_ind_fe:
    exp_data_SCE = SCE_est[['SCE_Mean','SCE_FE','SCE_Disg_rd','SCE_Var_rd']]
    exp_data_SCE = exp_data_SCE.rename(columns={"SCE_Mean": "Forecast", "SCE_FE": "FE",
                                            "SCE_Disg_rd":"Disg","SCE_Var_rd":"Var"})
else:
    exp_data_SCE = SCE_est[['SCE_Mean','SCE_FE','SCE_Disg','SCE_Var']]
    exp_data_SCE = exp_data_SCE.rename(columns={"SCE_Mean": "Forecast", "SCE_FE": "FE",
                                            "SCE_Disg":"Disg","SCE_Var":"Var"})

## inflation moments 

realized_CPI = CPIM

InfAV_data = np.mean(realized_CPI-np.mean(realized_CPI))
InfVar_data = np.var(realized_CPI)
InfATV_data = np.cov(np.stack( (realized_CPI[1:],realized_CPI[:-1]),axis = 0 ))[0,1]

## expectation moments 
FEs_data = exp_data_SCE['FE']
Disgs_data = exp_data_SCE['Disg']
Vars_data = exp_data_SCE['Var']

FE_data = np.mean(FEs_data)
FEVar_data = np.var(FEs_data)
FEATV_data = np.cov(np.stack( (FEs_data[1:],FEs_data[:-1]),axis = 0 ))[0,1]

Disg_data = np.mean(Disgs_data)
DisgVar_data = np.var(Disgs_data)
DisgATV_data = np.cov(np.stack( (Disgs_data[1:],Disgs_data[:-1]),axis = 0))[0,1]
Var_data = np.mean(Vars_data)
VarVar_data = np.var(Vars_data)
VarATV_data = np.cov(np.stack( (Vars_data[1:],Vars_data[:-1]),axis = 0))[0,1]


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

print('SPF\n')
print(dict(data_moms_dct_SPF))
print('\n')
print('SCE\n')
print(dict(data_moms_dct_SCE))

# #### model moments 

# + code_folding=[0, 18, 39]
fire_ar_mom_dct = {'InfAV':0.0,
           'InfVar':r'$\sigma^2_\omega/(1-\rho^2)$',
           'InfATV':r'$\rho\sigma^2_\omega/(1-\rho^2)$',
           'FE':0.0,
           'FEVar':r'$\sigma^2_\omega$',
           'FEATV':0.0,
           'Disg':0.0,
           'DisgVar':0.0,
           'DisgATV':0.0,
           'Var':r'$\sigma^2_\omega$',
            'VarVar':0.0,
            'VarATV':0.0
           }

fire_ar_mom = pd.DataFrame([dict(fire_ar_mom_dct)])
fire_ar_mom.index = ['FIRE+AR']


fire_sv_mom_dct = {'InfAV':0.0,
           'InfVar':'N/A',
           'InfATV':'N/A',
           'FE':0.0,
           'FEVar':r'$\bar\sigma^2_{\eta}+\bar\sigma^2_{z}$',
           'FEATV':0.0,
           'Disg':0.0,
           'DisgVar':0.0,
           'DisgATV':0.0,
           'Var':r'$\bar\sigma^2_{\eta}+\bar\sigma^2_{z}$',
            'VarVar':'>0',
            'VarATV':'>0'
           }

fire_sv_mom = pd.DataFrame([dict(fire_sv_mom_dct)])
fire_sv_mom.index = ['FIRE+SV']


## these moments have not been used yet 
## need to make sure all is correct 

sear_mom_dct = {'InfAV':0.0,
           'InfVar':r'$\sigma^2_\omega/(1-\rho^2)$',
           'InfATV':r'$\rho\sigma^2\omega/(1-\rho^2)$',
           'FE':0.0,
           'FEVar':r'$\lambda^2\sigma^2/(1-(1-\lambda)^2\rho^2)$',
           'FEATV':r'$(1-\lambda)\rho\text{FEVar}$',
           'Disg':0.0,
           'DisgVar':0.0,
           'DisgATV':0.0,
           'Var':r'$\sigma^2$',
            'VarVar':0.0,
            'VarATV':0.0
           }

sear_mom = pd.DataFrame([sear_mom_dct])
sear_mom.index = ['SE+AR']

# + code_folding=[0]
## data_moments 


data_mom_SPF = pd.DataFrame([dict(data_moms_dct_SPF)])
data_mom_SPF.index = ['SPF']
data_mom_SCE = pd.DataFrame([dict(data_moms_dct_SCE)])
data_mom_SCE.index = ['SCE']

data_mom_df = pd.concat([data_mom_SPF,
                         data_mom_SCE,
                         fire_ar_mom,
                        fire_sv_mom])

data_mom_df = data_mom_df.applymap(lambda x: round(x, 3) 
                                   if isinstance(x, (int, float)) else x)


### keep only selected moments 

data_mom_df = data_mom_df.drop(columns=['FEATV',
                                        'DisgVar',
                                        'DisgATV',
                                        'VarVar',
                                        'VarATV'])

data_mom_df.T.to_excel('tables/data_moments.xlsx')
data_mom_df.T
# -

# ### Data moments

# + code_folding=[0]
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
## be careful with the order, I define eta as the permanent and eps to be the transitory
 ######################################################################################
    

### quarterly plot 

plt.figure(figsize=(10,5))
plt.plot(CPICQ_UCSV_Est['sd_p_est'],
         'r--',
          lw = lw,
         label='permanent volatility')
plt.plot(CPICQ_UCSV_Est['sd_t_est'],
         'k-.',
         lw = lw,
         label='transitory volatility')
plt.title('Estimated Stochastic Volatility of Core CPI Inflation')
plt.ylabel('std')
plt.legend(loc=1)
plt.savefig('../graphs/inflation/UCSVQ.png')
    
### monthly plot 

plt.figure(figsize=(10,5))
plt.plot(CPIM_UCSV_Est['sd_p_est'],
         'r--',
          lw = lw,
         label='permanent volatility')
plt.plot(CPIM_UCSV_Est['sd_t_est'],
         'k-.',
         lw = lw,
         label='transitory volatility')
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

# ### Estimation 

# + code_folding=[0, 3, 6, 10, 17, 27, 37, 48, 58, 68, 81, 96, 103, 109, 120, 128, 136, 146, 156, 167, 171, 176, 181, 187]
agents_list = ['SPF',
               'SCE']

horizon_list = [4,
               12]

process_list = ['AR',
                'SV'
               ]

ex_model_list = ['SE',
                 'NI',
                 'DE',
                 'DENI'
                ]
nb_ex_model = len(ex_model_list)

moments_list = [['FE','FEVar','FEATV'],
               ['FE','FEVar','FEATV','Disg'],
               ['FE','FEVar','FEATV','Disg','Var']]

nb_moments = len(moments_list)

how_list =['2-step','Joint']

moments_list_general = ['FE','FE+Disg','FE+Disg+Var']

model_list = [sear0,
              niar0,
              dear0,
              deniar0,
              sesv0,
              nisv0,
              desv0,
              denisv0
]

algorithm_list = ['trust-constr',
                 'trust-constr',
                 'trust-constr',
                 'trust-constr',
                 'trust-constr',
                 'trust-constr',
                 'trust-constr',
                 'trust-constr'
                 ]


algorithm_joint_list = ['trust-constr',
                        'trust-constr',
                        'trust-constr',
                        'trust-constr',
                        None,
                        None,
                        None,
                        None
]

bns_list =[((0,1),),
           ((0,3),(0,3),),
           ((-2,2),(0,5),),
           ((-3,3),(0,3),),
           ((0,1),),
           ((0,3),(0,3),),
           ((-2,2),(0,np.inf),),
           ((-3,3),(0,3),)
]

bns_joint_list =[((0,1),(0.9,1),(0,np.inf),),
                 ((0,3),(0,3),(0.9,1),(0,np.inf),),
                 ((-2,2),(0,5),(0.9,1),(0,np.inf),),
                 ((-3,3),(0,3),(0.9,1),(0,np.inf),),
                 None,
                 None,
                 None,
                 None
]

data_mom_dict_list = [data_moms_dct_SPF,
                      data_moms_dct_SCE]

process_paras_list = [process_paraQ_est_ar,
                      process_paraQ_est_sv,
                      process_paraM_est_ar,
                      process_paraM_est_sv
]

realized_list = [realized_CPIC.astype(np.float64),
                 realized_CPI.astype(np.float64)]

real_time_list = [np.array(real_time_Q_ar),
                 np.array(real_time_Q_sv),  ## 4 x t array 
                np.array(real_time_M_ar),
                np.array(real_time_M_sv)
]  ## 4 x t array 

history_list = [np.array(history_Q_ar),
               np.array(history_Q_sv),     ## 4 x t array 
               np.array(history_M_ar), 
               np.array(history_M_sv)
]     ## 4 x t array 

## parameter guesses 
guesses_list = [np.array([0.2]),  ## se lbd 
               np.array([0.5,0.8]),  ## ni sigma_pb, sigma_pr
               np.array([0.3,0.4]),   ## de theta theta_sigma
               np.array([0.1,0.3])
               ]  ## deni theta, sigma_pb, sigma_pr

guesses_joint_list = [np.array([0.2,0.98,0.1]),            ## se lbd 
                     np.array([0.1,0.2,0.95,0.1]),      ## ni sigma_pb, sigma_pr
                     np.array([0.3,0.4,0.95,0.1]),      ## de theta theta_sigma
                     np.array([0.1,0.3,0.95,0.1]),  ## theta, sigma_pb, sigma_pr  
                       ## for sv models not used
                      np.array([0.3,0.2]),            ## se lbd 
                      np.array([0.1,0.2,0.2]),      ## ni sigma_pb, sigma_pr
                       np.array([0.3,0.2]),      ## de theta theta_sigma
                       np.array([0.1,0.2])
                     ]  ## deni theta, sigma_pb, sigma_pr]

n_exp_paras_list = [1,
                    2,
                    2,
                    2]


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
                  # r'$\hat\sigma_{pb}$',
                   r'$\hat\sigma_{pr}$',
                   r'$\rho$',
                   r'$\sigma$',
                   r'$\hat\theta$',
                  # r'$\hat\sigma_{pb}$',
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
                  # r'$\hat\sigma_{pb}$',
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

# + code_folding=[9, 134]
################################################################################
## A loop to estimate the model for different agents, theory, inflation process and joint/2-step 
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
            print('horizon', horizon_list[agent_id])
            instance.horizon = horizon_list[agent_id]
            
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
                                    moment_choice = List(moments_this),
                                    how ='expectation')
                    return scalor
                
                guess_this = guesses_list[exp_id]
                ## estimating
                para_est  = ParaEst(Obj_this,
                                 para_guess = guess_this,
                                 method= alg_this,
                                bounds = bounds_this)
                
                ## try it another time if no convergence 
                if np.isnan(para_est).any()==True:
                    para_est  = ParaEst(Obj_this,
                                 para_guess = guess_this,
                                 method= alg_this,
                                bounds = bounds_this)
                    
                ## same para est
                if np.isnan(para_est).any()==False:
                    
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
                                           moment_choice = List(moments_this),
                                           how ='expectation')
                        return scalor
                    para_est_step2  = ParaEst(Obj_this_step2,
                                        para_guess = guess_this,
                                        method= alg_this,
                                        bounds = bounds_this)
                    if np.isnan(para_est_step2).any()==True:
                        print('the 2-step estimation is nan')
                        para_est_step2_nan = np.array([np.nan])
                        para_est_step2 = para_est_step2_nan
                else:
                    print('the estimation is nan')
                    para_est_nan = np.array([np.nan])
                    para_est = para_est_nan
                                        
                    para_est_step2_nan = np.array([np.nan])
                    para_est_step2 = para_est_step2_nan
                    
                ## save para est
                
                para_est = np.round(para_est,2)
                print('Step 1:'+str(para_est))
                paras_list_this_model.append(para_est)
                paras_list.append(para_est)
                
                para_est_step2 = np.round(para_est_step2,2)
                print('Step 2:'+str(para_est_step2))
                paras_step2_list_this_model.append(para_est_step2)
                paras_step2_list.append(para_est_step2)
                
                
                ##  joint estimation 
                n_exp_paras_this = n_exp_paras_list[exp_id]
                
                moments_this_ = List(moments_this+['InfAV','InfVar','InfATV']) ## added inflation moments 
                
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
                    if  np.isnan(paras_joint_est).any()==False:
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
                        if np.isnan(para_est_joint_step2).any()==True:
                            print('the 2-step estimation is nan')
                            para_est_joint_step2 = np.array([np.nan])
                        
                    else:
                        print('the estimation is nan')
                        paras_joint_est = np.array([np.nan])
                        para_est_joint_step2 = np.array([np.nan])
                
                else:
                    paras_joint_est = np.array([np.nan])
                    para_est_joint_step2 = np.array([np.nan])
                
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
            para_all_est_tab_this_model.to_excel('tables/after2022/'+agent+'_'+process+'_'+ex_model+'.xlsx',
                                       float_format='%.2f',
                                       index = True)


# + code_folding=[]
## create multiple index to store coefficient estimates 

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

paras_table.columns = [['2-step Estimate',
                        '2-step Estimate 2nd step']]
paras_joint_table.columns = [['Joint Estimate',
                              'Joint Estimate 2nd step']]

# + code_folding=[]
paras_combine_table = pd.merge(paras_table,
                               paras_joint_table,
                               how='outer',
                               left_index=True,
                               right_index=True)
# -

print(process_paraQ_est_ar)
print(process_paraM_est_ar)

paras_combine_table

# +
## only 2nd-step estimates 

paras_combine_table_2step = paras_combine_table[['2-step Estimate 2nd step','Joint Estimate 2nd step']]

paras_combine_table_2step

# + code_folding=[0, 2]
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
    print(case)
    for name in paras_combine_table.columns:
        print(name)
        paras_combine_table.loc[case,name]= tuple(np.array([]))
        #paras_combine_table.loc[case][name] = tuple(np.array([]))

# + code_folding=[3, 14]
t_sim = 500
t_burn = 30

def simulate_history(pg_para,
                     t_sim,
                     t_burn):
    if len(pg_para)==2:
        ρ,σ = pg_para
        history_this = SimAR1(ρ,
                              σ,
                              t_sim)
        real_time_this = history_this[t_burn:-2] 
        realized_this = history_this[t_burn+1:-1]
        
    elif len(pg_para)==1:
        γ_fake = pg_para[0]
        
        
        history_this,p_history_this,vol_p_history_this,vol_t_history_this = SimUCSV(γ_fake,
                                                                                    nobs = t_sim,
                                                                                    p0 = p0_fake) 
        history_this =  np.array([history_this,
                                p_history_this,
                                vol_p_history_this,
                                vol_t_history_this]
                               )

        real_time_this = history_this[:,t_burn:-2]
        realized_this = history_this[0,t_burn+1:-1] ## notice, just Y itself is 
        
    return realized_this,history_this,real_time_this


# + code_folding=[6]
## generate model moments

smm_list = []
smm_joint_list = []
smm_ts_list = []

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
            instance = model_list[model_idx]
            print(instance)
            
            ## directly use the history 
            instance.GetRealization(realized_this)   
            instance.history = history_this
            instance.real_time = real_time_this
            print("Length of real time series is "+str(len(instance.real_time)))
            
            ## compute moments with these parameters 
            for moments in moments_list_general:
                ## 2-step moments 
                para_est_this = np.array(list(paras_combine_table.loc[agent,process,ex_model,moments]['2-step Estimate 2nd step'])).flatten()
                print(para_est_this)
                print(process_paras_this)
                
                try:
                    instance.process_para = process_paras_this
                    instance.exp_para = para_est_this
                    smm_ts_this = instance.SimForecasts()
                    print('Length of simulated moments is '+str(len(smm_ts_this['FE'])))
                    smm_this = instance.SMM()
                    print(smm_this)

                except:
                    print('failed')
                    smm_this = {}
                    smm_ts_this = {"Forecast":np.nan,
                                  "FE":np.nan,
                                  'Disg':np.nan,
                                  'Var':np.nan}
                smm_list.append(smm_this)
                smm_ts_list.append(smm_ts_this)

                ## joint moments
                n_exp_paras_this = n_exp_paras_list[exp_id]
                para_est_this_ = np.array(list(paras_combine_table.loc[agent,process,ex_model,moments]['Joint Estimate 2nd step'])).flatten()
                print(para_est_this_)
                try:
                    instance.exp_para = para_est_this_[0:n_exp_paras_this]     
                    instance.process_para = para_est_this_[n_exp_paras_this:]
                    
                    try:
                        smm_this = instance.SMM()
                        print(smm_this)
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

# +
## store times series of moments 

smm_names = ['Forecast','FE','Disg','Var']

iterables = [agents_list, process_list, ex_model_list,moments_list_general,smm_names]

midx_ts = pd.MultiIndex.from_product(iterables, names=['Agents', 'Process','Model','Moments','Variable'])

smm_ts_table = pd.DataFrame(index = midx_ts)

smm_ts_list_new = [ts[name] for ts in smm_ts_list for name in smm_names]

smm_ts_table['smm_ts'] = smm_ts_list_new
# -

#plt.plot(realized_CPIC)
plt.plot(smm_ts_table.loc['SPF','SV','DE','FE+Disg+Var',:].loc['Disg','smm_ts'])

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

# + code_folding=[]
## export 
mom_compare_spf.to_excel('./tables/after2022/spf_moments.xlsx')
mom_compare_sce.to_excel('./tables/after2022/sce_moments.xlsx')
mom_joint_compare_spf.to_excel('./tables/after2022/spf_joint_est_moments.xlsx')
mom_joint_compare_sce.to_excel('./tables/after2022/sce_joint_est_moments.xlsx')
# -

mom_compare_spf

mom_joint_compare_spf

mom_compare_sce

mom_joint_compare_sce
