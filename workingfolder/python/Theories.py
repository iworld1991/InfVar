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

# ## Competing theories of expectation formation
#
# - Tao Wang, Johns Hopkins
# - Edited: Feb, 2023
#

# +
import matplotlib.pyplot as plt 
import numpy as np 

plt.style.use('ggplot')
# -

# ### A General Framework 
#
#
# The density forecast of $h$-period ahead variable $y^i$ by agent i is 
#
# $$\widehat f_{i,t}(y^i_{t+h}|I_{i,t})$$
#
# $I_{i,t}$ is the information set available at time t for individual $i$. $y_t$ could be individual variable or macroeconomic variable. We focus on aggregate variable here, thus we drop supscript $i$ from $y$. 
#
# Thus the expected value at time $t$ is 
#
# $$E_{i,t}(y_{t+h}|I_{i,t}) =\int \widehat f_{i,t}(y_{t+h}|I_{i,t})y_{t+h} d y_{t+h} \equiv \bar y_{i,t,t+h}$$
#
# The variance is 
#
# $$Var_{i,t}(y_{t+h}|I_{i,t})=\int \widehat f_{i,t}(y_{t+h}|I_{i,t}) (y_{t+h} - \bar y_{i,t,t+h})^2d y_{t+h}$$
#
#

# #### Assumption on the Fundamental Process: AR(1)

# Assume the process of $y_t$ is AR(1)
#
# $$y_{t+1} = \rho y_t + \omega_t$$
#
# $$\omega_t \sim N(0,\sigma^2_{\omega})$$

# ###  Full information rational expectation (FIRE)
#
# Full information implies the most recent realized $y_t$ is in the information set. $y_t \in I_t$ 
#
# ##### Individual moments 
#
# ##### Expectation
# $$E^*_{i,t}(y_{t+h}|y_t) = \rho^h y_t $$
#
# ##### Variance 
# $$Var^*_{i,t}(y_{t+h}|y_t) = \sum^{h}_{s=1}\rho^{2s} \sigma^2_{\omega}$$
#
# ##### Change in variance 
#
# $$\Delta Var^*_{i,t}(y_{t+h}|y_t)=-\rho^{2h}\sigma^2_{\omega}$$
#
# ##### Forecast error
#
# $${FE}^*_{i,t+h|t}=y_{t+h} - E_{i,t}(y_{t+h}|y_t) = \sum^h_{s=0} \rho^s \omega_{t+h-s}$$ should be unrelated to information at time $t$. 
#
# ##### Population moments 
#
# The same as individual moments as there is heterogeneity. $i$ is dropped from all equations above.

# ### Sticky expectations (SE)
#
# Agent does not update information instantaneously, instead at a Possion rate $\lambda$. Specificaly, at any point of time $t$, the agent learns about the up-to-date realization of $y_t$ with probability of $\lambda$; otherwise, it holds the most recent up-to-date realization of $y_{t-\tau}$, where $\tau$ is the time experienced since previous update. 
#
# #### Individual moments 
#
# For an individual whose most recent update occurs at $\tau$ period before, $I_{i,t} = I_{i,t-\tau} = y_{t-\tau}$. Thus
# ##### Expectation
# $$E^{se}_{i,t|t-\tau}(y_{t+h}|I_{i,t}) = E^{*}_{i,t}(y_{t+h}|y_{t-\tau}) = \rho^{h+\tau} y_{t-\tau}$$
#
# ##### Forecast error
#
# $$FE^{se}_{i,t+h|t} = y_{t+h} - \rho^{h+\tau} y_{t-\tau} = \sum^{h+\tau}_{s=0} \rho^s \omega_{t+h-s}$$
#
# ##### Variance 
# $$Var^{se}_{i,t|t-\tau}(y_{t+h}|I_{i,t}) = Var^{*}_{i,t}(y_{t+h}|y_{t-\tau}) = \sum^{h+\tau}_{s=0}\rho^{2s} \sigma^2_{\omega}$$
#
# $Var^{se}_{i,t|t-\tau}(y_{t+h}|_{i,t})$ increases as $\tau$ increases. The model collapses to full-information rational expectation if $\tau=0$ for all individuals. 
#
# Agent updates infrequently. 
#
# - When the update happens, the variance is revised substantially.  
#
# $$Var^{se}_{i,t}(y_{t+h}|y_t) - Var^{se}_{i,t|\tau}(y_{t+h}|y_{t-\tau}) = \sum^{h}_{s=0}\rho^{2s} \sigma^2_{\omega} - \sum^{h+\tau}_{s=0}\rho^{2s} \sigma^2_{\omega} = \sum^{\tau}_{s=0} \rho^{2s}\sigma^2_{\omega}$$
#
# - When the update does not happen, the variance is revised relatively little. 
#
# $$Var^{se}_{i,t|t-\tau-1}(y_{t+h}|y_{t-\tau-1}) - Var^{se}_{i,t|\tau}(y_{t+h}|y_{t-\tau}) = \sigma^2_{\omega}$$
#
# At individual level, it is hard to recover the information rigidity parameter $\lambda$ directly. The only clue is if information rigidity exists, one testable prediction is that the change in forecast variance varies substantially depending on if updated or not. The longer period for which the agent stays unupdated(greater $\tau$), the bigger the change is of the variance. 
#
# However, the difference in average responses in variance to new information may speak to potential heterogeneity in information rigidity. According to the theory above, higher information regidity implies high volatility of variances responses speak to greater information regidity.  
#
# #### Population moments 
#
# ##### Average forecast 
#
# The mean forecast across population is a weighted average of past rational expectations 
#
# \begin{eqnarray}
# \begin{split}
# \bar E^{se}_t(y_{t+h}) & = \lambda \underbrace{E^{*}_t(y_{t+h})}_{\text{rational expectation at t}} + (1-\lambda) \underbrace{\bar E^{se}_{t-1}(y_{t+h})}_{\text{average expectation at} t-1} \\
# & = \lambda E^{*}_t(y_{t+h}) + (1-\lambda) (\lambda E^{se}_{t-1}(y_{t+h})+ (1-\lambda) \bar E^{se}_{t-2}(y_{t+h}))... \\
# & = \lambda \sum^{\infty}_{s=0} (1-\lambda)^s E^{*}_{t-s}(y_{t+h}) \\
# & = \lambda \sum^{\infty}_{s=0} (1-\lambda)^s \rho^{s+h}y_{t-s}
# \end{split}
# \end{eqnarray}
#
# The change in average forecast is 
#
# \begin{eqnarray}
# \begin{split}
# \Delta \bar E^{se}_t(y_{t+h})&  = (1-\lambda) \Delta \bar E^{se}_{t-1}(y_{t+h}) + \lambda (E^{*}_t(y_{t+h}) - E^{*}_{t-1}(y_{t+h})) \\ 
# & = (1-\lambda) \Delta \bar E^{se}_{t-1}(y_{t+h}) + \lambda (\rho^{h+1}(y_{t-1}+\omega_t) -\rho^{h+1}y_{t-1}) \\
# & = (1-\lambda) \Delta \bar E^{se}_{t-1}(y_{t+h}) + \lambda \rho^{h+1} \omega_t 
# \end{split}
# \end{eqnarray}
#
# This implies the change in average forecast is serially correlated, depending on the information rigidity, i.e. lower $\lambda$ implies higher serial correlation. Also lower $\lambda$ implies the expectation underreact to the shocks at $t$. 
#
# ##### Cross-sectional disagreement
#
# According to information rigity model, if everyone is instantaneously udpated, there should not be disagreements. In general, the dispersion in forecasting is non-zero because of different lags in updating. 
#
#
# The impulse response of disagreement at time $t$ to a shock that realized at $\tau$ periods before is equal to. 
#
# \begin{eqnarray}
# \rho^{2(h+\tau)} \underbrace{(1-(1-\lambda)^{\tau+1}))}_{\text{People who have updated about }\omega_{t-\tau} }\overbrace{(1-\lambda)^{\tau+1}}^{\text{those who have not}} \omega^2_{t-\tau}
# \end{eqnarray}
#
#
# Then the disagreement at time $t$ is the essentially the comulated sum of all the marginal impacts of past shocks between $t-\infty$ till $t$. 
#
# \begin{eqnarray}
# \begin{split}
# Disg_{t+h|t} = Var(E^{se}_{i,t}(y_{t+h}) ) & = \lambda \sum^{\infty}_{\tau=0} (1-\lambda)^{\tau} (E_{t|\tau}(y_{t+h}) - \bar E_t(y_{t+h}))^2  \\
# & = \lambda \sum^{\infty}_{\tau=0} (1-\lambda)^{\tau} \underbrace{\rho^{2(h+\tau)}(1-(1-\lambda)^{\tau+1})(1-\lambda)^{\tau+1} \omega^2_{t-\tau}}_{\text{Disg induced by the shock } \omega^{t-\tau}} \\
# & = \lambda \sum^{\infty}_{\tau=0} (1-\lambda)^{\tau} \rho^{2(h+\tau)} ((1-\lambda)^{\tau+1}-(1-\lambda)^{2\tau+2}) \omega^2_{t-\tau} \\
# & = \lambda (1-\lambda)\rho^{2h}\sum^{\infty}_{\tau=0} ((1-\lambda)\rho)^{2\tau} \omega^2_{t-\tau}-\lambda (1-\lambda)^2 \rho^{2h}\sum^{\infty}_{\tau=0} ((1-\lambda)^3 \rho^2)^\tau \omega^2_{t-\tau} 
# \end{split} 
# \end{eqnarray}
#
#
# From time $t$ to $t+1$, the change in dispersion comes from two sources. One is newly realized shock at time $t+1$. The other component is from people who did not update at time $t$ and update at time $t+1$.  
#
# The unconditional disagreement is, therefore, equal to the following. 
#
# \begin{eqnarray}
# \begin{split}
# Disg_{\bullet+h|\bullet} &= \lambda (1-\lambda)\rho^{2h}\sum^{\infty}_{\tau=0} ((1-\lambda)\rho)^{2\tau} \sigma^2_{\omega}-\lambda (1-\lambda)^2 \rho^{2h}\sum^{\infty}_{\tau=0} ((1-\lambda)^3 \rho^2)^\tau \sigma^2_{\omega}  \\
# & = \lambda (1-\lambda)\rho^{2h}\frac{1}{1-((1-\lambda)\rho)^2} \sigma^2_{\omega}-\lambda (1-\lambda)^2 \rho^{2h}\frac{1}{1-((1-\lambda)^6 \rho^4)} \sigma^2_{\omega}  \\
# & = (\frac{1}{1-(1-\lambda)^2\rho^2} -\frac{1-\lambda}{1-(1-\lambda)^6 \rho^4}) \lambda (1-\lambda) \rho^{2h}\sigma^2_{\omega}  \\
# \end{split} 
# \end{eqnarray}
#
#
#
# Notice the change is positive, meaning the dispersion rises in response to a shock. Importantly, the increase is the same regardless of the realization of the shock. 
#
#
# Then it gradually returns to its steady state level. 
#
# ##### Average uncertainty (variance) 
#
# Since we have individual level variance, we can also derive average variance of the population. Taking the average of variane across individual agents at time $t$. 
#
# \begin{eqnarray}
# \begin{split}
# \bar Var^{se}_{t}(y_{t+h}) & = \sum^{+\infty}_{\tau =0} \underbrace{\lambda (1-\lambda)^\tau}_{\text{fraction who does not update until }t-\tau} \underbrace{Var_{t|t-\tau}(y_{t+h})}_{\text{ Variance of most recent update at }t-\tau} \\
# & = \sum^{+\infty}_{\tau =0} \lambda (1-\lambda)^\tau \sum^{h+\tau}_{s=1}\rho^{2(s-1)} \sigma^2_{\omega} \\
# & = \sum^{+\infty}_{\tau =0} \lambda (1-\lambda)^\tau (1 + \rho^2+...+\rho^{2(h+\tau-1)}) \sigma^2_{\omega} \\
# & =\sum^{+\infty}_{\tau =0} \lambda (1-\lambda)^\tau \frac{\rho^{2(h+\tau)}-1}{\rho^2-1}\sigma^2_{\omega}\\
# & =\sum^{+\infty}_{\tau =0} \lambda (1-\lambda)^\tau \frac{\rho^{2(h+\tau)}}{\rho^2-1}\sigma^2_{\omega} - \sum^{+\infty}_{\tau =0} \lambda (1-\lambda)^\tau \frac{1}{\rho^2-1}\sigma^2_{\omega} \\
# & =\sum^{+\infty}_{\tau =0} \lambda (1-\lambda)^\tau\rho^{2\tau}\frac{\rho^{2h}}{\rho^2-1}\sigma^2_{\omega} - \sum^{+\infty}_{\tau =0} \lambda (1-\lambda)^\tau \frac{1}{\rho^2-1}\sigma^2_{\omega} \\
# & =\sum^{+\infty}_{\tau =0} \lambda ((1-\lambda)\rho^2)^\tau\frac{\rho^{2h}}{\rho^2-1}\sigma^2_{\omega} - \sum^{+\infty}_{\tau =0} \lambda (1-\lambda)^\tau \frac{1}{\rho^2-1}\sigma^2_{\omega} \\
# & =\frac{\lambda}{(1-\rho^2+\lambda\rho^2)} \sum^{+\infty}_{\tau =0} (1-\rho^2+\lambda\rho^2) (1-(1-\rho^2+\lambda\rho^2))^\tau\frac{\rho^{2h}}{\rho^2-1}\sigma^2_{\omega} - \sum^{+\infty}_{\tau =0} \lambda (1-\lambda)^\tau \frac{1}{\rho^2-1}\sigma^2_{\omega} \\
# & =(\frac{\lambda\rho^{2h}}{(1-\rho^2+\lambda\rho^2)(\rho^2-1)} -\frac{1}{\rho^2-1})\sigma^2_{\omega}  \\
# & = (\frac{\lambda\rho^{2h}}{1-\rho^2+\lambda\rho^2} -1)\frac{\sigma^2_{\omega} }{\rho^2-1} \\
# & =  (\frac{\lambda\rho^{2h}-1+\rho^2-\lambda\rho^2}{1-\rho^2+\lambda\rho^2})\frac{\sigma^2_{\omega} }{\rho^2-1}
# \end{split}
# \end{eqnarray}
#
#
# The average uncertainty of h-ahead forecast does not change over time. 
#
#

# + code_folding=[11, 16]
## some experiments 

rho = 0.98
sigma = 1.0

def FE2_SE(lbd):
    return lbd**2/(1-(1-lbd)**2*rho**2)*sigma**2

def Var_SE(lbd):
    return 1/(1-(1-lbd)*rho**2)*sigma**2

def Disg_SE(lbd):
    first_part = 1/(1-(1-lbd)**2*rho**2)
    second_part = 1/(1-(1-lbd)**6*rho**4)
    return (first_part-second_part)*lbd*(1-lbd)*rho**2**sigma**2
    
lbds = np.linspace(0.0,
                   0.999,
                   20)

FE_SE2_ratios = FE2_SE(lbds)/sigma**2
Var_SE_ratios = Var_SE(lbds)/sigma**2
Disg_SE_ratios = Disg_SE(lbds)/sigma**2


plt.title('SE')

plt.plot(lbds,
         FE_SE2_ratios,
        label=r'$FE^2_{\bullet+1|\bullet}/\sigma^2_\omega$')

plt.plot(lbds,
         Var_SE_ratios,
        label=r'$Var_{\bullet+1|\bullet}/\sigma^2_\omega$')
plt.plot(lbds,
         Disg_SE_ratios,
        label=r'$Disg_{\bullet+1|\bullet}/\sigma^2_\omega$')

plt.legend(loc=1)
plt.xlabel(r'$\lambda$')
# -

# #### Summary of predictions with information rigidity models with update rate $\lambda$
#
# - Individual expectation may or may not change depend upon if updating. 
# - Individual variances changes non-monotonically depending on if updating. Always increase with arrival of new information. 
# - Population mean of forecast responds with lags. Change is serially correlated. 
# - Population dispersion of forecasts rise in response to new shocks and return to steady state level gradually. 
# - Population average variance does not change over time. 

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
# %matplotlib inline

# + code_folding=[]
## parameters 
ρ = 0.9
λ = 0.25
T = 11
t_shock = 1
h = T-1 
σ = 1
sigma_y = np.sqrt(σ**2/(1-ρ**2))  # unconditional standard error 

## shocks
shock = np.zeros(T)
shock[t_shock] = 1

## y_t
y = np.zeros(T) 
for t in range(T-1):
    y[t+1]=ρ*y[t]+shock[t+1]

# + code_folding=[0]
### Rational Expectation (RE)

## individual forecast of y_{T-1}
IndExpRE = np.zeros(T)
for t in range(T):
    IndExpRE[t] = ρ**(h-t)*y[t]
    
## forecasting error    
IndFERE = y[T-1] - IndExpRE

## individual variance
IndVarRE = np.zeros(T)
for t in range(T):
    for i in range(T-t):
        IndVarRE[t] += σ**2*ρ**(2*i)
IndVarRE = IndVarRE-σ**2

#IndVarRE2 = np.zeros(T)
#for t in range(T):
#    IndVarRE2[t] = σ**2*(ρ**2*(1-ρ**(2*(T-t-1))))/(1-ρ**2)
#IndVarRE2 = np.zeros(T)
    
## population average 

PopExpRE = IndExpRE

## population forecast error

PopFERE = IndFERE

## population disgreements

PopDisgRE = np.zeros(T)

## population variance 

PopVarRE = IndVarRE

# + code_folding=[0]
# Sticky expectation (SE)

## individual forecast 

IndExpSEOld = IndExpRE[0]*np.ones(T)  # if not update 
IndExpSENew = IndExpRE    # if update

## forecast error 

IndFESEOld = y[T-1] - IndExpSEOld
IndFESENew = y[T-1] - IndExpSENew

## individual variance 
    
IndVarSENew = IndVarRE

IndVarSEOld = np.zeros(T)
for t in range(t_shock):
    IndVarSEOld[t] = σ**2*(ρ**2*(1-ρ**(2*(T-t-1))))/(1-ρ**2)
for t in range(t_shock,T):
    IndVarSEOld[t]=IndVarSEOld[0]
    
## population forecast

PopExpSE = np.zeros(T)

PopExpSE[t_shock] = λ*IndExpRE[t_shock]
for t in range(t_shock+1,T):
    PopExpSE[t]=(1-λ)*PopExpSE[t-1]+λ*IndExpRE[t] 
    
## population forecast error

PopFESE = y[T-1]-PopExpSE

## population disagreements

PopDisgSE = np.zeros(T)

for t in range(t_shock,T):
    j = t -t_shock
    PopDisgSE[t] = ρ**(2*(h+j))*(1-λ**(j+1))*λ**(j+1)*σ**2

## population variance  (problems?)

PopVarSE = np.zeros(T)

PopVarSE[0] =PopVarRE[0]

for t in range(1,T):
    PopVarSE[t] = (1-λ)*PopVarSE[t-1] + λ*PopVarRE[t]
    #PopVarSE[t] = sum( [λ*(1-λ)**t*IndVarRE[s] for s in range(t)])


# + code_folding=[0, 2]
## Noisy information(NI) 

def NI(T,y,sigma_pb=1,sigma_pr=1):
    # parameters
    H = np.asmatrix ([[1,1]]).T   # ones 
    sigma_eps = sigma_pb # public signal noisiness
    sigma_xi =sigma_pr   # private signal noisiness
    s_pb = y+sigma_eps*np.random.randn(T)
    s_pr = y+sigma_xi*np.random.randn(T)
    s = np.asmatrix(np.array([s_pb,s_pr]))  # signals 
    sigma_v =np.asmatrix([[sigma_eps**2,0],[0,sigma_xi**2]])
    nb_s=len(s) ## # of signals 
    
    ## individual and population forecast 
    IndExpNI = np.zeros(T)  ## forecast 
    IndRTExpNI= np.zeros(T)   ## real time expectation
    IndRTVarNI = np.zeros(T)  ## real time variance 
    IndVarNI = np.zeros(T)    ## forecast variance
    Pkalman = np.zeros([T,nb_s])

    PopExpNI = np.zeros(T)
    PopDisgNI = np.zeros(T)
    PopVarNI = np.zeros(T)

    RigidityNI=np.zeros(T)

    IndExpNI[0]= 0               ## starting from a prior equal to unconditional mean
    IndRTVarNI[0]= σ**2/(1-ρ**2)  ## starting from a prior equal to unconditional variance
    IndVarNI[0]= IndVarRE[0]
    Pkalman[0] = [0,0]
    IndRTExpNI[0]=0
    PopExpNI[0]= 0 
    PopDisgNI[0]=0

    for t in range(T-1):
        Pkalman[t+1]=IndRTVarNI[t]*H.T*np.linalg.inv(H*IndRTVarNI[t]*H.T+sigma_v)
        ## individual
        IndRTExpNI[t+1] = (1-Pkalman[t+1]*H)*IndExpNI[t]+ Pkalman[t+1]*s[:,t+1]
        IndExpNI[t+1]= ρ**(h-t)*IndRTExpNI[t+1]
        IndRTVarNI[t+1] =IndRTVarNI[t]-IndRTVarNI[t]*H.T*np.linalg.inv(H*IndRTVarNI[t]*H.T+sigma_v)*H*IndRTVarNI[t]
        IndVarNI[t+1] = ρ**(2*(h-t))*IndRTVarNI[t+1] + IndVarRE[t+1]
        ## population 
        PopExpNI[t+1]=ρ**(h-t)*((1-Pkalman[t+1]*H)*PopExpNI[t]+Pkalman[t+1,0]*s[0,t+1])
        #print('Kalman gain from public signal is '+ str(Pkalman[t+1,1]))
        #print('Kalman gain from private signal is '+str(Pkalman[t+1,0]))
        PopDisgNI[t+1]=(1-Pkalman[t+1]*H)**2*PopDisgNI[t] +  ρ**(2*(h-t))*(Pkalman[t+1,1]**2)*sigma_xi**2
        RigidityNI[t+1] = ρ**(h-t)*(1-Pkalman[t+1]*H)

    ## individual forecast error

    IndFENI = y[T-1]-IndExpNI

    ## individual forecast variance(computed above)

    ## population forecast errror 
    PopFENI = y[T-1]-PopExpNI

    ## population variance 

    PopVarNI = IndVarNI
    
    return{'IndExp':IndExpNI,'IndFE':IndFENI,'IndVar':IndVarNI,
           'IndRTExp': IndRTExpNI, 'IndRTVar':IndRTVarNI,
           'PopExp':PopExpNI,'PopFE':PopFENI,'PopVar':PopVarNI,'PopDisg':PopDisgNI,
           'Rigidity':RigidityNI,'sig_pb':s_pb,'sig_pr':s_pr}
    


# + code_folding=[]
#Invoke NI 

MomNI =NI(T,y,sigma_pb=sigma_y,sigma_pr=sigma_y)  # both signal's ste is equal to y's long-run ste. 

# + code_folding=[]
## Parameters for plots
linesize = 4
linesize2 =2

# +
fig=plt.figure(figsize=(12,12))
plt.suptitle('Illustration of Noisy Information \n'+r'$\sigma_\xi=\sigma_\epsilon=\sigma_y$',fontsize=20)
#plt.subtitle(r'$\sigma_\xi=\sigma_\epsilon=\sigma_y$',fontsize=20)


plt.subplot(2,2,1)
plt.title('Nowcast')
plt.ylim([-6,6])
plt.plot(y,'r-',linewidth=linesize,label=r'$y_{t+k}$')
plt.plot(MomNI['IndRTExp'],'-.',color='dodgerblue',linewidth=linesize,label=r'$y_{i,t+k|t+k}$')
plt.plot(MomNI['sig_pb'],'-.',linewidth=linesize2,label=r'$s^{pb}_t$')
plt.plot(MomNI['sig_pr'],'g--',linewidth=linesize2,label=r'$s^{pr}_{i,t}$')
plt.xlabel(r'$k$',fontsize=15)
plt.ylabel('',fontsize=15)
plt.legend(loc=0,fontsize=13)

plt.subplot(2,2,2)
plt.title('Forecast')
plt.ylim([-6,6])
plt.axhline(y[T-1],label=r'$y_{t+10}$',linewidth=linesize)
plt.plot(MomNI['IndExp'],'-.',color='dodgerblue',linewidth=linesize,label=r'$y_{i,t+10|t+k}$')
#plt.plot(s_pb,'-.',label=r'$s^{pb}_t$')
#plt.plot(s_pr,'-.',label=r'$s^{pr}_{i,t}$')
plt.ylabel('',fontsize=15)
plt.legend(loc='best',fontsize=13)
plt.xlabel(r'$k$',fontsize=15)

plt.subplot(2,2,3)
plt.title('Nowcast Uncertainty')
plt.ylim([-0.5,6])
plt.axhline(0,linewidth=linesize,label=r'$\sigma^2_{i,t+k|t+k}$ in RE')
plt.plot(MomNI['IndRTVar'],'-.',color='dodgerblue',linewidth=linesize,label=r'$\sigma^2_{i,t+k|t+k}$ in NI')
plt.xlabel(r'$k$',fontsize=15)
plt.ylabel('',fontsize=15)
plt.legend(loc='best',fontsize=13)

plt.subplot(2,2,4)
plt.title('Forecast Uncertainty')
plt.plot(IndVarRE,'r-',linewidth=linesize,label=r'$\sigma^2_{i,t+10|t+k}$ in RE')
plt.plot(MomNI['IndVar'],'-.',color='dodgerblue',linewidth=linesize,label=r'$\sigma^2_{i,t+10|t+k}$ in NI')
plt.xlabel(r'$k$',fontsize=15)
plt.ylabel('',fontsize=15)
plt.ylim([-0.5,6])
plt.legend(loc=1,fontsize=13)

plt.savefig('figures/ni_illustration.png')

# + code_folding=[0]
## Plot rigidity for a different values of noises 

noise_ls  = np.array([0.1,1,10])

plt.style.use('ggplot')
fig=plt.figure(figsize=(6,6))
plt.title('Implied Rigidity from Different Models')  # the rigidity parameter is defined as autoregression 
                                  ##  coefficient of change in average forecast 
plt.axhline(1-λ,linewidth=4,label=r'SE Rigidity: $1-\lambda$')

markers = [':','--','-.']

for i in range(len(noise_ls)):
    NewMomNI =NI(T,y,sigma_pb=noise_ls[i]*sigma_y,sigma_pr=noise_ls[i]*sigma_y)
    plt.plot(NewMomNI['Rigidity'],markers[i],linewidth=linesize,label=r'NI Rigidity: $\sigma_\epsilon=\sigma_\xi={} \sigma_y$'.format(noise_ls[i]))
plt.legend(loc='best')
plt.xlabel(r'$k$',fontsize=15)
plt.ylabel('',fontsize=15)
plt.ylabel('Rigidity Parameter',fontsize=1)
plt.savefig('figures/rigidity.png')

# +
plt.style.use('ggplot')
fig=plt.figure(figsize=(16,10))
fig.suptitle("Impulse Response to Shock at t: Individual Moments",fontsize=20)

plt.subplot(2,3,1)
plt.plot(shock,'-',linewidth=linesize2)
plt.xlabel(r'$k$',fontsize=15)
plt.ylim([-0.1,1.1])
plt.title(r'$\omega_{t+k}$',fontsize=15)

plt.subplot(2,3,2)
plt.plot(y,'-',linewidth=linesize2)
plt.xlabel(r'$k$',fontsize=15)
plt.ylim([-0.1,1.1])
plt.title(r'$y_{t+k}$',fontsize=15)

plt.subplot(2,3,3)
plt.plot(IndExpRE,'*',linewidth=linesize+2,label='FIRE')
plt.plot(IndExpSEOld,'-',linewidth=linesize,label='SE: non-updater')
#plt.plot(IndExpSENew,'.',label='SE: updater')
plt.plot(MomNI['IndExp'],'r-.',linewidth=linesize,label='NI')
plt.xlabel(r'$k$',fontsize=15)
plt.title(r'$y_{i,t+10|t+k}$',fontsize=15)
plt.ylim([-0.1,1.1])
plt.legend(loc='best')

plt.subplot(2,3,4)
plt.plot(IndFERE,'*',linewidth=linesize+2,label='FIRE')
plt.plot(IndFESEOld,'-',linewidth=linesize,label='SE: non-updater')
#plt.plot(IndFESENew,'.',label='SE: updater')
plt.plot(MomNI['IndFE'],'r-.',linewidth=linesize,label='NI')
plt.xlabel(r'$k$',fontsize=15)
plt.title(r'$FE_{i,t+10|t+k}$',fontsize=15)
plt.ylim([-0.1,1.1])
plt.legend(loc='best')

plt.subplot(2,3,5)
plt.plot(IndVarRE,'*',linewidth=linesize+2,label='FIRE')
plt.plot(IndVarSEOld,'-',linewidth=linesize,label='SE: non-updater')
#plt.plot(IndVarSENew,'.',label='SE: updater')
plt.plot(MomNI['IndVar'],'r-.',linewidth=linesize,label='NI')
plt.xlabel(r'$k$',fontsize=15)
plt.title(r'$\sigma^2_{i,t+10|t+k}$',fontsize=15)
plt.ylim([-0.1,5])
plt.legend(loc='best')

plt.savefig('figures/ir_indseni.png')

# +
plt.style.use('ggplot')
fig=plt.figure(figsize=(16,10))
fig.suptitle("Impulse Response to Shock at t: Population Moments",fontsize=20)

plt.subplot(2,3,1)
plt.title(r'$\omega_t$',fontsize=15)
plt.plot(shock,'-',linewidth=linesize2)
plt.xlabel(r'$k$',fontsize=15)


plt.subplot(2,3,2)
plt.title(r'$y_{t+k}$',fontsize=15)
plt.plot(y,'-',linewidth=linesize2)
plt.xlabel(r'$k$',fontsize=15)


plt.subplot(2,3,3)
plt.plot(PopExpRE,'*',linewidth=linesize,label='FIRE')
plt.plot(PopExpSE,'-',linewidth=linesize,label='SE')
plt.plot(MomNI['PopExp'],'r-.',linewidth=linesize,label='NI')
plt.xlabel(r'$k$',fontsize=15)
plt.title(r'$\bar y_{t+10|t+k}$',fontsize=15)
plt.ylim([-0.1,1.1])
plt.legend(loc=4)

plt.subplot(2,3,4)
plt.plot(PopFERE,'*',linewidth=linesize+5,label='FIRE')
plt.plot(PopFESE,'-.',linewidth=linesize,label='SE')
plt.plot(MomNI['PopFE'],'r-.',linewidth=linesize,label='NI')
plt.xlabel(r'$k$',fontsize=15)
plt.ylim([-0.1,1.1])
plt.title(r'$\widebar {FE}_{t+10|t+k}$',fontsize=15)
plt.legend(loc=0)

plt.subplot(2,3,5)
plt.plot(PopDisgRE,'*',linewidth=linesize,label='FIRE')
plt.plot(PopDisgSE,'-',linewidth=linesize,label='SE')
plt.plot(MomNI['PopDisg'],'r-.',linewidth=linesize,label='NI')
plt.xlabel(r'$k$',fontsize=15)
plt.title(r'$\widebar {Disg}_{t+10|t+k}$',fontsize=15)
plt.legend(loc=1)

plt.subplot(2,3,6)
plt.plot(PopVarRE,'*',linewidth=linesize,label='FIRE')
plt.plot(PopVarSE,'-',linewidth=linesize,label='SE')
plt.plot(MomNI['PopVar'],'r-.',linewidth=linesize,label='NI')
plt.xlabel(r'$k$',fontsize=15)
plt.title(r'$\widebar {\sigma}^2_{t+10|t+k}$',fontsize=15)
plt.legend(loc=1)
plt.savefig('figures/ir_popseni.png')
# -

# ### Noisy Information (Signal Extraction Models)
#
# A class of so-called noisy information model describes the expectation formation as a process extracting or filtering true fundamental state variable $y_t$ from a sequence of realized signals. The starting assumption is that agent cannot observe the true variable perfectly. Unlike information rigidity model, it is assumed that agents keep track of the realizations of the signals instantaneously all the time. 
#
# We assume agent $i$ observe two signals $s^{pb}$ and $s^{pr}_i$, with $s^{pb}$ being public signal common to all agents, and $s^{pr}_i$ private signals being individual specific. The generating process of two signals are 
#
# \begin{eqnarray}
# \begin{split}
# s^{pb}_t = y_t + \epsilon_t, \quad \epsilon_t \sim N(0,\sigma^2_\epsilon)\\ 
# s^{pr}_{i,t} = y_t + \xi_{i,t} \quad \xi_{i,t} \sim N(0,\sigma^2_\epsilon)
# \end{split}
# \end{eqnarray}
#
# We can stack the two signals into one vector $s_{i,t} = [s^{pb}_t,s^{pr}_{i,t}]'$ and $v_{i,t}= [\epsilon_t,\xi_{i,t}]' = G \times u_{i,t}$, where $u_{i,t}$ is 2 $\times$ 1 following joint standard normal and G is 2 $\times$ 2 defined as follows. 
#
# \begin{eqnarray}
# G = [\begin{array} & \sigma_\epsilon, 0 \\ 0,\sigma_\xi \end{array}]
# \end{eqnarray}
#
# So in a compact form, it can be written as
#
#
# \begin{eqnarray}
# \begin{split}
# s_{i,t} = H y_{t} + G u_{i,t} \\
# \text{where } & H=[1,1]' \quad \\
# \end{split}
# \end{eqnarray}
#
#
# In our general framework, the noisy information implies that the information set $I_{i,t}$ available to individual $i$ at time $t$ only includes past and recent realizations of the signals. The individual density forecast of $y_{t+h}$ is
#
# $$\widehat f_{i,t}(y_{t+h}|I_{i,t}) = \widehat  f_{i,t}(y_{t+h}|s_{i,t},s_{i,t-1}...) \equiv \widehat  f_{i,t|t}(y_{t+h})$$
#
#
# We use $t|k$ to denote the moments at time t based on information(signals) till time $k$. 
#
#
# Then we are ready to apply Kalman Filter in this context. The posterior distribution of $y_{t}$ after seeing all signals till $t$ is 
#
# \begin{eqnarray}
# \widehat  f_{i,t|t}(y_{t+h})  \sim  N(E_{i,t|t}(y_{t+h}), Var_{i,t|t}(y_{t+h}))
# \end{eqnarray}
#
# where the expectation and variances are functions of noisiness of signals and fundamentals. The expectation also depends on the realized values. But this is not the case for variance. 
#
# #### Individual moments 
#
# ##### Expectation 
#
# Now any agent trying to forecast future variables will have to form her expectation of the contemporaneous state variable, $E_{i,t|t}(y_t)$. Then the best h-period ahead forecast is simply iterated h periods forward based on the AR(1) process.  
#
# Thus, we first work out $E^{ni}_{i,t|t}(y_t)$.  
#
# \begin{eqnarray}
# \begin{split}
#  E^{ni}_{i,t|t}(y_{t}) 
#  & =  \underbrace{E^{ni}_{i,t|t-1}(y_{t})}_{\text{prior}} + P_t \underbrace {(s_{i,t|t}-s_{i,t|t-1})}_{\text{innovations to signals}} \\
# & = (1-P_tH) E^{ni}_{i,t|t-1}(y_{t}) + P_ts_{i,t} \\
# & = (1-P_tH) E^{ni}_{i,t|t-1}(y_{t}) + P_t H y_{i,t} + P_t v_{i,t} \\
# \text{where the Kalman gain }  & P_t = [P_{\epsilon,t},P_{\xi,t}]= Var^{ni}_{i,t|t-1} H(H'Var^{ni}_{i,t|t-1} H + \Sigma^v)^{-1} \\
# \text {where } & Var^{ni}_{i,t|t-1} \text{ is the variance of } y_t \text{ based on prior belief}\\
# \text {and } & \Sigma^v = [ \begin{array} & \sigma^2_{\epsilon},  0 \\ 0, \sigma^2_\xi \end{array}]
# \end{split}
# \end{eqnarray}
#
# The h-period ahead forecast is 
#
# \begin{eqnarray}
# \begin{split}
#  E^{ni}_{i,t|t}(y_{t+h}) & = \rho^{h}E^{ni}_{i,t|t}(y_{t+h}) 
# \end{split}
# \end{eqnarray}
#
#
# Individual forecast partially responds to new signals, i.e. $P<1$. $P=1$ is a special case when both signals are perfect thus $\Sigma^v = 0$, then the formula collapses to full rational expectation. 
#
# Now, the rigidity parameter is governed by $1-PH$ with multiple signals. It is a function of variance of $y$ from the prior of previous period and noisiness of the signals. Therefore, it is time variant as the variance is updated by the agent each period.  
#
#
# There are a few important distinctions between noisy information and sticky expectation. 
#
# - First, the persistence of expectation exists at individual level. There is serially correlation between $E^{ni}_{i|t|t}(y_t)$ and $E^{ni}_{i,t|t-1}(y_{t})$, or more generally, between $E^{ni}_{i,t|t}(y_{t+h})$ and $E^{ni}_{i,t|t-1}(y_{t+h})$. This pattern can be only observed from population moments according to sticky expectation models. 
#
#
# To see this, the change in individual forecast from $t-1$ to $t$ is 
#
# \begin{eqnarray}
# \begin{split}
# \Delta E^{ni}_{i,t|t}(y_{t+h}) & = \underbrace{\rho^h (1-PH)\Delta E_{i,t-1|t-1}(y_{t})}_{\text{Lagged response}} + \underbrace{\rho^hPH \Delta y_{i,t} + \rho^h P\Delta v_{i,t}}_{\text{Shocks to signals}}\\
# \end{split}
# \end{eqnarray}
#
# The serial correlation is $\rho^h(1-PH)$, it does not only depend on $PH$, but also the forecast horizon $h$. Therefore, one testable assumption is to see auto regression of change in forecast to see if the coefficient depends on horizon. 
#
# - Second, the expectation adjusts in each period as long as there is new information. In sticky expectation, however, the expectation adjusts only when the agent updates. 
#
#
# ##### Variance 
#
# The posterior variance at time $t$ is a linear function of prior variance and variance of signals. 
#
# \begin{eqnarray}
# \begin{split}
# Var^{ni}_{i,t|t} = Var^{ni}_{i,t|t-1} - Var^{ni}_{i,t|t-1} H'(H Var^{ni}_{i,t-1} H' +\Sigma^v)^{-1} H \Sigma^y_{i,t|t-1} 
# \end{split}
# \end{eqnarray}
#
# There are a few important properties in the variance. 
#
# - First, it does not depend on the realizations of the signal. 
#
# - Second, it decreases unambigously from $t-1$ to $t$. To see this 
#
# \begin{eqnarray}
# Var^{ni}_{i,t|t} - Var^{ni}_{i,t|t-1} = - Var^{ni}_{i,t|t-1} H'(H Var^{ni}_{i,t-1} H' +\Sigma^v)^{-1} H Var^{ni}_{i,t|t-1} <0
# \end{eqnarray}
#
# These two properties carry through to the h-period ahead forecast as well. As the forecast variance is the following 
#
# \begin{eqnarray}
#    Var^{ni}_{i,t|t} (y_{t+h}) = \rho^{2h} \underbrace{Var^{ni}_{i,t}(y_{t})}_{Var^{ni}_{i,t|t}} + \sum^{h}_{s=0}\rho^{2s} \sigma^2_{\omega}
# \end{eqnarray}
#
#
# \begin{eqnarray}
#    \Delta Var^{ni}_{i,t|t} (y_{t+h}) = \rho^{2h}\Delta Var^{ni}_{i,t|t} - \rho^{2h} \sigma^2_{\omega}
# \end{eqnarray}
#
# From $t$ to $t+1$, when $h\geq 1$, the decline in variance come from two sources. The first source is the pure gain from the new signals, i.e. $\Delta Var^{ni}_{i,t|t}$. It is scalled by the factor $\rho^{2h}$. The second source is present in full information rational expectation model: as time goes from $t-1$ to $t$, there is a reduction of uncertainty about $\omega_t$.
#
# #### Population moments
#
# ##### Average forecast 
#
#
# \begin{eqnarray}
# \begin{split}
# \bar E^{ni}_{t|t} (y_{t+h}) & = \rho^h [(1-PH) \underbrace{\bar E^{ni}_{t-1}(y_{t+h})}_{\text{Average prior}} + P \underbrace{\bar s_{t}}_{\text{Average Signals}}] \\
# & = (1-PH) \bar E^{ni}_{t-1}(y_{t+h}) + P [\epsilon_t, 0]' \\
# & = (1-PH) \bar E^{ni}_{t-1}(y_{t+h}) + P_\epsilon\epsilon_t
# \end{split}
# \end{eqnarray}
#
# ##### Change in average forecast 
#
# \begin{eqnarray}
# \begin{split}
# \Delta \bar E^{ni}_t|t (y_{t+h}) & = \rho^h (1-PH) \Delta \bar E^{ni}_{t-1}(y_{t+h}) + \rho^h P \Delta \epsilon_{t}
# \end{split}
# \end{eqnarray}
#
#
# Same to the individual forecast, the change in average forecasts has serial correlation with the same auto regression parameter $\rho^h(1-PH)$.  
#
# ##### Cross-sectional disagreement
#
# In this model, the only disagreements across agents come from the difference in realized private signals. Therefore, in short-cut, the disagreement is essentialy the weighted sum of all past dispersions due to private signals, and the weight for each past period is proprotional to the updating weight (1-PH)
#
# \begin{eqnarray}
# \begin{split}
# Disg^{ni}_t(y_{t+h}) & = E((E_{i,t|t}(y_{t+h}) - \bar E_t(y_{t+h}))^2) \\
# & = \rho^{2h}\sum^{\infty}_{\tau=0}[(1-PH)\rho]^{2\tau}P^2_{\xi}\sigma^2_\omega \\
# & = \rho^{2h}\frac{1}{1-(1-PH)^2\rho^2}P^2_\xi\sigma^2_\omega
# \end{split}
# \end{eqnarray}
#
#
# Several properties. 
#
# - First, the disagreements increase with the forecast horizon. 
# - Second, the disagreements depends on noisiness private signals, but not on that of public signals and the variance of the true variable $y$. 
# - Third, similar to sticky expectation model, the disagreements also increase with the rigidity parameter $P$ in this model.
#
# ##### Average variance 
#
# Since the variance does not depend on signals and the precision is the same aross the agents, average variance is equal to the variance of each individual. 
#
# \begin{eqnarray}
# \begin{split}
# \bar Var^{ni}_t (y_{t+h}) = \bar Var^{ni}_{i,t+h|t}
# \end{split}
# \end{eqnarray}
#
# Also, the same as in the individual variance, the variance unambiguiously drop as time goes by. 
#
# \begin{eqnarray}
# \Delta Var^{ni}_t(y_{t+h}) < 0 
# \end{eqnarray}
#
# ##### Summary of predictions from noisy information 
#
# - Invididual expectation adjusts in each period, but only partially adjusts to new information. 
# - Unlike sticky expectation, slugishness in adjustment or serial correlation of adjustment exists in individual level. The correlation parameter decreases with forecast horizon, which is not the case in sticky expectation.
# - Individual variance unambiguously drops each period as one approaches the period of realization. In sticky expectation, it increases regardless of updating or not. 
# - Population average forecast partially adjusts to news and has serial correlation as the individual level. 
# - Population disagreements rise in each period as time approaches the period of realization. Disagreetments will never be zero. 
# - Average variance declines unambiguously each period. 

# + code_folding=[4, 19, 28, 30, 55, 58, 70, 77, 83]
from SMMEst import SteadyStateVar,Pkalman

rho,sigma = 0.95,0.1

def FE2_NI(sigma_pb,
           sigma_pr):
    now_var_ss = SteadyStateVar(np.array([rho,
                                          sigma]),
                                np.array([sigma_pb,
                                          sigma_pr])
                               )
    P_ss = Pkalman(np.array([rho,
                             sigma]),
                  np.array([sigma_pb,
                            sigma_pr]),
                  now_var_ss)
    P_pb, P_pr = P_ss
    return (rho**2*P_pb**2*sigma_pb**2+sigma**2)/(P_pb+P_pr)**2

def Var_NI(sigma_pb,
                 sigma_pr):
    now_var_ss = SteadyStateVar(np.array([rho,
                                          sigma]),
                                np.array([sigma_pb,
                                          sigma_pr])
                               )
    return rho**2*now_var_ss + sigma**2

def Disg_NI(sigma_pb,
                 sigma_pr):
    now_var_ss = SteadyStateVar(np.array([rho,
                                          sigma]),
                                np.array([sigma_pb,
                                          sigma_pr])
                               )
    P_ss = Pkalman(np.array([rho,
                             sigma]),
                  np.array([sigma_pb,
                            sigma_pr]),
                  now_var_ss)
    P_pb, P_pr = P_ss
    
    return (rho**2*P_pr**2)/(1-(1-P_pb-P_pr)**2*rho**2)*sigma_pr**2
                                       
## need to plot 3d 
#plt.title('NI')

import matplotlib.pyplot as plt 
sigma_pb_ = np.linspace(0.01, 0.8, 50)
sigma_pr_ = np.linspace(0.01, 0.8, 50)

sigma_pbs,sigma_prs = np.meshgrid(sigma_pb_,sigma_pr_)

row,col  = sigma_pbs.shape

FE2_NIs = np.array([FE2_NI(sigma_pbs[i,j],
                 sigma_prs[i,j]) for i in range(row) for j in range(col)]).reshape((row,col))
                 
Var_NIs = Var_NI(sigma_pbs,
                 sigma_prs)

Disg_NIs = np.array([Disg_NI(sigma_pbs[i,j],
                 sigma_prs[i,j]) for i in range(row) for j in range(col)]).reshape((row,col))

fig = plt.figure(figsize = (20,10))
#fig.colorbar(surf)    
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.invert_xaxis()
ax.set_title(r'$FE^2$, Var and Disg of NI')

surf = ax.plot_surface(sigma_pbs, 
                       sigma_prs, 
                       FE2_NIs, 
                       label='FE',
                       #cmap = plt.cm.red
                      color='red')

surf2 = ax.plot_surface(sigma_pbs, 
                       sigma_prs, 
                       Var_NIs, 
                       #cmap = plt.cm.summer
                       color='blue')

surf3 = ax.plot_surface(sigma_pbs, 
                       sigma_prs, 
                       Disg_NIs, 
                       #cmap = plt.cm.winter
                       color='gray')

# Set axes label
ax.set_xlabel(r'$\sigma_{pb}$',size=20)
ax.set_ylabel('$\sigma_{pr}$',size=20)
ax.set_zlabel(r'moments',size=20)
ax.view_init(elev=10,
             azim=50)

# -

# ### Diagnostic expectations (DE)
#
# #### Individual expectations 
#
# \begin{eqnarray}
# E^{de}_{i,t}(y_{t+h}) = \rho^h y_t + \theta_i(\rho^h y_t - E^{de}_{i,t-1}(y_{t+h}))
# \end{eqnarray}
#
# where $\theta_i>0$ is the overreaction parameter. 
#
# #### Individual forecast errors 
#
# \begin{eqnarray}
# \begin{split}
#  FE^{de}_{i,t}(y_{t+h})& = E^{de}_{i,t}(y_{t+h}) - y_{t+h}\\
# & =  \rho^h y_t -y_{t+h} + \theta_i(\rho^h y_t - E^{de}_{i,t-1}(y_{t+h})) \\
# & = \rho^h y_t - y_{t+h} + \theta_i (\rho^hy_t-y_{t+h}-FE^{de}_{i,t+h|t-1}) \\
# & = \hat FE^{*}_{t+h|t} +\theta_i (\rho^hy_t-y_{t+h}-FE^{de}_{i,t+h|t-1}) \\
# & = (1+\theta_i) FE^{*}_{t+h|t} - \theta_i FE^{de}_{i,t+h|t-1} \\
# & = (1+\theta_i) FE^{*}_{t+h|t} - \theta_i (\rho FE^{de}_{i,t+h-1|t-1}-\omega_{t+h}) \\
# & = (1+\theta_i) FE^{*}_{t+h|t} - \theta_i \rho FE^{de}_{i,t+h-1|t-1} + \theta_i \omega_{t+h}  \\
# & = (1+\theta_i) FE^{*}_{t+h|t} + \theta_i(\omega_{t+h}- \rho FE^{de}_{i,t+h-1|t-1}) \\
# & = (1+\theta_i) FE^{*}_{t+h-1|t} + (1+\theta_i)(-\omega_{t+h})+\theta_i(\omega_{t+h}- \rho FE^{de}_{i,t+h-1|t-1}) \\
# & = (1+\theta_i) FE^{*}_{t+h-1|t} -\omega_{t+h}-\theta_i\rho FE^{de}_{i,t+h-1|t-1} \\
# & = FE^{*}_{t+h|t} +\theta_iFE^{*}_{t+h-1|t} -\theta_i\rho FE^{de}_{i,t+h-1|t-1}\\
# & = FE^{*}_{t+h|t} +\theta_i(FE^{*}_{t+h-1|t}- \rho FE^{de}_{i,t+h-1|t-1}) 
# \end{split}
# \end{eqnarray}
#
# when $h=1$, the equation collapses to 
#
# \begin{eqnarray}
# \begin{split}
#  FE^{de}_{i,t}(y_{t+1})&  = (1+\theta_i) FE^{*}_{i,t+1|t} - \theta_i \rho FE^{de}_{i,t|t-1} + \theta_i \omega_{t+1}  \\
#  & = (1+\theta_i) (-\omega_{t+1} ) - \theta_i \rho FE^{de}_{i,t|t-1} + \theta_i \omega_{t+1} \\
#  & =  FE^{*}_{i,t+1|t} - \theta_i \rho FE^{de}_{i,t|t-1}
# \end{split}
# \end{eqnarray}
#
# The variance of DE model is the same as FIRE model, at both individual and population level.
#
# #### Population moments
#
# Average forecast errors $\hat{FE}^{de}_{t+h|t}$ takes the same form as the individual forecast errors except for substituting the individual specific $\theta_i$ with the average $\theta$. 
#
# The variance of population forecast error is equal to the following.
#
#
# \begin{eqnarray}
# \begin{split}
#  FE^{de2}_{\bullet+h|\bullet}&  =\frac{(1+\theta)^2}{1+\theta^2\rho^2}{FE}^{*2}_{\bullet+h-1|\bullet} + \frac{\sigma^2_\omega}{(1+\theta^2\rho^2)}
# \end{split}
# \end{eqnarray}
#
# The average uncertainty is also equal to FIRE benchmark at the population level. 
#
# Disagreement is the following, which is non zero as long as there is heterogeneity in degrees of overreaction.
#
# \begin{eqnarray}
# \begin{split}
# Disg^{de}_{t+h|t} & = Var_t(\rho^h y_t + \theta_i(\rho^h y_t - E^{de}_{i,t-1}(y_{t+h})))  \\
#  & =  Var_t(\theta_i(\rho^h y_t - E^{de}_{i,t-1}(y_{t+h})))   \\
#   & =  \rho^{2h}y^2_t \sigma^2_\theta - \rho^2 Disg^{de}_{t+h-1|t-1}  \\
# \end{split}
# \end{eqnarray}
#
# In steady state, it is the following.
#
# \begin{eqnarray}
# \begin{split}
#  & Disg^{de}_{\bullet+h|\bullet} & = \rho^{2h} y^2_\bullet \sigma^2_\theta + \rho^2 Disg^{de}_{\bullet+h|\bullet}  \\
# & \rightarrow Disg^{de}_{\bullet+h|\bullet} & = \frac{\rho^{2h} \sigma^2_\theta}{1-\rho^2} \frac{\sigma^2_\omega}{1-\rho^2}
# \end{split}
# \end{eqnarray}

# + code_folding=[]
## DE

theta_hats = np.linspace(0.0,0.5,30)
sigma_thetas = 0.8

rho,sigma = 0.95,0.1

def FE2_DE(theta_hat,sigma_theta):
    return sigma**2/(1+theta_hat**2*rho**2)

def Disg_DE(theta_hat,sigma_theta):
    return rho**2*sigma_theta**2*sigma**2/(1-rho*2)**2

FE_DE2_ratios = FE2_DE(theta_hats,sigma_thetas)/sigma**2
Var_DE_ratios = np.ones(len(theta_hats))
Disg_DE_ratios = Disg_DE(theta_hats,sigma_thetas)/sigma**2

## plot 
plt.title('DE')
plt.plot(theta_hats,
         FE_DE2_ratios,
        label=r'$FE^2_{\bullet+1|\bullet}$')
plt.plot(theta_hats,
         Var_DE_ratios,
        label=r'$Var_{\bullet+1|\bullet}$')
plt.plot(theta_hats,
         Disg_DE_ratios*np.ones(len(theta_hats)),
        label=r'$Disg_{\bullet+1|\bullet}$')
plt.legend(loc=1)
plt.xlabel(r'$\theta$')
