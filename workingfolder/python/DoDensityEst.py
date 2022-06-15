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

# ### Density Estimation 
#
# - Following [Manski et al.(2009)](https://www.tandfonline.com/doi/abs/10.1198/jbes.2009.0003)
# - Three cases 
#    - case 1. 3+ intervals with positive probabilities, or 2 intervals with positive probabilities but open-ended from either end, to be fitted with a generalized beta distribution
#    - case 2. exactly 2 adjacent and close-ended bins positive probabilities, to be fitted with a triangle distribution 
#    - case 3. __one or multiple__ adjacent intervals with equal probabilities, to be fitted with a uniform distribution
#    - cases excluded for now:
#      - nonadjacent bins with positive probabilities with bins with zero probs in between 
#      -  only one bin with positive probabilities at either end 
#    

from scipy.stats import beta
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# +
## import all functions from DensityEst 
## see DensityEst.ipynb for detailed explanations 

from DensityEst import GeneralizedBetaEst,GeneralizedBetaStats,TriangleEst,TriangleStats,UniformEst,UniformStats,SynDensityStat
# -

# ### Case 1. Generalized Beta Distribution

# ### Case 2. Isosceles Triangle distribution
#
# Two adjacent intervales $[a,b]$,$[b,c]$ are assigned probs $\alpha$ and $1-\alpha$, respectively. In the case of $\alpha<1/2$, we need to solve parameter $t$ such that $[b-t,c]$ is the interval of the distribution. Denote the height of the trangle distribution $h$. Then following two restrictions need to satisfy
#
# \begin{eqnarray}
# \frac{t^2}{t+c-b} h = \alpha \\
# (t+(c-b))h = 2
# \end{eqnarray}
#
# The two equations can solve $t$ and $h$
#
# $$\frac{t^2}{(t+c-b)^2}=\alpha$$
#
# $$t^2 = \alpha t^2 + 2\alpha t(c-b) + \alpha(c-b)^2$$
#
# $$(1-\alpha) t^2 - 2\alpha(c-b) t - \alpha(c-b)^2=0$$
#
# $$\implies t =\frac{2\alpha(c-b)+\sqrt{4\alpha^2(c-b)^2+4(1-\alpha)\alpha(c-b)^2}}{2(1-\alpha)} = \frac{\alpha(c-b)+(c-b)\sqrt{\alpha}}{(1-\alpha)}$$
#
# $$\implies h = \frac{2}{t+c-b}$$
#
# Rearrange to solve for c:
# $$ \implies c = (t - b) - \frac{2}{h} $$
# $$ \implies c = ub - \frac{2}{h} $$

# #### pdf of a triangle distribution
#
# \begin{eqnarray}
# f(x)= & 1/2(x-lb) \frac{x-lb}{(ub+lb)/2}h \quad \text{if } x <(lb+ub)/2 \\
# & = 1/2(ub-x) \frac{ub-x}{(ub+lb)/2}h \quad \text{if } x \geq(lb+ub)/2
# \end{eqnarray}
#
# \begin{eqnarray}
# & Var(x) & = \int^{ub}_{lb} (x-(lb+ub)/2)^2 f(x) dx \\
# & & = 2 \int^{(ub+lb)/2}_{lb} (x-(lb+ub)/2)^2 (x-lb) \frac{x-lb}{(ub+lb)/2}h dx
# \end{eqnarray}
#
#

# ### Case 3. Uniform Distribution

# ### Test using made-up data

# + code_folding=[]
## test 1: GenBeta Dist
sim_bins= np.array([0,0.2,0.32,0.5,1,1.2])
sim_probs=np.array([0,0.5,0.05,0.05,0.4])
GeneralizedBetaEst(sim_bins,sim_probs)

# + code_folding=[0]
## test 2: Triangle Dist
sim_bins2 = np.array([0,0.2,0.32,0.5,1,1.2])
sim_probs2=np.array([0,0.2,0.8,0,0])
TriangleEst(sim_bins2,sim_probs2)

# + code_folding=[]
## test 3: Uniform Dist with one interval
sim_bins3 = np.array([0,0.2,0.32,0.5,1,1.2])
sim_probs3 = np.array([0,0,1,0,0])
para_est= UniformEst(sim_bins3,sim_probs3)
print(para_est)
UniformStats(para_est['lb'],para_est['ub']) 
# -

## test 4: Uniform Dist with multiple adjacent bins with same probabilities 
sim_bins3 = np.array([0,0.2,0.32,0.5,1,1.2])
sim_probs3=np.array([0.2,0.2,0.2,0.2,0.2])
para_est= UniformEst(sim_bins3,sim_probs3)
print(para_est)
UniformStats(para_est['lb'],para_est['ub']) 


## test 5: Uniform Dist with multiple non-adjacent bins with equal probabilities
sim_bins3 = np.array([0,0.2,0.32,0.5,1,1.2])
sim_probs3=np.array([0,0.5,0,0.5,0])
para_est= UniformEst(sim_bins3,sim_probs3)
print(para_est)
UniformStats(para_est['lb'],para_est['ub']) 

# ### Test with simulated data from known distribution 
# - we simulate data from a true beta distribution with known parameters
# - then we estimate the parameters with our module and see how close it is with the true parameters 

# + code_folding=[]
## simulate a generalized distribution
sim_n=1000
true_alpha,true_beta,true_loc,true_scale=0.5,0.6,0.1,2
sim_data = beta.rvs(true_alpha,true_beta,
                    loc=true_loc,
                    scale=true_scale,
                    size=sim_n)
sim_bins2=plt.hist(sim_data)[1]
sim_probs2=plt.hist(sim_data)[0]/sim_n
sim_est=GeneralizedBetaEst(sim_bins2,sim_probs2)

print('Estimated parameters',sim_est)

print('Estimated moments:',GeneralizedBetaStats(sim_est[0],
                          sim_est[1],
                          sim_est[2],
                          sim_est[3]))

print('True simulated moments:',
      np.mean(sim_data),
     np.std(sim_data)**2,
     np.std(sim_data)
     )

# + code_folding=[0]
## plot the estimated generalized beta versus the histogram of simulated data drawn from a true beta distribution 
sim_x = np.linspace(true_loc,true_loc+true_scale,sim_n)
sim_pdf=beta.pdf(sim_x,sim_est[0],sim_est[1],loc=true_loc,scale=true_scale)
plt.plot(sim_x,sim_pdf,label='Estimated pdf')
plt.hist(sim_data,density=True,label='Dist of Simulated Data')
plt.legend(loc=0)

# + code_folding=[]
## testing the synthesized estimator function using an arbitrary example created above
SynDensityStat(sim_bins2,sim_probs2)['variance']
SynDensityStat(sim_bins2,sim_probs2)['iqr1090']

# + code_folding=[]
### loading probabilistic data  
IndSPF=pd.read_stata('../SurveyData/SPF/individual/InfExpSPFProbIndQ.dta')   
# SPF inflation quarterly 
# 2 Inf measures: CPI and PCE
# 2 horizons: y-1 to y  and y to y+1

# + code_folding=[]
## survey-specific parameters 
nobs=len(IndSPF)
SPF_bins=np.array([-10,0,0.5,1,1.5,2,2.5,3,3.5,4,10])
print("There are "+str(len(SPF_bins)-1)+" bins in SPF")

# + code_folding=[]
##############################################
### attention: the estimation happens here!!!!!
###################################################

## creating positions 
index  = IndSPF.index
columns=['PRCCPIMean0','PRCCPIVar0', 'PRCCPIStd0','PRCCPIIqr10900',
         'PRCCPIMean1','PRCCPIVar1','PRCCPIStd1','PRCCPIIqr10901',
         'PRCPCEMean0','PRCPCEVar0','PRCPCEStd0','PRCPCEIqr10900',
         'PRCPCEMean1','PRCPCEVar1','PRCPCEStd1','PRCPCEIqr10901']
IndSPF_moment_est = pd.DataFrame(index=index,columns=columns)

## Invoking the estimation

for i in range(nobs):
    print(i)
    ## take the probabilities (flip to the right order, normalized to 0-1)
    PRCCPI_y0 = np.flip(np.array([IndSPF.iloc[i,:]['PRCCPI'+str(n)]/100 for n in range(1,11)]))
    print(PRCCPI_y0)
    PRCCPI_y1 = np.flip(np.array([IndSPF.iloc[i,:]['PRCCPI'+str(n)]/100 for n in range(11,21)]))
    print(PRCCPI_y1)
    PRCPCE_y0 = np.flip(np.array([IndSPF.iloc[i,:]['PRCPCE'+str(n)]/100 for n in range(1,11)]))
    print(PRCCPI_y0)
    PRCPCE_y1 = np.flip(np.array([IndSPF.iloc[i,:]['PRCPCE'+str(n)]/100 for n in range(11,21)]))
    print(PRCCPI_y1)
    if not np.isnan(PRCCPI_y0).all():
        stats_est=SynDensityStat(SPF_bins,PRCCPI_y0)
        if stats_est is not None and len(stats_est)>0:
            IndSPF_moment_est['PRCCPIMean0'][i]=stats_est['mean']
            print(stats_est['mean'])
            IndSPF_moment_est['PRCCPIVar0'][i]=stats_est['variance']
            print(stats_est['variance'])
            IndSPF_moment_est['PRCCPIStd0'][i]=stats_est['std']
            print(stats_est['std'])
            IndSPF_moment_est['PRCCPIIqr10900'][i]=stats_est['iqr1090']
            print(stats_est['iqr1090'])
        else:
            IndSPF_moment_est['PRCCPIMean0'][i] = np.nan
            IndSPF_moment_est['PRCCPIVar0'][i] = np.nan
            IndSPF_moment_est['PRCCPIStd0'][i] = np.nan
            IndSPF_moment_est['PRCCPIIqr10900'][i] =np.nan
    if not np.isnan(PRCCPI_y1).all():
        stats_est=SynDensityStat(SPF_bins,PRCCPI_y1)
        if  stats_est is not None and len(stats_est):
            print(stats_est['mean'])
            IndSPF_moment_est['PRCCPIMean1'][i]=stats_est['mean']
            print(stats_est['variance'])
            IndSPF_moment_est['PRCCPIVar1'][i]=stats_est['variance']
            print(stats_est['std'])
            IndSPF_moment_est['PRCCPIStd1'][i]=stats_est['std']
            print(stats_est['iqr1090'])
            IndSPF_moment_est['PRCCPIIqr10901'][i]=stats_est['iqr1090']
        else:
            IndSPF_moment_est['PRCCPIMean1'][i] = np.nan
            IndSPF_moment_est['PRCCPIVar1'][i] = np.nan
            IndSPF_moment_est['PRCCPIStd1'][i] = np.nan
            IndSPF_moment_est['PRCCPIIqr10901'][i] =np.nan
    if not np.isnan(PRCPCE_y0).all():
        stats_est=SynDensityStat(SPF_bins,PRCPCE_y0)
        if  stats_est is not None and len(stats_est)>0:
            print(stats_est['mean'])
            IndSPF_moment_est['PRCPCEMean0'][i]=stats_est['mean']
            print(stats_est['variance'])
            IndSPF_moment_est['PRCPCEVar0'][i]=stats_est['variance']
            print(stats_est['std'])
            IndSPF_moment_est['PRCPCEStd0'][i]=stats_est['std']
            print(stats_est['iqr1090'])
            IndSPF_moment_est['PRCPCEIqr10900'][i]=stats_est['iqr1090']
        else:
            IndSPF_moment_est['PRCPCEMean0'][i] = np.nan
            IndSPF_moment_est['PRCPCEVar0'][i] = np.nan
            IndSPF_moment_est['PRCPCEStd0'][i] = np.nan
            IndSPF_moment_est['PRCPCEIqr10900'][i] =np.nan
    if not np.isnan(PRCPCE_y1).all():
        stats_est=SynDensityStat(SPF_bins,PRCPCE_y1)
        if  stats_est is not None and len(stats_est)>0:
            print(stats_est)
            print(stats_est['mean'])
            IndSPF_moment_est['PRCPCEMean1'][i]=stats_est['mean']
            print(stats_est['variance'])
            IndSPF_moment_est['PRCPCEVar1'][i]=stats_est['variance']
            print(stats_est['std'])
            IndSPF_moment_est['PRCPCEStd1'][i]=stats_est['std']
            print(stats_est['iqr1090'])
            IndSPF_moment_est['PRCPCEIqr10900'][i]=stats_est['iqr1090']
        else:
            IndSPF_moment_est['PRCPCEMean1'][i] = np.nan
            IndSPF_moment_est['PRCPCEVar1'][i] = np.nan
            IndSPF_moment_est['PRCPCEStd1'][i] = np.nan
            IndSPF_moment_est['PRCPCEIqr10901'][i] =np.nan
# -

### exporting moments estimates to pkl
IndSPF_est = pd.concat([IndSPF,IndSPF_moment_est], join='inner', axis=1)
IndSPF_est.to_pickle("./IndSPFDstIndQ.pkl")
IndSPF_pk = pd.read_pickle('./IndSPFDstIndQ.pkl')

# +
## convert all to numeric 

type_map = {columns[i]:'float' for i,column in enumerate(columns)}
IndSPF_pk =IndSPF_pk.astype(type_map)

# +
## export to stata 

IndSPF_pk.to_stata('../SurveyData/SPF/individual/InfExpSPFDstIndQ.dta')

# + code_folding=[]
### Robustness checks: focus on big negative mean estimates 
nan_est = IndSPF_pk['PRCCPIMean0'].isna()
missing_data = IndSPF_pk['PRCCPI1'].isna() ## no data to estimate in the first place 

print(str(sum(nan_est))+' missing estimates')
print(str(sum(missing_data))+' of which is due to missing data')

print('\n')
print('All nan estimates due to other reasons\n')

ct=0
figure=plt.plot()
for id in IndSPF_pk.index[(nan_est) & (~missing_data)]:
    print(id)
    print(IndSPF_pk['PRCCPIMean1'][id])
    sim_probs_data= np.flip(np.array([IndSPF['PRCCPI'+str(n)][id]/100 for n in range(11,21)]))
    plt.bar(SPF_bins[1:],sim_probs_data)
    print(sim_probs_data)
    ## do estimation again 
    stats_est=SynDensityStat(SPF_bins,sim_probs_data)
    if stats_est is not None:
        print(stats_est['mean'])
    else:
        print('Estimation is None!!!!')