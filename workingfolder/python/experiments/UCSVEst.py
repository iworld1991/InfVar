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

# ## Stock-Watson Estimation of UC-SV Model
#
# - Translated from the Matlab code by Jonathan Wright

import numpy as np
import matplotlib.pyplot as plt
from GMMEstSV import UCSV_simulator as sv_sim

# + {"code_folding": []}
## default parameters 

vague = 1000
burnin = 100
ndraw = burnin + 5000

## paramters for log-chi-squared errors
r_p = .086
r_m1 = -8.472
r_m2 = -0.698
r_sig2 = 1.411
r_sig = np.sqrt(r_sig2)
        
## parameters for RW innovation variance
tau1 = .20
tau2 = .20
q_p = 1.0
q1 = tau1**2
q2 = tau2**2
# -

params_default = np.array([r_p,r_m1,r_m2,r_sig,r_sig2,q_p,q1,q2,tau1,tau2,vague])


# + {"code_folding": [1, 7, 39, 103]}
class UCSVEst:
    def __init__(self,
                 y,
                 params = params_default):
        self.params = params
        self.y = y
        
    def draw_eps_eta(self,
                     var_eps_n,
                     var_eta_n):
        params = self.params
        
        y = self.y
        n = len(y)
        ltone = np.tril(np.ones([n,n]))
        cov_eps = np.diag(var_eps_n)
        cov_tau = ltone*cov_eps*(ltone.T)
        diag_y  = np.diag(cov_tau)+var_eta_n
        cov_y = cov_tau 
        for i in range(n):
            cov_y[i,i] = diag_y[i]
        kappa = cov_tau*np.linalg.inv(cov_y)
        mutau_y = kappa*y
        covtau_y = cov_tau - kappa*cov_tau.T
        print(covtau_y)
        #print(covtau_y)
        chol_covtau_y = np.linalg.cholesky(covtau_y)
        tau = mutau_y + chol_covtau_y.T*np.random.randn(n)
        print(tau.shape)
        eta = y - tau 
        #print(tau[0].shape)
        eps = np.concatenate([tau[0],tau[1:] - tau[0:-2]])
        
        self.eps = eps
        self.eta = eta
        self.tau = tau
        
        return self.eps,self.eta,self.tau
    
    def draw_var(self,
                 x,
                 r_pt,
                 q_pt,
                 var_min,
                 params):
    
        n = len(x)
        bsum = np.tril(np.ones([n+1,n+1])
                      )
        lnres2 = np.log(x**2)
        
        ## initial draws of indicators
        tmp = np.random.uniform(0,1,n)
        ir = tmp<r_pt
        temp = np.random.uniform(0,1,n)
        iq = temp<q_pt
        
        ## compute system parameters given indicators
        mut = (ir*r_m1) + ((1-ir)*r_m2)
        qt = (iq*q1) + ((1-iq)*q2)
        
        ## compute covariance matrix
        vd = np.diag([vague,qt])
        valpha = bsum*vd*bsum.T
        vy = valpha[1:n,1:n]
        cv = valpha[0:n,1:n]
        diagvy = np.diag(vy)+r_sig2
        for i in range(n):
            vy[i,i] = diagvy[i]
        kgain = cy*np.linalg.inv(vy)
        
        # compute draws of state and shocks
        ye = lnres2 - mut
        ahat0 = kgain*ye
        ahat1 = ahat0[1:n]
        vhat0 = valpha - kgain*cy.T
        cvhat0 = np.linalg.cholesky(vhat0)
        adraw0 = ahat0 + cvhat0.T*np.random.randn(n+1)
        adraw1 = adraw0[1:n]
        vardraw = np.exp(adraw1)
        
        edraw = lnres2-adraw1
        udraw = adraw0[1:] - adraw0[:n]

        # Compute Mixture Probabilities 
        f1 = np.exp((-0.5)*(((edraw-r_m1)/r_sig2)**2)  )
        f2 = np.exp((-0.5)*(((edraw-r_m2)/r_sig2)**2)  )
        fe = r_p*f1 + (1-r_p)*f2
        r_pt = (r_p*f1)/fe
        
        # u shocks -- Means are both zero%
        f1 = (1/tau1)*np.exp((-0.5)*((udraw/tau1)**2) )
        f2 = (1/tau1)*np.exp((-0.5)*((udraw/tau2)**2) )
        fu = q_p*f1 + (1-q_p)*f2
        q_pt = (q_p*f1)/fu
                

        self.vardraw = vardraw
        self.r_pt = r_pt
        self.q_pt = q_pt
        
        return self.vardraws,self.r_pt,self.q_pt
    
    def stockwatson(self,
                    var_eps_min,
                    var_eta_min):
        y = self.y
        n = len(y)
        params = self.params
        
        ## parameters for initial conditions, bounds and so forth
        tau0 = np.mean(y[:4])
        dy = y[1:] - y[0:-1]
        var_dy = np.std(dy)**2
        
        ## lower bounds on variance
        #var_eta_min = 0.015*var_dy
        #var_eps_min = 0.005*var_dy
        
        ## initial values
        var_eps_initial = var_dy/3
        var_eta_initial = var_dy/4
        
        y = y - tau0
        r_pt_eps = r_p*np.ones(n)
        q_pt_eps = q_p*np.ones(n)
        r_pt_eta = r_p*np.ones(n)
        q_pt_eta = q_p*np.ones(n)
        var_eps_n = var_eps_initial*np.ones(n)
        var_eta_n = var_eta_initial*np.ones(n)
        
        sd_eps_save = np.zeros([n,ndraw-burnin])
        sd_eta_save = np.zeros([n,ndraw-burnin])
        tau_save = np.zeros([n,ndraw-burnin])
        for idraw in range(ndraw):
            eps,eta,tau = self.draw_eps_eta(var_eps_n,var_eta_n)
            var_eps_n,r_pt_eps,q_pt_eps = self.draw_var(eps,r_pt_eps,q_pt_eps,var_eps_min,params)
            var_eta_n,r_pt_eta,q_pt_eta = self.draw_var(eta,r_pt_eta,q_pt_eta,var_eta_min,params)
            if idraw > burnin:
                sd_eps_n = np.sqrt(var_eps_n)
                sd_eta_n = np.sqrt(var_eta_n)
                sd_eps_save[:,idraw-burnin] = sd_eps_n
                sd_eta_save[:,idraw-burnin] = sd_eta_n
                tau_save[:,idraw-burnin] = tau
        sd_eps = np.zeros(n)
        sd_eta = np.zeros(n)
        tau = np.zeros(n)
        for i in range(n):
            sd_eps[i] = median(sd_eps_save[i,:])
            sd_eta[i] = median(sd_eta_save[i,:])
            tau[i] = median(tau_save[i,:])
        tau = tau + tau0
        
        self.sd_eps = sd_eps
        self.sd_eta = sd_eta
        self.tau = tau
        return self.sd_eps,self.sd_eta,self.tau

# + {"code_folding": []}
## simulate some uc-sv model 

sv_instance = sv_sim(0.2,
                     nobs = 1000,
                     eta0 = 2)
y_fake = sv_instance[0][200:]
tau_fake = sv_instance[1][200:]
var_eps_fake = sv_instance[2][200:]**2
var_eta_fake = sv_instance[3][200:]**2
# -

y_fake2 = np.random.uniform(0,1,100)

# + {"code_folding": []}
## test UCSVEst

est_fake = UCSVEst(y_fake2,
                   params = params_default)

# +
#var_eps_fake = np.random.uniform(0,1,101)
#var_eta_fake = np.random.uniform(0,1,101)

#est_fake.stockwatson(0.01,0.01)
# -



