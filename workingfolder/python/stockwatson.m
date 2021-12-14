function [sd_eps,sd_eta,tau]=stockwatson(y,var_eps_min,var_eta_min);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Code for computing the permanent-transitory decomposition in Stock     %
%-Watson, JMCB, 2007, by MCMC.                                          %
%Written by Mark Watson in Gauss (incorporating revision of May 3 2007) %
%Translated to Matlab by Jonathan Wright                                %
%Input: The inflation series                                            %
%Outputs:                                                               %
%sd_eps: Standard deviation of permanent component                      %
%sd_eta: Standard deviation of transitory component                     %
%tau: Estimated permanent component                                     % 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
n=length(y);
randn('seed',123); rand('seed',123);

vague = 1000;
burnin=100;
ndraw=5100;

%Parameters for log-chi-squared errors%
r_p=.086;
r_m1=-7.472;
r_m2=-0.698;
r_sig2=1.411;
r_sig=sqrt(r_sig2);

%Parameters for RW Innovation Variance%
tau1=.20;  
tau2=.20;  
q_p=1.0;     %Prob of Regime 1...note that here we are always in regime 1, but it could be otherwise.
q1=tau1^2;
q2=tau1^2;

params=[r_p;r_m1;r_m2; r_sig; r_sig2;q_p;q1;q2;tau1;tau2;vague];


%Parameters for Initial Conditions, bounds and so forth%
    tau0=mean(y(1:4));  %average of first few observations
    dy=y(2:end)-y(1:end-1);
    var_dy = (std(dy))^2;
%Lower Bounds on variances			%
% these are needed to keep algorithm away from boundary   %
%    -- the values are problem specific: these seem to work well for the US	%
  %  var_eta_min = 0.015*var_dy;
    %var_eps_min = 0.005*var_dy;


%Initial Values				%
    var_eps_initial = var_dy/3;
    var_eta_initial = var_dy/3;

sprintf('Estimate of tau0 %10.4f',tau0);
sprintf('Lower bound on SD eta %10.4f',sqrt(var_eta_min));
sprintf('Lower bound on SD eps %10.4f',sqrt(var_eps_min));
sprintf('Initial guess of sd_eta %10.4f',sqrt(var_eta_initial));
sprintf('Initial guess of sd_eps %10.4f',sqrt(var_eps_initial));

   y=y-tau0; % Eliminate initial value of tau from analysis 
   r_pt_eps = r_p*ones(n,1);
   q_pt_eps = q_p*ones(n,1);
    r_pt_eta = r_p*ones(n,1);
    q_pt_eta = q_p*ones(n,1);
    var_eps_n = var_eps_initial*ones(n,1);
    var_eta_n = var_eta_initial*ones(n,1);
    
    sd_eps_save=zeros(n,ndraw-burnin);
    sd_eta_save=zeros(n,ndraw-burnin);
    tau_save=zeros(n,ndraw-burnin);
for idraw=1:ndraw;
[eps,eta,tau]=draw_eps_eta(y,var_eps_n,var_eta_n);
[var_eps_n,r_pt_eps,q_pt_eps]=draw_var(eps,r_pt_eps,q_pt_eps,var_eps_min,params);
[var_eta_n,r_pt_eta,q_pt_eta]=draw_var(eta,r_pt_eta,q_pt_eta,var_eta_min,params);  
     if idraw > burnin;
      sd_eps_n=sqrt(var_eps_n);
      sd_eta_n=sqrt(var_eta_n);
      sd_eps_save(:,idraw-burnin)=sd_eps_n;
      sd_eta_save(:,idraw-burnin)=sd_eta_n;     
      tau_save(:,idraw-burnin)=tau;
     end;
end;

sd_eps=zeros(n,1); sd_eta=zeros(n,1); tau=zeros(n,1);
for i=1:n;
sd_eps(i)=median(sd_eps_save(i,:));  %Could change these from medians into means or other percentiles
sd_eta(i)=median(sd_eta_save(i,:));
tau(i)=median(tau_save(i,:));
end;

tau=tau+tau0;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [eps,eta,tau]=draw_eps_eta(y,var_eps_n,var_eta_n);
n=length(y);
ltone=tril(ones(n,n));
cov_eps=diag(var_eps_n);
cov_tau=ltone*cov_eps*ltone'; 
diag_y=diag(cov_tau)+var_eta_n;
cov_y=cov_tau; for i=1:n; cov_y(i,i)=diag_y(i); end;
kappa=cov_tau*inv(cov_y);
mutau_y=kappa*y;
covtau_y=cov_tau-kappa*cov_tau';
chol_covtau_y=chol(covtau_y);
tau=mutau_y+chol_covtau_y'*randn(n,1);
eta=y-tau;
eps=[tau(1);tau(2:n)-tau(1:n-1)];


function [vardraw,r_pt,q_pt]=draw_var(x,r_pt,q_pt,var_min,params);
r_p=params(1); r_m1=params(2); r_m2=params(3); r_sig=params(4); r_sig2=params(5); q_p=params(6); q1=params(7); q2=params(8); tau1=params(9); tau2=params(10); vague=params(11);
n=size(x,1);
bsum=tril(ones(n+1,n+1));
lnres2=log(x.^2);

%Step 1 -- initial draws of Indicator Variables %
 tmp=rand(n,1);
 ir = tmp<r_pt;
 tmp=rand(n,1);
 iq = tmp<q_pt;

% Step 2; compute system parameters given indicators %
 mut = (ir*r_m1) + ((1-ir)*r_m2);
 qt = (iq*q1) + ((1-iq)*q2);

% Compute Covariance Matrix  % 
 vd=diag([vague;qt]);  
 valpha=bsum*vd*bsum';
 vy=valpha(2:n+1,2:n+1);
 cy=valpha(1:n+1,2:n+1);
 diagvy=diag(vy)+r_sig2;
 for i=1:n; vy(i,i)=diagvy(i); end;
 vyi=inv(vy);
 kgain=cy*vyi;

%Compute draws of state and shocks %
 ye=lnres2-mut;
 ahat0=kgain*ye;
 ahat1=ahat0(2:n+1);
 vhat0=valpha-kgain*cy';
 cvhat0=chol(vhat0);
 adraw0=ahat0+cvhat0'*randn(n+1,1);
 adraw1=adraw0(2:n+1);
 edraw=lnres2-adraw1;
 udraw=adraw0(2:n+1)-adraw0(1:n);

% Compute Mixture Probabilities %
  f1=exp(   (-0.5)* (((edraw-r_m1)./r_sig).^2)  );
  f2=exp(   (-0.5)* (((edraw-r_m2)./r_sig).^2)  );
  fe= r_p*f1 + (1-r_p)*f2;
  r_pt=(r_p*f1)./fe;

% u shocks -- Means are both zero%
  f1=(1/tau1)*exp(   (-0.5)* ((udraw./tau1).^2)  );
  f2=(1/tau1)*exp(   (-0.5)* ((udraw./tau2).^2)  );
  fu= q_p*f1 + (1-q_p)*f2;
  q_pt=(q_p*f1)./fu;

vardraw = exp(adraw1); 
vardraw=max(vardraw,var_min); % Impose minimum value
