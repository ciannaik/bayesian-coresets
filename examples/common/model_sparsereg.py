import numpy as np
import scipy.linalg as sl
from scipy.special import gammaln

def log_likelihood(z, prm, sig):
    z = np.atleast_2d(z)
    x = z[:, :-1]
    y = z[:, -1]
    th = np.atleast_2d(prm)[:, :x.shape[1]]
    XST = x.dot(th.T)
    return -1./2.*np.log(2.*np.pi*sig**2) - 1./(2.*sig**2)*(y[:,np.newaxis]**2 - 2*XST*y[:,np.newaxis] + XST**2)

def grad_log_likelihood(z, prm, sig):
    z = np.atleast_2d(z)
    x = z[:, :-1]
    y = z[:, -1]
    th = np.atleast_2d(prm)[:, :x.shape[1]]
    grad = np.zeros((z.shape[0], prm.shape[0], prm.shape[1]))
    grad[:, :, :x.shape[1]] = 1./sig**2*(y[:, np.newaxis] - x.dot(th.T))[:,:,np.newaxis]* x[:, np.newaxis, :]
    return grad

def log_prior(prm, sig0, a0, b0):
    prm = np.atleast_2d(prm)
    d = (prm.shape[1]-2)/2
    th = prm[:, :d]
    lmb = prm[:, d:-2]
    sig = prm[:, -2]
    tau = prm[:, -1]
    # th[i] ~ normal(mu=0, sig=lmb*tau)
    logp_th = -1./2.*np.log(2.*np.pi*lmb**2*tau[:,np.newaxis]**2).sum(axis=1) - 1./2.*(th**2/lmb**2/tau[:,np.newaxis]**2).sum(axis=1)
    # lmb[i] ~ half-cauchy(0, 1)
    logp_lmb = (np.log(2./np.pi) - np.log(lmb**2 + 1.)).sum(axis=1)
    # sig ~ gamma(a0, b0)
    logp_sig = a0*np.log(b0) - gammaln(a0) + (a0-1.)*np.log(sig) - b0*sig
    # tau ~ half-cauchy(0, sig0)
    logp_tau = np.log(2.*sig0/np.pi) - np.log(tau**2 + sig0**2)
    return logp_th + logp_lmb + logp_sig + logp_tau

def grad_log_prior(prm, sig0):
    return -(th - mu0)/sig0**2

def log_joint(z, th, w, sig, sig0):
    return (w[:,np.newaxis]*log_likelihood(z, prm, sig)).sum(axis=0) + log_prior(prm, sig0)

def grad_log_joint(z, th, w, sig, sig0):
    return (w[:,np.newaxis,np.newaxis]*grad_log_likelihood(z, th, sig)).sum(axis=0) + grad_log_prior(th, mu0, sig0)

def hess_log_joint(z, th, w, sig, mu0, sig0):
    z = np.atleast_2d(z)
    th = np.atleast_2d(th)
    x = z[:, :-1]
    y = z[:, -1]
    return -(w*x.T).dot(x)/sig**2 - np.eye(x.shape[1])/sig0**2

stan_code = """
data {
  int<lower=0> n; // number of observations
  int<lower=0> d; // number of predictors
  vector[n] y; // outputs
  matrix[n,d] x; // inputs
  vector<lower=0>[n] w;  // weights
  real sig0; // global scale hyperparam scale
  real a0; // measurement noise gamma hyperparameter shape
  real b0; // measurement noise gamma hyperparameter scale
}
parameters {
  vector[d] theta; // auxiliary parameter
  real<lower=0> tau;
  vector<lower=0>[d] lambda;
}
model {
  sig ~ gamma(a0, b0);
  tau ~ cauchy(0, sig0);
  for(i in 1:d){
    lambda[i] ~ cauchy(0, 1);
    theta[i] ~ normal(0, tau*lambda[i]);
  }
  for(i in 1:n){
    target +=  normal_lpdf(y[i] | x[i]*theta, sig) * w[i];
  }
}
"""
