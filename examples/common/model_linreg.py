import numpy as np
import scipy.linalg as sl

def log_likelihood(z, th, sig):
    z = np.atleast_2d(z)
    th = np.atleast_2d(th)
    x = z[:, :-1]
    y = z[:, -1]
    XST = x.dot(th.T)
    return -1./2.*np.log(2.*np.pi*sig**2) - 1./(2.*sig**2)*(y[:,np.newaxis]**2 - 2*XST*y[:,np.newaxis] + XST**2)

def grad_log_likelihood(z, th, sig):
    z = np.atleast_2d(z)
    th = np.atleast_2d(th)
    x = z[:, :-1]
    y = z[:, -1]
    return 1./sig**2*(y[:, np.newaxis] - x.dot(th.T))[:,:,np.newaxis]* x[:, np.newaxis, :]

def log_prior(th, mu0, sig0):
    return -th.shape[1]/2.*np.log(2.*np.pi*sig0**2) - 1./2.*((th - mu0)**2).sum(axis=1)/sig0**2

def grad_log_prior(th, mu0, sig0):
    return -(th - mu0)/sig0**2

def log_joint(z, th, w, sig, mu0, sig0):
    return (w[:,np.newaxis]*log_likelihood(z, th, sig)).sum(axis=0) + log_prior(th, mu0, sig0)

def grad_log_joint(z, th, w, sig, mu0, sig0):
    return (w[:,np.newaxis]*grad_log_likelihood(z, th, sig)).sum(axis=0) + grad_log_prior(th, mu0, sig0)

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
  real mu0;
  real<lower=0> sig0;
  real<lower=0> sig;
}
parameters {
  vector[d] theta; // auxiliary parameter
}
model {
  for(i in 1:d){
      theta[i] ~ normal(mu0, sig0);
  }
  for(i in 1:n){
    target += normal_lpdf(y[i] | x[i]*theta, sig) * w[i];
  }
}
"""

