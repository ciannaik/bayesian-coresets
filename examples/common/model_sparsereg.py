import numpy as np
import scipy.linalg as sl
from scipy.special import gammaln

def gen_synthetic(n, d, d_active, sig):
  th = np.random.randn(d)
  idcs = np.arange(d)
  np.random.shuffle(idcs)
  th[idcs[d_active:]] = 0.
  X = np.random.randn(n, d)
  y = X.dot(th) + sig*np.random.randn(n)
  return X, y

def load_data(dnm):
  data = np.load(dnm)
  X = data['X']
  Y = data['y']
  return X, Y

def log_likelihood(z, prm):
    z = np.atleast_2d(z)
    x = z[:, :-1]
    y = z[:, -1]
    prm = np.atleast_2d(prm)
    d = x.shape[1]
    th = prm[:, :d]
    lmb = prm[:, d:-2]
    sig = prm[:, -2]
    tau = prm[:, -1]
    XST = x.dot(th.T)
    return -1./2.*np.log(2.*np.pi*sig**2) - 1./(2.*sig**2)*(y[:,np.newaxis]**2 - 2*XST*y[:,np.newaxis] + XST**2)

def grad_log_likelihood(z, prm):
    z = np.atleast_2d(z)
    x = z[:, :-1]
    y = z[:, -1]
    prm = np.atleast_2d(prm)
    d = x.shape[1]
    th = prm[:, :d]
    lmb = prm[:, d:-2]
    sig = prm[:, -2]
    tau = prm[:, -1]
    XST = x.dot(th.T)
    grad = np.zeros((z.shape[0], prm.shape[0], prm.shape[1]))
    grad[:, :, :th.shape[1]] = (1./sig[np.newaxis, :]**2*(y[:, np.newaxis] - XST))[:,:,np.newaxis]* x[:, np.newaxis, :]
    grad[:, :, -2] = -1./sig[np.newaxis, :] +  1./(sig**3)*(y[:,np.newaxis]**2 - 2*XST*y[:,np.newaxis] + XST**2)
    return grad

def log_prior(prm, sig0, a0, b0):
    prm = np.atleast_2d(prm)
    d = int((prm.shape[1]-2)/2)
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

def grad_log_prior(prm, sig0, a0, b0):
    prm = np.atleast_2d(prm)
    d = int((prm.shape[1]-2)/2)
    th = prm[:, :d]
    lmb = prm[:, d:-2]
    sig = prm[:, -2]
    tau = prm[:, -1]
    d_logp_th = - th/lmb**2/tau[:,np.newaxis]**2
    d_logp_lmb = -1./(lmb**2+1.)*2*lmb - 1./lmb + th**2/lmb**3/tau[:,np.newaxis]**2
    d_logp_tau = -1./(tau**2+sig0**2)*2*tau  -th.shape[1]/tau + (th**2/lmb**2/tau[:,np.newaxis]**3).sum(axis=1)
    d_logp_sig = (a0-1.)/sig - b0
    grad = np.zeros((prm.shape[0], prm.shape[1]))
    grad[:, :d] = d_logp_th
    grad[:, d:-2] = d_logp_lmb
    grad[:, -2] = d_logp_sig
    grad[:, -1] = d_logp_tau
    return grad

def log_joint(z, prm, w, sig0, a0, b0):
    return (w[:,np.newaxis]*log_likelihood(z, prm)).sum(axis=0) + log_prior(prm, sig0, a0, b0)

def grad_log_joint(z, prm, w, sig0, a0, b0):
    return (w[:,np.newaxis,np.newaxis]*grad_log_likelihood(z, prm)).sum(axis=0) + grad_log_prior(prm, sig0, a0, b0)

def hess_log_joint(z, prm, w, sig0, a0, b0):
    eps = 1e-3
    prm = np.atleast_2d(prm)
    hess = np.zeros((prm.shape[0], prm.shape[1], prm.shape[1]))
    for i in range(prm.shape[1]):
        prmr = prm.copy()
        prmr[:, i] += eps/2.
        gradr = grad_log_joint(z, prmr, w, sig0, a0, b0)
        prml = prm.copy()
        prml[:, i] -= eps/2.
        gradl = grad_log_joint(z, prml, w, sig0, a0, b0)
        hess[:, :, i] = (gradr - gradl)/eps
    return hess

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
  real<lower=0> sig;
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
