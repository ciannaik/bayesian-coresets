import numpy as np
import scipy.linalg as sl

def gen_synthetic(n, d, sig):
  return 1. + sig*np.random.randn(n, d)

def log_likelihood(x, th, sig):
  x = np.atleast_2d(x)
  th = np.atleast_2d(th)
  xx = (x**2).sum(axis=1)/sig**2
  thth = (th**2).sum(axis=1)/sig**2
  xth = x.dot(th.T)
  return -x.shape[1]/2.*np.log(2*np.pi*sig**2) - 1./2.*(xx[:, np.newaxis] + thth - 2*xth)

def log_prior(th, mu0, sig0):
  th = np.atleast_2d(th)
  return -th.shape[1]/2.*np.log(2*np.pi*sig0**2) - 1./2.*((th - mu0)**2).sum(axis=1)/sig0**2

def log_joint(x, th, wts, sig, mu0, sig0):
    return (wts[:, np.newaxis]*log_likelihood(x, th, sig)).sum(axis=0) + log_prior(th, mu0, sig0)

def grad_log_likelihood(x, th, sig):
  x = np.atleast_2d(x)
  th = np.atleast_2d(th)
  return -(th[np.newaxis, :, :] - x[:, np.newaxis, :])/sig**2

def grad_log_prior(th, mu0, sig0):
  th = np.atleast_2d(th)
  return -(th - mu0)/sig0**2

def grad_log_joint(x, th, wts, sig, mu0, sig0):
  return (wts[:, np.newaxis]*grad_log_likelihood(x, th, sig)).sum(axis=0) + grad_log_prior(th, mu0, sig0)

def hess_log_likelihood(x, th, sig):
  return -np.ones((x.shape[0], th.shape[0], th.shape[1]))/sig**2

def hess_log_prior(th, mu0, sig0):
  return -np.ones((th.shape[0], th.shape[1]))/sig0**2

def hess_log_joint(x, th, wts, sig, mu0, sig0):
  return (wts[:, np.newaxis]*hess_log_likelihood(x, th, sig)).sum(axis=0) + hess_log_prior(th, mu0, sig0)

def weighted_post(mu0, sig0, sig, x, w):
  sigp = np.sqrt(1./(1./sig0**2 + w.sum()/sig**2))
  if w.shape[0] > 0:
    mup = sigp**2*(mu0/sig0**2 + (w[:,np.newaxis]*x).sum(axis=0)/sig**2)
  else:
    mup = mu0*np.ones(x.shape[1])
  return mup, sigp

def KL(mu0, Sig0, mu1, Sig1inv):
  t1 = np.dot(Sig1inv, Sig0).trace()
  t2 = np.dot((mu1-mu0),np.dot(Sig1inv, mu1-mu0))
  t3 = -np.linalg.slogdet(Sig1inv)[1] - np.linalg.slogdet(Sig0)[1]
  return 0.5*(t1+t2+t3-mu0.shape[0])


