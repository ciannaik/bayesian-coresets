import numpy as np

def load_data(dnm):
  data = np.load(dnm)
  X = data['X']
  Y = data['y']
  #standardize the covariates; last col is intercept, so no stdization there
  m = X[:, :-1].mean(axis=0)
  V = np.cov(X[:, :-1], rowvar=False)+1e-12*np.eye(X.shape[1]-1)
  X[:, :-1] = np.linalg.solve(np.linalg.cholesky(V), (X[:, :-1] - m).T).T
  Z = data['y'][:, np.newaxis]*X
  data.close()
  return X, Y, Z, None, Z.shape[1]

def gen_synthetic(n):
  mu = np.array([0, 0, 0])
  cov = np.eye(3)
  th = np.array([3, 3, 3, 3, 3, 3, 0])
  X = np.zeros((n,7))
  X[:,-1] = 1
  X[:,3:-1] = np.random.multivariate_normal(mu, cov, n)
  ps = 1.0/(1.0+np.exp(-(X*th).sum(axis=1)))
  y = (np.random.rand(n) <= ps).astype(int)
  y[y==0] = -1
  Z = y[:, np.newaxis]*X
  return X, y, Z, None, Z.shape[1], X[:,3:], X[:,:3], Z[:,3:], Z[:,:3], Z[:,3:].shape[1], Z[:,:3].shape[1]

def log_likelihood(z, th):
  z = np.atleast_2d(z)
  th = np.atleast_2d(th)
  m = -z.dot(th.T)
  idcs = m < 100
  m[idcs] = -np.log1p(np.exp(m[idcs]))
  m[np.logical_not(idcs)] = -m[np.logical_not(idcs)]
  return m

def log_prior(th):
  th = np.atleast_2d(th)
  sigma = np.repeat(2, th.shape[1])
  sigma = np.atleast_2d(sigma)
  ll = -np.log(np.pi*(sigma + th**2/sigma))
  return ll.sum(axis=1)

def log_joint(z, th, wts):
    return (wts[:, np.newaxis]*log_likelihood(z, th)).sum(axis=0) + log_prior(th)

def grad_th_log_likelihood(z, th):
  z = np.atleast_2d(z)
  th = np.atleast_2d(th)
  m = -z.dot(th.T)
  idcs = m < 100
  m[idcs] = np.exp(m[idcs])/(1.+np.exp(m[idcs]))
  m[np.logical_not(idcs)] = 1.
  return m[:, :, np.newaxis]*z[:, np.newaxis, :]

def grad_z_log_likelihood(z, th):
  z = np.atleast_2d(z)
  th = np.atleast_2d(th)
  m = -z.dot(th.T)
  idcs = m < 100
  m[idcs] = np.exp(m[idcs])/(1.+np.exp(m[idcs]))
  m[np.logical_not(idcs)] = 1.
  return m[:, :, np.newaxis]*th[np.newaxis, :, :]

def grad_th_log_prior(th):
  th = np.atleast_2d(th)
  sigma = np.repeat(2, th.shape[1])
  sigma = np.atleast_2d(sigma)
  return -2*th/(sigma**2 + th**2)

def grad_th_log_joint(z, th, wts):
  return grad_th_log_prior(th) + (wts[:, np.newaxis, np.newaxis]*grad_th_log_likelihood(z, th)).sum(axis=0)

def hess_th_log_likelihood(z, th):
  z = np.atleast_2d(z)
  th = np.atleast_2d(th)
  m = -z.dot(th.T)
  idcs = m < 100
  m[idcs] = np.exp(m[idcs])/(1.+np.exp(m[idcs]))**2
  m[np.logical_not(idcs)] = 0.
  return -m[:, :, np.newaxis, np.newaxis]*z[:, np.newaxis, :, np.newaxis]*z[:, np.newaxis, np.newaxis, :]

def hess_th_log_prior(th):
  th = np.atleast_2d(th)
  sigma = np.repeat(2, th.shape[1])
  sigma = np.atleast_2d(sigma)
  h = -2 * (sigma ** 2 - th ** 2) / ((sigma ** 2 + th ** 2) ** 2)
  h = np.diag(h[0,:])
  return np.tile(h, (th.shape[0], 1, 1))

def hess_th_log_joint(z, th, wts):
  return hess_th_log_prior(th) + (wts[:, np.newaxis, np.newaxis, np.newaxis]*hess_th_log_likelihood(z, th)).sum(axis=0)

def diag_hess_th_log_likelihood(z, th):
  z = np.atleast_2d(z)
  th = np.atleast_2d(th)
  m = -z.dot(th.T)
  idcs = m < 100
  m[idcs] = np.exp(m[idcs])/(1.+np.exp(m[idcs]))**2
  m[np.logical_not(idcs)] = 0.
  return -m[:, :, np.newaxis]*z[:, np.newaxis, :]**2

def diag_hess_th_log_prior(th):
  th = np.atleast_2d(th)
  sigma = np.repeat(2, th.shape[1])
  sigma = np.atleast_2d(sigma)
  h = -2 * (sigma ** 2 - th ** 2) / (sigma ** 2 + th ** 2) ** 2
  dh = h[0, :]
  return np.tile(dh, (th.shape[0], 1))

def diag_hess_th_log_joint(z, th, wts):
  return diag_hess_th_log_prior(th) + (wts[:, np.newaxis, np.newaxis]*diag_hess_th_log_likelihood(z, th)).sum(axis=0)

# stan_representation = """
# data {
#   int<lower=0> n; // number of observations
#   int<lower=0> d; // number of predictors
#   int<lower=0,upper=1> y[n]; // outputs
#   matrix[n,d] x; // inputs
# }
# parameters {
#   vector[d] theta; // auxiliary parameter
# }
# transformed parameters {
#   vector[n] f;
#   f = x*theta;
# }
# model {
#   theta ~ normal(0, 1);
#   y ~ bernoulli_logit(f);
# }
# """


stan_representation = """
data {
  int<lower=0> n; // number of observations
  int<lower=0> d; // number of predictors
  int<lower=0,upper=1> y[n]; // outputs
  matrix[n,d] x; // inputs
  vector<lower=0>[n] w;  // weights
}
parameters {
  vector[d] theta; // auxiliary parameter
}
transformed parameters {
  vector[n] f;
  f = x*theta;
}
model {
  for(i in 1:d){
    theta[i] ~ cauchy(0, 2);
  }
  for(i in 1:n){
    target +=  bernoulli_logit_lupmf(y[i] | f[i]) * w[i];
  }
}
"""

# stan_representation = """
# data {
#   int<lower=0> n; // number of observations
#   int<lower=0> d; // number of predictors
#   int<lower=0,upper=1> y[n]; // outputs
#   matrix[n,d] x; // inputs
#   vector<lower=0>[n] w;  // weights
# }
# parameters {
#   vector[d] theta; // auxiliary parameter
# }
# transformed parameters {
#   vector[n] f;
#   f = x*theta;
# }
# model {
#   theta ~ normal(0, 2);
#   for(i in 1:n){
#     target +=  bernoulli_logit_lupmf(y[i] | f[i]) * w[i];
#   }
# }
# """
