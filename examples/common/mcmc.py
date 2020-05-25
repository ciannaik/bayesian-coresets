import numpy as np
import pystan
from model_lr import logistic_code
from model_poiss import poisson_code
import os
import pickle as pk
import time

def load_data(dnm):
  data = np.load(dnm)
  X = data['X']
  Y = data['y']
  Xt = data['Xt']
  #standardize the covariates; last col is intercept, so no stdization there
  m = X[:, :-1].mean(axis=0)
  V = np.cov(X[:, :-1], rowvar=False)+1e-12*np.eye(X.shape[1]-1)
  X[:, :-1] = np.linalg.solve(np.linalg.cholesky(V), (X[:, :-1] - m).T).T
  Xt[:, :-1] = np.linalg.solve(np.linalg.cholesky(V), (Xt[:, :-1] - m).T).T
  data.close()
  return X[:, :-1], Y

def build_model(resfldr, modelName, modelCode):
  if not os.path.exists(os.path.join(resfldr, modelName)): 
      print('STAN: building model')
      sm = pystan.StanModel(model_code=modelCode)
      f = open(os.path.join(resfldr, modelName),'wb')
      pk.dump(sm, f)
      f.close()
  else:
      f = open(os.path.join(resfldr, modelName),'rb')
      sm = pk.load(f)
      f.close()
  return sm

def sampler(dnm, lr, datafldr, resfldr, N_samples):
  print('STAN: loading data')
  X, Y = load_data(os.path.join(datafldr,dnm+'.npz'))
  Y[Y == -1] = 0 #convert to Stan LR label style if necessary

  sampler_data = {'x': X, 'y': Y.astype(int), 'd': X.shape[1], 'n': X.shape[0]}

  print('STAN: building/loading model')
  if lr:
    sm = build_model("caching", 'pystan_model_logistic.pk', logistic_code)
  else:
    sm = build_model("caching", 'pystan_model_poisson.pk', poisson_code)

  print('STAN: sampling posterior: ' + dnm)
  t0 = time.process_time()
  thd = sampler_data['d']+1
  fit = sm.sampling(data=sampler_data, iter=N_samples*2, chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=True)
  samples = fit.extract(permuted=False)[:, 0, :thd]
  np.save(os.path.join(resfldr, dnm+'_samples.npy'), samples) 
  tf = time.process_time()
  np.save(os.path.join(resfldr, dnm+'_mcmc_time.npy'), tf-t0)
