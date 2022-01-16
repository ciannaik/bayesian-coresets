import numpy as np
import stan
import os
import pickle as pk
import time
import hashlib
import logging

#Takes in the name of and code for a statistical model that allows stan to run MCMC, and returns cpp code that stan can use to perform MCMC on a coreset.  
# def load_modified_cpp_code(stan_folder, model_name, model_code):
#   code_hash = hashlib.sha1(model_code.encode('utf-8')).hexdigest()
#   cpp_filename = os.path.join(stan_folder, "weighted_" + model_name + "_" + code_hash + ".cpp")
#   cpp_filename = os.path.join(stan_folder, "poiss_weighted_coreset_version_2eb5fdf9460ca0b941e7240dfac57ef030aceee3.cpp")
#   if os.path.exists(cpp_filename):
#     f = open(cpp_filename,'r')
#     modified_code = f.read()
#     f.close()
#     return modified_code
#   else:
#     if not os.path.exists(stan_folder):
#       os.mkdir(stan_folder)
#
#     sm = pystan.StanModel(model_code=model_code)
#     f = open(cpp_filename, "w")
#     f.write("This line will create an error. Remove it once you have modified the code below to handle weighted coresets. See ReadMe for more information.\n")
#     f.write(sm.model_cppcode)
#     f.close()
#     unweighted_cpp_filename = os.path.join(stan_folder, 'unweighted_' + model_name + "_" + code_hash + '.cpp')
#     f = open(unweighted_cpp_filename, "w")
#     f.write(sm.model_cppcode)
#     f.close()
#     raise EnvironmentError("No modified code to handle weighted data present - unable to use stan for MCMC sampling. Please modify the file "+str(cpp_filename)+" to handle weighted data. See the ReadMe for more information.")


def build_model(stan_folder, model_name, model_code, sampler_data, seed):
  # suppress the large amount of stan output and newlines
  logging.getLogger('httpstan').setLevel('WARNING')
  logging.getLogger('aiohttp').setLevel('WARNING')
  logging.getLogger('asyncio').setLevel('WARNING')

  code_hash = hashlib.sha1(model_code.encode('utf-8')).hexdigest()
  model_filename = os.path.join(stan_folder, model_name + "_" + code_hash)
  if os.path.exists(model_filename):
    print('STAN: cached model found; loading')
    f = open(model_filename, 'rb')
    sm = pk.load(f)
    f.close()
  else: 
    if not os.path.exists(stan_folder):
      os.mkdir(stan_folder)
    print('STAN: no cached model found; building')
    try:
        sm = stan.build(model_code, data=sampler_data, random_seed=seed)
    except:
        print('Compiling failed. Did you make sure to modify the weighted_[model_name]_[code_hash].cpp file to handle weighted data?')
        raise
    
    # f = open(model_filename,'wb')
    # pk.dump(sm,f)
    # f.close()
  return sm

def run(sampler_data, N_samples, model_name, model_code, seed, stan_folder = '../common/stan_cache/', chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose=True, init = None):
    # suppress the large amount of stan output and newlines
    logging.getLogger('httpstan').setLevel('WARNING')
    logging.getLogger('aiohttp').setLevel('WARNING')
    logging.getLogger('asyncio').setLevel('WARNING')

    print('STAN: building/loading model ' + model_name)
    sm = build_model(stan_folder, model_name, model_code, sampler_data, seed)

    print('STAN: generating ' + str(N_samples) + ' samples from ' + model_name)
    t0 = time.perf_counter()
    #call sampling with N_samples actual iterations, and some number of burn iterations
    if init is None:
        fit = sm.sample(num_samples=N_samples, num_chains=chains, delta=0.9, max_depth=15)
    else:
        fit = sm.sample(num_samples=N_samples, num_chains=chains, init=init, delta=0.9, max_depth=15)

    t_sample = time.perf_counter() - t0
    t_per_iter = t_sample/(2.*N_samples) #denominator *2 since stan doubles the number of samples for tuning
    print('STAN: total time: ' + str(t_sample) + ' seconds')
    print('STAN: time per iteration: ' + str(t_per_iter) + ' seconds')
    return fit, t_sample, t_per_iter


# TODO test weighted CPP using randomly chosen integer weights
# make sure each step is equivalent
# use permuted=False in extract to avoid stan reordering samples
#arguments = parser.parse_args()
#dnm = arguments.dnm
#ID = arguments.ID
#model = arguments.model
#numIdcs = arguments.numIdcs
#mcmc_samples = arguments.mcmc_samples
#verbose_compiling = arguments.verbose_compiling
#
##lrdnms = ['synth_lr', 'phishing', 'ds1', 'synth_lr_large', 'phishing_large', 'ds1_large']
##prdnms = ['synth_poiss', 'biketrips', 'airportdelays', 'synth_poiss_large', 'biketrips_large', 'airportdelays_large']
#if model=="lr":
#  from model_lr import *
#elif model=="poiss":
#  from model_poiss import *
#
#np.random.seed(ID)
#
#print('Loading dataset '+dnm)
#X,Y,Z, Zt, D = load_data('../data/'+dnm+'.npz')
#print("N, d: "+str(X.shape))
#
#idcs = np.random.choice(X.shape[0],numIdcs, replace=True) #random.choice is a bit slower without replacement, but may still be worth trying
#weights = np.random.randint(1,10,numIdcs)
#curX = X[idcs, :]
#curY = Y[idcs]
##run the weighted version of sampler
## (we want to make sure we use a model based on the current cpp code for weighted coresets, so we don't use any cached models)
#samples_using_code_for_weights = mcmc.sampler(dnm, curX, curY, mcmc_samples, stan_representation, weights=weights, model_caching_folder = None, verbose_compiling = verbose_compiling, seed = ID) 
## build the modified dataset so that we can run the unweighted version of sampler
#for i in range(len(idcs)):
#    toAdd = np.ones(weights[i]-1,dtype=int) * (i) #list i a number of times equal to the weight of the ith index
#    curX = np.append(curX, curX[toAdd], axis=0)
#    curY = np.append(curY, curY[toAdd])
#samples_using_standard_stan_code = mcmc.sampler(dnm, curX, curY, mcmc_samples, stan_representation, seed = ID)
#
#relative_err = np.abs(np.linalg.norm(samples_using_code_for_weights - samples_using_standard_stan_code, axis=1)/np.linalg.norm(samples_using_standard_stan_code,axis=1))
#print("largest relative error between samples using code for weights and samples using standard stan code (should be 0): ")
#print(np.max(relative_err))
#print("average relative error")
#print(np.mean(relative_err))
#print("median relative error")
#print(np.quantile(relative_err,.5))
#print("first quartile relative error")
#print(np.quantile(relative_err,.25))
#print("third quartile relative error")
#print(np.quantile(relative_err,.75))

