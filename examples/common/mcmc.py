import stan
import time
import hashlib
import logging

def sample(sampler_data, N_samples, model_name, model_code, seed, 
        chains=1, control={'adapt_delta':0.9, 'max_treedepth':15}, verbose = False):
    
    # suppress the large amount of stan output and newlines
    logging.getLogger('httpstan').setLevel('WARNING')
    logging.getLogger('aiohttp').setLevel('WARNING')
    logging.getLogger('asyncio').setLevel('WARNING')

    # pystan 3 caches models itself, so no need to do that any more
    if verbose: print('STAN: building/loading model ' + model_name)
    sm = stan.build(model_code, data=sampler_data, random_seed=seed)

    if verbose: print('STAN: generating ' + str(N_samples) + ' samples from ' + model_name)
    t0 = time.perf_counter()
    #call sampling with N_samples actual iterations, and some number of burn iterations
    fit = sm.sample(num_samples=N_samples, num_chains=chains, delta=0.9, max_depth=15)

    t_sample = time.perf_counter() - t0
    t_per_iter = t_sample/(2.*N_samples) #denominator *2 since stan doubles the number of samples for tuning
    if verbose: print('STAN: total time: ' + str(t_sample) + ' seconds')
    if verbose: print('STAN: time per iteration: ' + str(t_per_iter) + ' seconds')
    return fit, t_sample, t_per_iter
