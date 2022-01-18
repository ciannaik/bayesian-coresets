from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
from scipy.stats import cauchy
import time
import sys, os
import argparse
import cProfile, pstats, io
from pstats import SortKey
# make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import mcmc
import laplace
import results
import plotting
import stein
from model_gaussian import KL


def plot(arguments):
    # load the dataset of results that matches these input arguments
    df = results.load_matching(arguments, match = ['model', 'dataset', 'samples_inference', 'proj_dim', 'opt_itrs', 'step_sched'])
    # call the generic plot function
    plotting.plot(arguments, df)


def run(arguments):
    # suffix for printouts
    log_suffix = '(coreset size: ' + str(arguments.coreset_size) + ', numdata: ' + str(arguments.data_num) + ', nbases: ' + str(arguments.n_bases_per_scale) + ', alg: ' + arguments.alg + ', trial: ' + str(arguments.trial)+')'

    #######################################
    #######################################
    ############# Setup ###################
    #######################################
    #######################################

    # check if result already exists for this run, and if so, quit
    if results.check_exists(arguments):
        print('Results already exist for arguments ' + str(arguments))
        print('Quitting.')
        quit()

    np.random.seed(arguments.trial)
    bc.util.set_verbosity(arguments.verbosity)

    #######################################
    #######################################
    ############# Load Dataset ############
    #######################################
    #######################################

    #load data and compute true posterior
    #each row of x is [lat, lon, price]
    print('Loading data')

    data = np.load('../data/prices2018.npy')
    print('dataset size : ', data.shape)

    print('Subsampling down to '+str(arguments.data_num) + ' points')
    idcs = np.arange(data.shape[0])
    np.random.shuffle(idcs)
    data = data[idcs[:arguments.data_num], :]

    #log transform the prices
    data[:, 2] = np.log10(data[:, 2])

    #get empirical mean/std
    datastd = data[:,2].std()
    datamn = data[:,2].mean()

    #bases of increasing size; the last one is effectively a constant
    basis_unique_scales = np.array([.2, .4, .8, 1.2, 1.6, 2., 100])
    basis_unique_counts = np.hstack((arguments.n_bases_per_scale*np.ones(6, dtype=np.int64), 1))

    #the dimension of the scaling vector for the above bases
    d = basis_unique_counts.sum()
    print('Basis dimension: ' + str(d))

    #generate basis functions by uniformly randomly picking locations in the dataset
    print('Trial ' + str(arguments.trial))
    print('Creating bases')
    basis_scales = np.array([])
    basis_locs = np.zeros((0,2))
    for i in range(basis_unique_scales.shape[0]):
      basis_scales = np.hstack((basis_scales, basis_unique_scales[i]*np.ones(basis_unique_counts[i])))
      idcs = np.random.choice(np.arange(data.shape[0]), replace=False, size=basis_unique_counts[i])
      basis_locs = np.vstack((basis_locs, data[idcs, :2]))

    print('Converting bases and observations into X/Y matrices')
    #convert basis functions + observed data locations into a big X matrix
    X = np.zeros((data.shape[0], basis_scales.shape[0]))
    for i in range(basis_scales.shape[0]):
      X[:, i] = np.exp( -((data[:, :2] - basis_locs[i, :])**2).sum(axis=1) / (2*basis_scales[i]**2) )
    Y = data[:, 2]
    Z = np.hstack((X, Y[:,np.newaxis]))

    #######################################
    #######################################
    ############ Define Model #############
    #######################################
    #######################################

    # import the model specification
    import model_linreg as model

    #model params
    mu0 = datamn
    sig0 = np.sqrt(datastd**2+datamn**2)
    sig = datastd

    ####################################################################
    ####################################################################
    ############ Construct weighted posterior sampler ##################
    ####################################################################
    ####################################################################

    print('Creating weighted sampler ' + log_suffix)
    def sample_w(n, wts, pts, get_timing=False):
        t0 = time.perf_counter()
        if pts.shape[0] > 0:
            samples = model.weighted_post_sampler(n, pts, wts, sig, mu0, sig0)
        else:
            samples = model.weighted_post_sampler(n, np.zeros((1, Z.shape[1])), np.zeros(1), sig, mu0, sig0)
        t_tot = time.perf_counter()-t0
        if get_timing:
            return samples, t_tot, t_tot/n
        else:
            return samples

    #def sample_w(n, wts, pts, get_timing=False):
    #    # passing in X, Y into the stan sampler
    #    # is equivalent to passing in pts, [1, ..., 1] (since Z = X*Y and pts is a subset of Z)
    #    if pts.shape[0] > 0:
    #        sampler_data = {'x': pts[:, :-1], 'y': pts[:, -1], 'w': wts,
    #                        'd': X.shape[1], 'n': pts.shape[0], 'mu0': mu0, 'sig0': sig0, 'sig': sig}
    #    else:
    #        sampler_data = {'x': np.zeros((1, X.shape[1])), 'y': np.zeros(1), 'w': np.zeros(1),
    #                        'd': X.shape[1], 'n': 1, 'mu0': mu0, 'sig0': sig0, 'sig': sig}
    #    samples, t_mcmc, t_mcmc_per_itr = mcmc.sample(sampler_data, n, 'linear_regression',
    #                                                        model.stan_code, arguments.trial)
    #    if get_timing:
    #        return samples['theta'].T, t_mcmc, t_mcmc_per_itr
    #    else:
    #        return samples['theta'].T

    #######################################
    #######################################
    ###### Get samples on the full data ###
    #######################################
    #######################################

    print('Checking for cached full samples ' + log_suffix)
    cache_filename = 'full_cache/full_samples_' + str(arguments.data_num) + '_' + str(arguments.n_bases_per_scale) + '.npz'
    if os.path.exists(cache_filename):
        print('Cache exists, loading')
        tmp__ = np.load(cache_filename)
        full_samples = tmp__['samples']
        t_full_per_itr = float(tmp__['t'])
    else:
        print('Cache doesn\'t exist, running sampler')
        full_samples, t_full, t_full_per_itr = sample_w(arguments.samples_inference, np.ones(Z.shape[0]), Z, get_timing=True)
        if not os.path.exists('full_cache'):
            os.mkdir('full_cache')
        np.savez(cache_filename, samples=full_samples, t=t_full_per_itr, allow_pickle=True)

    #######################################
    #######################################
    ## Step 4: Construct Coreset
    #######################################
    #######################################

    print('Creating coreset construction objects ' + log_suffix)
    # create coreset construction objects
    projector = bc.BlackBoxProjector(sample_w, arguments.proj_dim, lambda x, th : model.log_likelihood(x, th, sig), None)
    unif = bc.UniformSamplingCoreset(Z)
    giga = bc.HilbertCoreset(Z, projector)
    sparsevi = bc.SparseVICoreset(Z, projector, opt_itrs=arguments.opt_itrs, step_sched=eval(arguments.step_sched))
    newton = bc.QuasiNewtonCoreset(Z, projector, opt_itrs=arguments.newton_opt_itrs)
    lapl = laplace.LaplaceApprox(lambda th : model.log_joint(Z, th, np.ones(Z.shape[0]), sig, mu0, sig0)[0],
				    lambda th : model.grad_log_joint(Z, th, np.ones(Z.shape[0]), sig, mu0, sig0)[0,:],
                                    np.zeros(X.shape[1]),
				    hess_log_joint = lambda th : model.hess_log_joint(Z, th, np.ones(Z.shape[0]), sig, mu0, sig0))

    algs = {'SVI' : sparsevi,
            'QNC' : newton,
            'LAP' : lapl,
            'GIGA': giga,
            'UNIF': unif}
    alg = algs[arguments.alg]


    print('Building ' + log_suffix)
    # Recursive alg needs to be run fully each time
    t0 = time.perf_counter()
    alg.build(arguments.coreset_size)
    t_build = time.perf_counter() - t0


    print('Sampling ' + log_suffix)

    __get = getattr(alg, "get", None)
    if callable(__get):
        wts, pts, idcs = alg.get()
        # Use MCMC on the coreset, measure time taken
        approx_samples, t_approx_sampling, t_approx_per_sample = sample_w(arguments.samples_inference, wts, pts, get_timing=True)
    else:
        approx_samples, t_approx_sampling, t_approx_per_sample = alg.sample(arguments.samples_inference, get_timing=True)


    print('Evaluation ' + log_suffix)
    # get full/approx posterior mean/covariance
    mu_approx = approx_samples.mean(axis=0)
    Sig_approx = np.cov(approx_samples, rowvar=False)
    LSig_approx = np.linalg.cholesky(Sig_approx)
    LSigInv_approx = solve_triangular(LSig_approx, np.eye(LSig_approx.shape[0]), lower=True, overwrite_b=True, check_finite=False)
    mu_full = full_samples.mean(axis=0)
    Sig_full = np.cov(full_samples, rowvar=False)
    LSig_full = np.linalg.cholesky(Sig_full)
    LSigInv_full = solve_triangular(LSig_full, np.eye(LSig_full.shape[0]), lower=True, overwrite_b=True, check_finite=False)
    # compute the relative 2 norm error for mean and covariance
    mu_err = np.sqrt(((mu_full - mu_approx) ** 2).sum()) / np.sqrt((mu_full ** 2).sum())
    Sig_err = np.linalg.norm(Sig_approx - Sig_full, ord=2)/np.linalg.norm(Sig_full, ord=2)
    # compute gaussian reverse/forward KL
    rklw = KL(mu_approx, Sig_approx, mu_full, LSigInv_full.T.dot(LSigInv_full))
    fklw = KL(mu_full, Sig_full, mu_approx, LSigInv_approx.T.dot(LSigInv_approx))
    # compute mmd discrepancies
    gauss_mmd = stein.gauss_mmd(approx_samples, full_samples)
    imq_mmd = stein.imq_mmd(approx_samples, full_samples)
    # compute stein discrepancies
    scores_approx = model.grad_log_joint(Z, approx_samples, np.ones(Z.shape[0]), sig, mu0, sig0)
    gauss_stein = stein.gaussian_stein_discrepancy(approx_samples, scores_approx)
    imq_stein = stein.imq_stein_discrepancy(approx_samples, scores_approx)


    print('Saving ' + log_suffix)
    results.save(arguments, t_build=t_build, t_per_sample=t_approx_per_sample, t_full_per_sample=t_full_per_itr,
                 rklw=rklw, fklw=fklw, mu_err=mu_err, Sig_err=Sig_err, gauss_mmd=gauss_mmd, imq_mmd=imq_mmd,
                 gauss_stein=gauss_stein, imq_stein=imq_stein)
    print('')
    print('')


############################
############################
## Parse arguments
############################
############################

parser = argparse.ArgumentParser(description="Runs Hilbert coreset construction on a model and dataset")
subparsers = parser.add_subparsers(help='sub-command help')
run_subparser = subparsers.add_parser('run', help='Runs the main computational code')
run_subparser.set_defaults(func=run)
plot_subparser = subparsers.add_parser('plot', help='Plots the results')
plot_subparser.set_defaults(func=plot)

parser.add_argument('--data_num', type=int, default='10000', help='Dataset subsample to use')
parser.add_argument('--n_bases_per_scale', type=int, default=50, help="The number of Radial Basis Functions per scale")#TODO: verify help message
parser.add_argument('--alg', type=str, default='GIGA',
                    choices=['SVI', 'QNC', 'GIGA', 'UNIF', 'LAP'],
                    help="The algorithm to use for solving sparse non-negative least squares")  # TODO: find way to make this help message autoupdate with new methods
parser.add_argument("--samples_inference", type=int, default=1000,
                    help="number of MCMC samples to take for actual inference and comparison of posterior approximations (also take this many warmup steps before sampling)")
parser.add_argument("--proj_dim", type=int, default=2000,
                    help="The number of samples taken when discretizing log likelihoods")
parser.add_argument('--coreset_size', type=int, default=100, help="The coreset size to evaluate")
parser.add_argument('--opt_itrs', type=str, default=100,
                    help="Number of optimization iterations (for SVI)")
parser.add_argument('--newton_opt_itrs', type=str, default=20,
                    help="Number of optimization iterations (for QNC)")
parser.add_argument('--step_sched', type=str, default="lambda i : 1./(i+1)",
                    help="Optimization step schedule (for methods that use iterative weight refinement); entered as a python lambda expression surrounded by quotes")

parser.add_argument('--trial', type=int, default=15,
                    help="The trial number - used to initialize random number generation (for replicability)")
parser.add_argument('--results_folder', type=str, default="results/",
                    help="This script will save results in this folder")
parser.add_argument('--verbosity', type=str, default="debug", choices=['error', 'warning', 'critical', 'info', 'debug'],
                    help="The verbosity level.")

# plotting arguments
plot_subparser.add_argument('plot_x', type=str, help="The X axis of the plot")
plot_subparser.add_argument('plot_y', type=str, help="The Y axis of the plot")
plot_subparser.add_argument('--plot_title', type=str, help="The title of the plot")
plot_subparser.add_argument('--plot_x_label', type=str, help="The X axis label of the plot")
plot_subparser.add_argument('--plot_y_label', type=str, help="The Y axis label of the plot")
plot_subparser.add_argument('--plot_x_type', type=str, choices=["linear", "log"], default="log",
                            help="Specifies the scale for the X-axis")
plot_subparser.add_argument('--plot_y_type', type=str, choices=["linear", "log"], default="log",
                            help="Specifies the scale for the Y-axis.")
plot_subparser.add_argument('--plot_legend', type=str, help="Specifies the variable to create a legend for.")
plot_subparser.add_argument('--plot_height', type=int, default=850, help="Height of the plot's html canvas")
plot_subparser.add_argument('--plot_width', type=int, default=850, help="Width of the plot's html canvas")
plot_subparser.add_argument('--plot_type', type=str, choices=['line', 'scatter'], default='scatter',
                            help="Type of plot to make")
plot_subparser.add_argument('--plot_fontsize', type=str, default='32pt', help="Font size for the figure, e.g., 32pt")
plot_subparser.add_argument('--plot_toolbar', action='store_true', help="Show the Bokeh toolbar")
plot_subparser.add_argument('--groupby', type=str,
                            help='The command line argument group rows by before plotting. No groupby means plotting raw data; groupby will do percentile stats for all data with the same groupby value. E.g. --groupby Ms in a scatter plot will compute result statistics for fixed values of M, i.e., there will be one scatter point per value of M')

arguments = parser.parse_args()
arguments.func(arguments)
# run(arguments)

