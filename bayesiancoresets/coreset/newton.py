import numpy as np
from ..util.errors import NumericalPrecisionError
from ..util.opt import nn_opt, an_opt
from .coreset import Coreset
from scipy.linalg import solve_triangular


class QuasiNewtonCoreset(Coreset):
    def __init__(self, data, projector, n_subsample_select=None, n_subsample_opt=None, opt_itrs=100,
                 step_sched=lambda i: 1. / (i + 1), **kw):
        self.data = data
        self.cts = []
        self.ct_idcs = []
        self.projector = projector
        self.n_subsample_select = None if n_subsample_select is None else min(data.shape[0], n_subsample_select)
        self.n_subsample_opt = None if n_subsample_opt is None else min(data.shape[0], n_subsample_opt)
        self.step_sched = step_sched
        self.opt_itrs = opt_itrs
        super().__init__(**kw)

    def reset(self):
        self.cts = []
        self.ct_idcs = []
        super().reset()

    def _build(self, size):
        # # reset data points
        # self.reset()
        # uniformly subset data points
        self._select(size)
        # optimize the weights
        self._optimize()

    def _get_projection(self, n_subsample, w, p, return_sum=False):
        # update the projector
        self.projector.update(w, p)

        # construct a tangent space
        if n_subsample is None:
            sub_idcs = None
            vecs = self.projector.project(self.data, return_sum=return_sum)
            sum_scaling = 1.
        else:
            sub_idcs = np.random.randint(self.data.shape[0], size=n_subsample)
            vecs = self.projector.project(self.data[sub_idcs], return_sum=return_sum)
            sum_scaling = self.data.shape[0] / n_subsample

        if self.pts.size > 0:
            corevecs = self.projector.project(self.pts)
        else:
            corevecs = np.zeros((0, vecs.shape[1]))

        return vecs, sum_scaling, sub_idcs, corevecs

    def _select(self, size):
        for i in range(size):
            f = np.random.randint(self.data.shape[0])
            if f in self.ct_idcs:
                self.cts[self.ct_idcs.index(f)] += 1
            else:
                self.ct_idcs.append(f)
                self.cts.append(1)
        self.wts = self.data.shape[0] * np.array(self.cts) / np.array(self.cts).sum()
        self.idcs = np.array(self.ct_idcs)
        self.pts = self.data[self.idcs]

    def _optimize(self):
        
        def search_direction(w, tau=0.01):
            # vecs_sum, sum_scaling, sub_idcs, corevecs = self._get_projection(self.n_subsample_opt, w, self.pts,return_sum=True)
            vecs, sum_scaling, sub_idcs, corevecs = self._get_projection(self.n_subsample_opt, w, self.pts,
                                                                                     return_sum=False)
            # resid = sum_scaling*vecs_sum - w.dot(corevecs)
            resid = sum_scaling * vecs.sum(axis=0) - w.dot(corevecs)

            corevecs_cov = corevecs.dot(corevecs.T) / corevecs.shape[1]
            # add regularization term to hessian
            np.fill_diagonal(corevecs_cov, corevecs_cov.diagonal() + tau)
            # np.fill_diagonal(corevecs_cov, (1+tau)*corevecs_cov.diagonal())
            print("Quasi-Hessian condition number: {}".format(np.linalg.cond(corevecs_cov)))
            grd = (corevecs.dot(resid) / corevecs.shape[1])
            print("gradient norm: {}".format(np.sqrt(((grd)**2).sum())))
            # output gradient of weights at idcs
            search_direction = np.linalg.solve(corevecs_cov, grd)
            return search_direction

        def grd(x, tau=0.01):
            # vecs_sum, sum_scaling, sub_idcs, corevecs, muw, Lsigw = self._get_projection(self.n_subsample_opt, w, self.pts,return_sum=True)
            vecs, sum_scaling, sub_idcs, corevecs = self._get_projection(self.n_subsample_opt, x, self.pts,
                                                                               return_sum=False)
            # resid = sum_scaling*vecs_sum - w.dot(corevecs)
            resid = sum_scaling * vecs.sum(axis=0) - x.dot(corevecs)
            grd = (corevecs.dot(resid) / corevecs.shape[1])
            return -grd

        x0 = self.wts
        self.wts = an_opt(x0, grd, search_direction, opt_itrs=self.opt_itrs, step_sched=self.step_sched)
        # use uniform weights if sum of weights is negative
        if self.wts.sum() <= 0:
            self.wts = self.data.shape[0] * np.ones(self.pts.shape[0]) / self.pts.shape[0]
