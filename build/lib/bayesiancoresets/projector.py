import numpy as np
from .util.errors import NumericalPrecisionError

class Projector(object):
    def project(self, pts, grad=False):
        raise NotImplementedError

    def update(self, wts, pts):
        raise NotImplementedError

    def mv(self):
        raise NotImplementedError

class BlackBoxProjector(Projector):
    def __init__(self, sampler, projection_dimension, loglikelihood, grad_loglikelihood = None):
        self.projection_dimension = projection_dimension
        self.sampler = sampler
        self.loglikelihood = loglikelihood
        self.grad_loglikelihood = grad_loglikelihood
        self.update(np.array([]), np.array([]))

    def project(self, pts, grad=False, return_sum=False):
        if return_sum:
            lls_sum = 0
            for i in range(pts.shape[0]):
                lls_current = self.loglikelihood(pts[i, :], self.samples)
                lls_current -= lls_current.mean(axis=1)[:, np.newaxis]
                lls_sum += lls_current
            return lls_sum.ravel()
        else:
            lls = self.loglikelihood(pts, self.samples)
            lls -= lls.mean(axis=1)[:,np.newaxis]
            if grad:
                if self.grad_loglikelihood is None:
                    raise ValueError('grad_loglikelihood was requested but not initialized in BlackBoxProjector.project')
                glls = self.grad_loglikelihood(pts, self.samples)
                glls -= glls.mean(axis=2)[:, :, np.newaxis]
                return lls, glls
            else:
                return lls

    def mv(self):
        return self.muw, self.Lsigw

    def update(self, wts, pts):
        self.samples, self.muw, self.Lsigw = self.sampler(self.projection_dimension, wts, pts)

class ImportanceSamplingProjector(Projector):
    def __init__(self, sampler, projection_dimension, loglikelihood, grad_loglikelihood = None):
        self.projection_dimension = projection_dimension
        self.sampler = sampler
        self.loglikelihood = loglikelihood
        self.grad_loglikelihood = grad_loglikelihood
        self.update(np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), 1, 1)


    def project(self, pts, grad=False):
        # TODO: fix this
        lls = self.loglikelihood(pts, self.samples)
        lls -= lls.dot(self.is_weights)[:,np.newaxis]
        lls = np.multiply(np.sqrt(self.is_weights),lls)
        if grad:
            raise ValueError('grad_loglikelihood not implemented for IS')
            # if self.grad_loglikelihood is None:
            #     raise ValueError('grad_loglikelihood was requested but not initialized in BlackBoxProjector.project')
            # glls = self.grad_loglikelihood(pts, self.samples)
            # glls -= glls.mean(axis=2)[:, :, np.newaxis]
            # return lls, glls
        else:
            return lls

    def update(self, data, data_rec, wts, idcs, pts, temp, temp_rec):
        self.samples, self.is_weights = self.sampler(self.projection_dimension, data, data_rec, wts, idcs, pts, temp, temp_rec)


class ImportanceSamplingMCMCProjector(Projector):
    def __init__(self, sampler, projection_dimension, loglikelihood, grad_loglikelihood = None):
        self.projection_dimension = projection_dimension
        self.sampler = sampler
        self.loglikelihood = loglikelihood
        self.grad_loglikelihood = grad_loglikelihood
        self.samples = None
        self.is_weights = None
        self.update(np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), 1, 1)


    def project(self, pts, grad=False):
        # TODO: fix this
        lls = self.loglikelihood(pts, self.samples)
        lls -= lls.dot(self.is_weights)[:,np.newaxis]
        lls = np.multiply(np.sqrt(self.is_weights),lls)
        if grad:
            raise ValueError('grad_loglikelihood not implemented for IS')
            # if self.grad_loglikelihood is None:
            #     raise ValueError('grad_loglikelihood was requested but not initialized in BlackBoxProjector.project')
            # glls = self.grad_loglikelihood(pts, self.samples)
            # glls -= glls.mean(axis=2)[:, :, np.newaxis]
            # return lls, glls
        else:
            return lls

    def update(self, data, data_rec, wts, idcs, pts, ptsX, ptsY, temp, temp_rec):
        self.samples, self.is_weights = self.sampler(self.projection_dimension, self.samples, self.is_weights, data, data_rec, wts, idcs, pts, ptsX,
                                                     ptsY, temp, temp_rec)

    def reset_samples(self):
        self.samples = None
        self.is_weights = None
        self.update(np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), 1, 1)