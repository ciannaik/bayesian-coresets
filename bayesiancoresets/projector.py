import numpy as np
from .util.errors import NumericalPrecisionError

class Projector(object):
    def project(self, pts, grad=False):
        raise NotImplementedError

    def update(self, wts, pts):
        raise NotImplementedError

class BlackBoxProjector(Projector):
    def __init__(self, sampler, projection_dimension, loglikelihood, grad_loglikelihood = None):
        self.projection_dimension = projection_dimension
        self.sampler = sampler
        self.loglikelihood = loglikelihood
        self.grad_loglikelihood = grad_loglikelihood
        self.update(np.array([]), np.array([]))

    def project(self, pts, grad=False):
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

    def update(self, wts, pts):
        self.samples = self.sampler(self.projection_dimension, wts, pts)

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
        self.samples, self.is_weights = self.sampler(self.projection_dimension, data, data_rec, wts, idcs, pts, ptsX, ptsY, temp, temp_rec)
