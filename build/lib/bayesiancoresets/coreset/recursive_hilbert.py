import numpy as np
from ..util.errors import NumericalPrecisionError
from ..snnls.giga import GIGA
from .coreset import Coreset

class RecursiveHilbertCoreset(Coreset):
  def __init__(
          self,
          data,
          ll_projector,
          n_subsample=None,
          snnls=GIGA,
          r_subsample_prob=0.5,
          temp = 1.,
          temp_sched=lambda b : b/1.1,
          **kw
  ):
    if n_subsample is None:
      # user requested to work with the whole dataset
      sub_idcs = np.arange(data.shape[0])
      vecs = ll_projector.project(data)
    else:
      # user requested to work with a subsample of the large dataset
      # randint is efficient (doesn't enumerate all possible indices) but we need to call unique after to avoid duplicates
      sub_idcs = np.unique(np.random.randint(data.shape[0], size=n_subsample))
      # vecs = ll_projector.project(data[sub_idcs])

      # TODO: add this back in + fix
      # remove any zero vectors; won't affect the coreset and may cause exception in snnls
      # nonzero_vecs = np.sqrt((vecs ** 2).sum(axis=1)) > 0.
      # sub_idcs = sub_idcs[nonzero_vecs]
      # vecs = vecs[nonzero_vecs, :]

    # TODO: add this back in + fix
    self.snnls_method = snnls
    self.sub_idcs = sub_idcs
    self.data = data
    self.temp = temp
    self.temp_sched = temp_sched
    self.r_subsample_prob = r_subsample_prob
    self.ll_projector = ll_projector
    self.n_subsample = None if n_subsample is None else min(data.shape[0], n_subsample)
    super().__init__(**kw)

  def reset(self):
    self.snnls.reset()
    super().reset()

  def _build_nonrecursive(self, itrs):
    self.snnls.build(itrs)
    w = self.snnls.weights()
    self.wts = w[w>0]
    self.idcs = self.sub_idcs[w>0]
    self.pts = self.data[self.idcs]

  def _build(self, itrs):
    self.wts, self.idcs, self.pts = self._recursive_build(itrs, self.sub_idcs, self.r_subsample_prob, self.temp)

  def _recursive_build(self, itrs, data_idcs, sub_p, temp):
    data = self.data[data_idcs]
    # base case when the size of the data is <= the coreset size
    if data.shape[0] <= itrs:
      wts = np.ones(data.shape[0])
      idcs = data_idcs
      pts = data
      return wts, idcs, pts

    # otherwise, change tempering, subsample, and recurse
    temp_rec = self.temp_sched(temp)
    rec_idcs = np.random.random(data.shape[0]) < sub_p
    if not any(rec_idcs):
      rec_idcs[np.random.randint(data_idcs.shape[0])] = True
    data_idcs_rec = data_idcs[rec_idcs]

    wts_proj, idcs_proj, pts_proj = self._recursive_build(itrs, data_idcs_rec, sub_p, temp_rec)

    # wts_proj = np.zeros(data.shape[0])
    # wts_proj[rec_idcs] = wts_rec
    #
    # pts_proj = np.zeros(data.shape)
    # pts_proj[rec_idcs] = pts_rec

    # Use the recursively build coreset to get an IS projection
    vecs, _, _ = self._get_projection(self.n_subsample,
                                      data_idcs,
                                      data_idcs_rec,
                                      wts_proj,
                                      idcs_proj,
                                      pts_proj,
                                      temp)

    # Use the projection to build the coreset
    self.snnls = self.snnls_method(vecs.T, vecs.sum(axis=0))

    self.snnls.build(itrs)
    w = self.snnls.weights()
    wts = w[w > 0]
    idcs = data_idcs[w > 0]
    pts = self.data[idcs]
    return wts, idcs, pts

  def _get_projection(self, n_subsample, data_idcs, data_idcs_rec, w, i, p, temp):
    data = self.data[data_idcs]
    data_rec = self.data[data_idcs_rec]
    #update the projector
    # TODO: this needs to take the data as an argument as well, figure out a way to incorporate this into the original method
    # self.ll_projector.update(w, p)

    temp_rec = self.temp_sched(temp)

    self.ll_projector.update(data, data_rec, w, i, p, temp, temp_rec)

    #construct a tangent space
    if n_subsample is None:
      #user requested to work with the whole dataset
      sub_idcs = None
      vecs = self.ll_projector.project(data)
      sum_scaling = 1.
    else:
      # user requested to work with a subsample of the large dataset
      # randint is efficient (doesn't enumerate all possible indices) but we need to call unique after to avoid duplicates
      sub_idcs = np.random.randint(data.shape[0], size=n_subsample)
      vecs = self.ll_projector.project(data[sub_idcs])
      sum_scaling = data.shape[0]/n_subsample

      # remove any zero vectors; won't affect the coreset and may cause exception in snnls
      nonzero_vecs = np.sqrt((vecs ** 2).sum(axis=1)) > 0.
      sub_idcs = sub_idcs[nonzero_vecs]
      vecs = vecs[nonzero_vecs, :]

    return vecs, sum_scaling, sub_idcs

  def _optimize(self):
    self.snnls.optimize()
    w = self.snnls.weights()
    self.wts = w[w>0]
    self.idcs = self.sub_idcs[w>0]
    self.pts = self.data[self.idcs]

  def error(self):
    return self.snnls.error()


class RecursiveHilbertCoresetMCMC(Coreset):
  def __init__(
          self,
          data,
          ll_projector,
          n_subsample=None,
          snnls=GIGA,
          r_subsample_prob=0.5,
          temp = 1.,
          temp_sched=lambda b : b/1.1,
          dataX=None,
          dataY=None,
          **kw
  ):
    if n_subsample is None:
      # user requested to work with the whole dataset
      sub_idcs = np.arange(data.shape[0])
      vecs = ll_projector.project(data)
    else:
      # user requested to work with a subsample of the large dataset
      # randint is efficient (doesn't enumerate all possible indices) but we need to call unique after to avoid duplicates
      sub_idcs = np.unique(np.random.randint(data.shape[0], size=n_subsample))
      # vecs = ll_projector.project(data[sub_idcs])

      # TODO: add this back in + fix
      # remove any zero vectors; won't affect the coreset and may cause exception in snnls
      # nonzero_vecs = np.sqrt((vecs ** 2).sum(axis=1)) > 0.
      # sub_idcs = sub_idcs[nonzero_vecs]
      # vecs = vecs[nonzero_vecs, :]

    # TODO: add this back in + fix
    self.snnls_method = snnls
    self.sub_idcs = sub_idcs
    self.data = data
    self.dataX = dataX
    self.dataY = dataY
    self.temp = temp
    self.temp_sched = temp_sched
    self.r_subsample_prob = r_subsample_prob
    self.ll_projector = ll_projector
    self.n_subsample = None if n_subsample is None else min(data.shape[0], n_subsample)
    super().__init__(**kw)

  def reset(self):
    self.snnls.reset()
    super().reset()

  def _build_nonrecursive(self, itrs):
    self.snnls.build(itrs)
    w = self.snnls.weights()
    self.wts = w[w>0]
    self.idcs = self.sub_idcs[w>0]
    self.pts = self.data[self.idcs]

  def _build(self, itrs):
    self.wts, self.idcs, self.pts, _, _ = self._recursive_build(itrs, self.sub_idcs, self.r_subsample_prob, self.temp)


  def _recursive_build(self, itrs, data_idcs, sub_p, temp):
    data = self.data[data_idcs]
    dataX = self.dataX[data_idcs]
    dataY = self.dataY[data_idcs]
    # Reset projector samples and weights
    self.ll_projector.reset_samples()

    # base case when the size of the data is <= the coreset size
    if data.shape[0] <= itrs:
      wts = np.ones(data.shape[0])
      idcs = data_idcs
      pts = data
      ptsX = dataX
      ptsY = dataY
      return wts, idcs, pts, ptsX, ptsY

    # otherwise, change tempering, subsample, and recurse
    temp_rec = self.temp_sched(temp)
    rec_idcs = np.random.random(data.shape[0]) < sub_p
    if not any(rec_idcs):
      rec_idcs[np.random.randint(data_idcs.shape[0])] = True
    data_idcs_rec = data_idcs[rec_idcs]

    wts_proj, idcs_proj, pts_proj, ptsX_proj, ptsY_proj = self._recursive_build(itrs, data_idcs_rec, sub_p, temp_rec)

    # wts_proj = np.zeros(data.shape[0])
    # wts_proj[rec_idcs] = wts_rec
    #
    # pts_proj = np.zeros(data.shape)
    # pts_proj[rec_idcs] = pts_rec

    # Use the recursively build coreset to get an IS projection
    vecs, _, _ = self._get_projection(self.n_subsample,
                                      data_idcs,
                                      data_idcs_rec,
                                      wts_proj,
                                      idcs_proj,
                                      pts_proj,
                                      ptsX_proj,
                                      ptsY_proj,
                                      temp)

    # Use the projection to build the coreset
    self.snnls = self.snnls_method(vecs.T, vecs.sum(axis=0))

    self.snnls.build(itrs)
    w = self.snnls.weights()
    wts = w[w > 0]
    idcs = data_idcs[w > 0]
    pts = self.data[idcs]
    ptsX = self.dataX[idcs]
    ptsY = self.dataY[idcs]

    return wts, idcs, pts, ptsX, ptsY

  def _get_projection(self, n_subsample, data_idcs, data_idcs_rec, w, i, p, pX, pY, temp):
    data = self.data[data_idcs]
    data_rec = self.data[data_idcs_rec]
    #update the projector
    # TODO: this needs to take the data as an argument as well, figure out a way to incorporate this into the original method
    # self.ll_projector.update(w, p)

    temp_rec = self.temp_sched(temp)

    self.ll_projector.update(data, data_rec, w, i, p, pX, pY, temp, temp_rec)

    #construct a tangent space
    if n_subsample is None:
      #user requested to work with the whole dataset
      sub_idcs = None
      vecs = self.ll_projector.project(data)
      sum_scaling = 1.
    else:
      # user requested to work with a subsample of the large dataset
      # randint is efficient (doesn't enumerate all possible indices) but we need to call unique after to avoid duplicates
      sub_idcs = np.random.randint(data.shape[0], size=n_subsample)
      vecs = self.ll_projector.project(data[sub_idcs])
      sum_scaling = data.shape[0]/n_subsample

      # remove any zero vectors; won't affect the coreset and may cause exception in snnls
      nonzero_vecs = np.sqrt((vecs ** 2).sum(axis=1)) > 0.
      sub_idcs = sub_idcs[nonzero_vecs]
      vecs = vecs[nonzero_vecs, :]

    return vecs, sum_scaling, sub_idcs

  def _optimize(self):
    self.snnls.optimize()
    w = self.snnls.weights()
    self.wts = w[w>0]
    self.idcs = self.sub_idcs[w>0]
    self.pts = self.data[self.idcs]
    self.pts = self.pts[:,-1]*self.pts[:,-1:]

  def error(self):
    return self.snnls.error()
