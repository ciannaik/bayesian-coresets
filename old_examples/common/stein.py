# The below stein kernel functions were modified using the following code as a base:
# https://github.com/pierreablin/ksddescent/blob/main/ksddescent/kernels.py
# under the following license:

# MIT License
#
# Copyright (c) 2021 pierreablin
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np

def gauss_mmd(x, y, sigma=1):
    d = x.shape[1]

    # K(X,X)
    xx_diffs = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    xx_sq_dists = (xx_diffs**2).sum(axis=2)
    kernel_xx = np.exp(-xx_sq_dists/(2.*sigma**2))

    # K(Y,Y)
    yy_diffs = y[:, np.newaxis, :] - y[np.newaxis, :, :]
    yy_sq_dists = (yy_diffs**2).sum(axis=2)
    kernel_yy = np.exp(-yy_sq_dists/(2.*sigma**2))

    # K(X, Y)
    xy_diffs = x[:, np.newaxis, :] - y[np.newaxis, :, :]
    xy_sq_dists = (xy_diffs**2).sum(axis=2)
    kernel_xy = np.exp(-xy_sq_dists/(2.*sigma**2))

    # sum K(X,X)/N^2 + sum K(Y,Y)/M^2 - 2 sum K(X,Y)/(M*N)
    return kernel_xx.sum()/x.shape[0]**2 + kernel_yy.sum()/y.shape[0]**2 - 2.*kernel_xy.sum()/x.shape[0]*y.shape[0]

def imq_mmd(x, y, sigma=1, beta=0.5):
    d = x.shape[1]

    # K(X,X)
    xx_diffs = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    xx_sq_dists = (xx_diffs**2).sum(axis=2)
    kernel_xx = 1./(xx_sq_dists/(2.*sigma**2) + 1.)**beta

    # K(Y,Y)
    yy_diffs = y[:, np.newaxis, :] - y[np.newaxis, :, :]
    yy_sq_dists = (yy_diffs**2).sum(axis=2)
    kernel_yy = 1./(yy_sq_dists/(2.*sigma**2) + 1.)**beta

    # K(X, Y)
    xy_diffs = x[:, np.newaxis, :] - y[np.newaxis, :, :]
    xy_sq_dists = (xy_diffs**2).sum(axis=2)
    kernel_xy = 1./(xy_sq_dists/(2.*sigma**2) + 1.)**beta

    # sum K(X,X)/N^2 + sum K(Y,Y)/M^2 - 2 sum K(X,Y)/(M*N)
    return kernel_xx.sum()/x.shape[0]**2 + kernel_yy.sum()/y.shape[0]**2 - 2.*kernel_xy.sum()/x.shape[0]*y.shape[0]


def gauss_stein(x, scores, sigma=1):
    _, p = x.shape
    d = x[:, None, :] - x[None, :, :]
    dists = (d ** 2).sum(axis=-1)
    k = np.exp(-dists / sigma**2 / 2)
    scalars = scores.dot(scores.T)
    scores_diffs = scores[:, None, :] - scores[None, :, :]
    diffs = (d * scores_diffs).sum(axis=-1)
    der2 = p - dists / sigma**2
    stein_kernel = k * (scalars + diffs / sigma**2 + der2 / sigma**2)
    return stein_kernel.sum()/(x.shape[0]*y.shape[0])


def imq_stein(x, scores, sigma=1, beta=0.5):
    _, p = x.shape
    d = x[:, None, :] - x[None, :, :]
    dists = (d ** 2).sum(axis=-1)
    res = 1 + dists /(2.*sigma**2)
    kxy = res ** (-beta)
    scores_d = scores[:, None, :] - scores[None, :, :]
    temp = d * scores_d
    dkxy = 2 * beta /(2.*sigma**2) * (res) ** (-beta - 1) * temp.sum(axis=-1)
    d2kxy = 2 * (
        beta / (2.*sigma**2) * (res) ** (-beta - 1) * p
        - 2 * beta * (beta + 1) /(2.*sigma**2)** 2 * dists * res ** (-beta - 2)
    )
    k_pi = scores.dot(scores.T) * kxy + dkxy + d2kxy
    return k_pi.sum() / (x.shape[0] ** 2)
