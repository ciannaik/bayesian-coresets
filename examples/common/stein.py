# The below stein kernel functions were taken from:
# https://github.com/pierreablin/ksddescent/blob/main/ksddescent/kernels.py
# with minor modifications to convert pytorch code to numpy code
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

def gaussian_stein_kernel(x, y, scores_x, scores_y, sigma):
    """Compute the Gaussian Stein kernel between x and y
    Parameters
    ----------
    x : numpy array, shape (n, p)
        Input particles
    y : numpy array, shape (n, p)
        Input particles
    score_x : numpy array, shape (n, p)
        The score of x
    score_y : numpy array, shape (n, p)
        The score of y
    sigma : float
        Bandwidth
    Return
    ------
    the stein discrepancy V statistic estimate
    """
    _, p = x.shape
    d = x[:, None, :] - y[None, :, :]
    dists = (d ** 2).sum(axis=-1)
    k = torch.exp(-dists / sigma / 2)
    scalars = scores_x.dot(scores_y.T)
    scores_diffs = scores_x[:, None, :] - scores_y[None, :, :]
    diffs = (d * scores_diffs).sum(axis=-1)
    der2 = p - dists / sigma
    stein_kernel = k * (scalars + diffs / sigma + der2 / sigma)
    return stein_kernel.sum()/(x.shape[0]*y.shape[0])


def imq_kernel(x, y, score_x, score_y, g=1, beta=0.5):
    """Compute the IMQ Stein kernel between x and y
    Parameters
    ----------
    x : numpy array, shape (n, p)
        Input particles
    y : numpy array, shape (n, p)
        Input particles
    score_x : numpy array, shape (n, p)
        The score of x
    score_y : numpy array, shape (n, p)
        The score of y
    g : float
        Bandwidth
    beta : float
        Power of the kernel
    Return
    ------
    the stein discrepancy V statistic estimate
    """
    _, p = x.shape
    d = x[:, None, :] - y[None, :, :]
    dists = (d ** 2).sum(axis=-1)
    res = 1 + g * dists
    kxy = res ** (-beta)
    scores_d = score_x[:, None, :] - score_y[None, :, :]
    temp = d * scores_d
    dkxy = 2 * beta * g * (res) ** (-beta - 1) * temp.sum(axis=-1)
    d2kxy = 2 * (
        beta * g * (res) ** (-beta - 1) * p
        - 2 * beta * (beta + 1) * g ** 2 * dists * res ** (-beta - 2)
    )
    k_pi = score_x.dot(score_y.T) * kxy + dkxy + d2kxy
    return k_pi.sum()/(x.shape[0]*y.shape[0])
