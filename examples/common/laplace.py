import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular

class LaplaceApprox(object):
    def __init__(self, log_joint, grad_log_joint, th0, hess_log_joint=None, diag_hess_log_joint=None, trials=10):
        self.log_joint = log_joint
        self.grad_log_joint = grad_log_joint
        self.trials = trials
        self.th0 = th0
        if (hess_log_joint is None and diag_hess_log_joint is None) or ((hess_log_joint is not None) and (diag_hess_log_joint is not None)):
            raise ValueError("ERROR: Exactly one of hess_log_joint or diag_hess_log_joint must be specified")
        if hess_log_joint is not None:
            self.hess_log_joint = hess_log_joint
            self.diag = False
        else:
            self.hess_log_joint = diag_hess_log_joint
            self.diag = True

    def build(self):
        _th0 = self.th0.copy()
        for i in range(self.trials):
            try:
                res = minimize(lambda mu: -self.log_joint(mu), _th0,
                               jac=lambda mu: -self.grad_log_joint(mu))
                self.th = res.x
            except Exception as e:
                print(e)
                _th0 += np.sqrt((_th0 ** 2).sum()) * 0.1 * np.random.randn(_th0.shape[0])
                continue
            break
        if self.diag:
            self.LSigInv = np.sqrt(-self.hess_log_joint(self.th))
            self.LSig = 1. / LSigInv
        else:
            self.LSigInv = np.linalg.cholesky(-self.hess_log_joint(self.th))
            self.LSig = solve_triangular(LSigInv, np.eye(LSigInv.shape[0]), lower=True, overwrite_b=True, check_finite=False)

    def sample(self, n):
        if self.diag:
            return self.th + self.LSig*np.random.randn(n, self.th.shape[0])
        else:
            return self.th + (self.LSig.dot(np.random.randn(self.th.shape[0],n))).T
