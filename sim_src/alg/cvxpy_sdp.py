import math

import numpy as np
import cvxpy as cp


from sim_src.alg.power_adaption import power_adaption
from sim_src.alg.vec_rounding import vec_rand_rounding
from sim_src.scipy_util import *
from sim_src.util import profile, STATS_OBJECT

class cvxpy_sdp(STATS_OBJECT):
    def __init__(self, K, min_sinr):
        self.K = K
        self.min_sinr = min_sinr

        self.rxpr = None

        self.H = None
        self.H_x = None
        self.H_y = None
        self.s_max = None
        self.I_max = None

        self.prob = None
    def set_st(self, rxpr):
        self._process_state(rxpr)

    # @profile
    def _process_state(self, rxpr):
        self.rxpr = rxpr
        ss = rxpr.toarray()
        asso = np.argmax(ss, axis=1)

        self.s_max = np.max(ss, axis=1)
        self.I_max = self.s_max / self.min_sinr - 1.

        self.H = rxpr[:, asso]
        self.H.setdiag(0)
        self.H.eliminate_zeros()
        self.H = self.H.transpose()
        self.H = self.H.tocsr()
        self.H.sort_indices()
        self.H_x, self.H_y = self.H.nonzero()
        self.H = np.asarray(self.H.todense())

    def run_fc(self, Z, num_iterations=None, solver = cp.SCS):
        X = cp.Variable((self.K,self.K), symmetric=True)

        constraints = [X >> 0]

        constraints += [
            cp.sum(cp.multiply(self.H,  (1.+(Z-1.) * X)/Z),axis=1) <= self.I_max
        ]
        constraints += [
            cp.diag(X) <= 1.
        ]
        constraints += [
            cp.trace(X) == 1.
        ]
        constraints += [
             (1.+(Z-1.) * X[self.H_x, self.H_y])/Z <= self.I_max[self.H_x]
        ]
        self.prob = cp.Problem(cp.Minimize(0),
                          constraints)

        if solver == cp.SCS:
            self.prob.solve()

if __name__ == '__main__':
    import cvxpy as cp
    import sdpap

    n = 3
    p = 3
    np.random.seed(1)
    C = np.random.randn(n, n)
    A = []
    b = []
    for i in range(p):
        A.append(np.random.randn(n, n))
        b.append(np.random.randn())

    # Define and solve the CVXPY problem.
    # Create a symmetric matrix variable.
    X = cp.Variable((n,n), symmetric=True)
    # The operator >> denotes matrix inequality.
    constraints = [X >> 0]
    constraints += [
        cp.trace(A[i] @ X) == b[i] for i in range(p)
    ]
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)),
                      constraints)
    prob.solve(verbose=True,solver = cp.SDPA)


    print(cp.installed_solvers())
