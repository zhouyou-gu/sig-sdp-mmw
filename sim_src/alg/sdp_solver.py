import numpy as np
import scipy
import cvxpy as cp

from sim_src.util import STATS_OBJECT, profile


class sdp_solver:
    def run_with_state(self, iteration, Z, state):
        pass

    # @profile
    def rounding(self,Z,gX,state,nattempt=1):
        pass

class pdip(sdp_solver, STATS_OBJECT):
    def __init__(self, nit=100, alpha=1.):
        self.nit = nit
        self.alpha = alpha

    def run_with_state(self, iteration, Z, state):
        ps_tic = self._get_tic()
        ret = self._process_state(Z, state[0],state[1],state[2])
        tim = self._get_tim(ps_tic)
        K = state[0].shape[0]
        self._add_np_log("pdip_state_process",iteration,np.array([Z,K,tim]))
        return ret

    def _process_state(self, Z, S_gain, Q_asso, h_max):
        K = S_gain.shape[0]

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
    def run_fc(self, Z, num_iterations=None, solver = cp.SCS):


        if solver == cp.SCS:
            self.prob.solve()

    pass
class admm(sdp_solver):
    pass
