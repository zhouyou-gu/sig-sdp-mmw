import numpy as np
import scipy
import cvxpy as cp

from sim_src.util import STATS_OBJECT, profile


class sdp_solver:
    def __init__(self, nit=100, rank_radio=2, alpha=1.):
        self.nit = nit
        self.rank_radio = rank_radio
        self.alpha = alpha

    def run_with_state(self, iteration, Z, state):
        pass

    def rounding(self,Z,gX,state,nattempt=1):
        pass

class pdip_sdp_solver(sdp_solver, STATS_OBJECT):
    def __init__(self, nit=100, rank_radio=2, alpha=1.):
        sdp_solver.__init__(self, nit=nit, rank_radio=rank_radio, alpha=alpha)

    def run_with_state(self, iteration, Z, state):
        ps_tic = self._get_tic()
        prob, X = self._setup_problem(Z,state[0],state[1],state[2])
        tim = self._get_tim(ps_tic)
        K = state[0].shape[0]
        self._add_np_log("pdip_problem_setup",iteration,np.array([Z,K,tim]))

        ps_tic = self._get_tic()
        prob.solve()
        u, s, v = np.linalg.svd(X.value)
        rank = np.min([K , (Z-1)*self.rank_radio])
        X_half = u[:,0:rank] * np.sqrt(s[0:rank])
        tim = self._get_tim(ps_tic)
        self._add_np_log("pdip_solve",iteration,np.array([Z,K,tim]))
        return True, X_half

    def _setup_problem(self, Z, S_gain, Q_asso, h_max):
        K = S_gain.shape[0]
        S_gain_T_no_asso_no_diag = S_gain.copy().transpose()
        nz_idx_asso_x, nz_idx_asso_y = Q_asso.nonzero()
        S_gain_T_no_asso_no_diag[nz_idx_asso_x, nz_idx_asso_y] = 0
        S_gain_T_no_asso_no_diag.setdiag(0)
        S_gain_T_no_asso_no_diag.eliminate_zeros()
        S_gain_T_no_asso_no_diag.sort_indices()
        S_sum = np.asarray(S_gain_T_no_asso_no_diag.sum(axis=1)).ravel()

        s_max = S_gain.diagonal()
        S_gain_T_no_asso_no_diag_square = S_gain_T_no_asso_no_diag.copy()
        S_gain_T_no_asso_no_diag_square.data = S_gain_T_no_asso_no_diag_square.data ** 2
        norm_H = np.sqrt(np.asarray(S_gain_T_no_asso_no_diag_square.sum(axis=1)).ravel()) * (Z-1)/(2*Z) + np.abs(1/K*h_max-1/K/Z*S_sum)


        X = cp.Variable((K,K), symmetric=True)
        S = np.asarray(S_gain_T_no_asso_no_diag.toarray())
        SX = cp.multiply(S,X)
        const_D = (cp.diag(X) == 1)
        const_F = (X[nz_idx_asso_x,nz_idx_asso_y] <= -1./(Z-1))
        const_H = (SX.sum(axis=1)*(Z-1.)/Z <= (h_max -1./Z*S_sum))
        constraints = [X>>0,const_D,const_F,const_H]

        prob = cp.Problem(cp.Minimize(0), constraints)

        return prob, X
class admm(pdip_sdp_solver):
    pass
