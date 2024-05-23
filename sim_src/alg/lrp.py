import numpy as np
import scipy
import cvxpy as cp

from sim_src.util import STATS_OBJECT, profile

class lrp_solver(STATS_OBJECT):
    def __init__(self, nit=100, alpha=1.):
        self.nit = nit
        self.alpha = alpha

    def run_with_state(self, bs_iteration, Z, state):
        ps_tic = self._get_tic()
        prob, P = self._setup_problem(Z,state[0],state[1],state[2])
        tim = self._get_tim(ps_tic)
        K = state[0].shape[0]
        self._add_np_log("lrp_problem_setup",bs_iteration,np.array([Z,K,tim]))

        solving_tic = self._get_tic()
        prob.solve(solver=cp.SCS, max_iters=self.nit)
        tim = self._get_tim(solving_tic)
        self._add_np_log("lrp_solve",bs_iteration,np.array([Z,K,tim]))
        return True, P.value

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


        P = cp.Variable((Z,K))
        P.value = np.ones((Z,K),dtype=float)/Z
        S = np.asarray(S_gain_T_no_asso_no_diag.toarray()).transpose()
        PS = P @ S
        P_ub = cp.multiply(P , np.tile(-(S_sum-h_max),(Z,1))) + np.tile(S_sum,(Z,1))
        const_P_lb = (P>=0.)
        const_P_ub = (P<=1.)
        const_P = (P.sum(0) == 1)
        const_F = (P[:,nz_idx_asso_x] + P[:,nz_idx_asso_y] <= 1)
        const_H = (PS <= P_ub)
        constraints = [const_P_lb,const_P_ub,const_P,const_F,const_H]

        prob = cp.Problem(cp.Minimize(0), constraints)

        return prob, P

    def rounding(self,Z,gX,state,nattempt=1):
        K = state[0].shape[0]
        D = gX.shape[1]
        S_gain = state[0].copy()
        Q_asso = state[1]
        h_max = state[2]
        S_gain.setdiag(0)
        S_gain.eliminate_zeros()
        S_gain_T_no_diag = S_gain.transpose()

        # A_sum = np.asarray(Q_asso.sum(axis=1)).ravel()
        # rank = np.argsort(-A_sum)

        S_gain_index = np.split(S_gain.indices, S_gain.indptr)[1:-1]
        Q_asso_index = np.split(Q_asso.indices, S_gain.indptr)[1:-1]

        not_assigned = np.ones(K, dtype=bool)

        rank = np.arange(K)
        sorted_indices = np.argsort(-gX, axis=0)

        z_vec = np.zeros(K)

        gain_sum = []
        asso_sum = []
        slot_asn = []
        for z in range(Z):
            gain_sum.append(np.zeros(K))
            asso_sum.append(np.zeros(K))
            slot_asn.append([])

        user_sum = 0
        for kk in range(K):
            k = rank[kk]
            for zz in range(Z):
                z = sorted_indices[zz,k]

                if not not_assigned[k]:
                    break

                # do interference check
                neighbor_index = np.intersect1d(np.array(slot_asn[z]),S_gain_index[k])
                neighbor_index = np.append(neighbor_index,k).astype(int)
                tmp_h = np.asarray(S_gain[k].toarray()).ravel()
                vio = (gain_sum[z][neighbor_index] + tmp_h[neighbor_index]) > h_max[neighbor_index]
                if np.any(vio == True):
                    continue

                # do association check
                neighbor_index = np.intersect1d(np.array(slot_asn[z]),Q_asso_index[k])
                neighbor_index = np.append(neighbor_index,k).astype(int)
                tmp_a = np.asarray(Q_asso[k].toarray()).ravel()
                vio = (asso_sum[z][neighbor_index] + tmp_a[neighbor_index]) >= 1
                if np.any(vio == True):
                    continue

                gain_sum[z] += np.asarray(S_gain[k].toarray()).ravel()
                asso_sum[z] += np.asarray(Q_asso[k].toarray()).ravel()
                slot_asn[z].append(k)

                user_sum += 1
                not_assigned[k] = False
                z_vec[k] = z
                break


        if not np.all(not_assigned == False):
            z_vec[not_assigned] = np.random.randint(Z,size = int(not_assigned.sum()))

        return z_vec, Z, np.sum(not_assigned)