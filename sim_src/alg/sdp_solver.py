import numpy as np
import scipy
import cvxpy as cp

from sim_src.util import STATS_OBJECT, profile


class sdp_solver:
    def run_with_state(self, iteration, Z, state):
        pass

    @profile
    def rounding(self,Z,gX,state,nattempt=1):
        K = gX.shape[0]
        D = gX.shape[1]
        S_gain = state[0]
        Q_asso = state[1]
        h_max = state[2]
        S_gain_T_no_diag = S_gain.transpose()
        S_gain_T_no_diag.setdiag(0)
        not_assigned = np.ones(K, dtype=bool)


        z_vec = np.zeros(K)
        ZZ = 0
        for z in range(Z):
            ZZ += 1
            tmp_gain_sum = np.zeros(K)
            tmp_asso_sum = np.zeros(K)
            k_list_z = []
            for n in range(nattempt):
                k_list = []
                # randv = np.random.randn(D,1)
                kindx = np.arange(K)[not_assigned]
                krand = kindx[np.random.choice(kindx.size)]
                randv = np.matmul(gX[not_assigned],gX[krand]).ravel()
                krank = kindx[np.argsort(-randv)]
                for i in range(krank.size):
                    tmp = k_list.copy()
                    tmp.append(krank[i])
                    # do interference check
                    tmp_h = np.asarray(S_gain_T_no_diag[krank[i]].toarray()).ravel()
                    vio = (tmp_gain_sum[tmp] + tmp_h[tmp]) > h_max[tmp]
                    if np.any(vio == True):
                        continue

                    # do association check
                    tmp_a = np.asarray(Q_asso[krank[i]].toarray()).ravel()
                    vio = (tmp_asso_sum[tmp] + tmp_a[tmp]) >= 1

                    if np.any(vio == True):
                        continue

                    tmp_gain_sum += tmp_h
                    tmp_asso_sum += tmp_a
                    k_list.append(krank[i])

                if len(k_list) > len(k_list_z):
                    k_list_z = k_list
            z_vec[k_list_z] = z
            not_assigned[k_list_z] = False
            if np.all(not_assigned == False):
                break

        if not np.all(not_assigned == False):
            z_vec[not_assigned] = np.random.randint(Z,size = int(not_assigned.sum()))

        return z_vec, ZZ, np.sum(not_assigned)
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
