import numpy as np
import scipy
import cvxpy as cp

from sim_src.util import STATS_OBJECT


class sdp_solver:
    def run_with_state(self, iteration, Z, state):
        pass

    def rounding(self,Z,gX,state,nattempt=1):
        K = gX.shape[0]
        D = gX.shape[1]
        S_gain = state[0]
        Q_asso = state[1]
        h_max = state[3]
        S_gain_T_no_diag = S_gain.transpose()
        S_gain_T_no_diag.setdiag(0)
        not_assigned = np.ones(K, dtype=bool)

        z_vec = np.zeros(K)
        for z in range(Z):
            k_list_z = []
            for n in range(nattempt):
                k_list = []
                randv = np.random.randn(D,1)
                randv = np.asarray(scipy.sparse.linalg.expm_multiply(gX[not_assigned],randv)).ravel()
                kindx = np.arange(K)[not_assigned]
                krank = kindx[np.argsort(randv[not_assigned])]
                for i in range(krank.size):
                    tmp = k_list.copy()
                    tmp.append(krank[i])
                    # do interference check
                    SS = S_gain_T_no_diag[tmp,:].tocsc()
                    SS = SS[:,tmp].tocsr()
                    h = np.asarray(SS.sum(axis=1)).ravel()
                    vio = h > h_max[tmp]
                    if np.any(vio == True):
                        continue

                    # do association check
                    QQ = Q_asso[tmp,:].tocsc()
                    QQ = QQ[:,tmp].tocsr()
                    q = np.asarray(QQ.sum(axis=1)).ravel()
                    if np.any(q >= 1.):
                        continue
                    k_list.append(krank[i])

                if len(k_list) > len(k_list_z):
                    k_list_z = k_list

            z_vec[k_list_z] = z
            not_assigned[k_list_z] = False

        if not np.all(not_assigned == False):
            z_vec[not_assigned] = np.random.randint(Z,size = int(not_assigned.sum()))

        return z_vec, Z
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
