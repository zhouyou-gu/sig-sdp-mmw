import numpy as np
import scipy

from sim_src.scipy_util import *
from sim_src.util import profile

from sim_src.alg.interface import feasibility_check_alg


class mmwm_scipy(feasibility_check_alg):
    def __init__(self,K,min_sinr):
        self.K = K
        self.min_sinr = min_sinr

        self.H = None
        self.H_x = None
        self.H_y = None
        self.s_max = None
        self.I_max = None

        self.width_bound = 0.

    @profile
    def run_fc(self, Z, num_iterations = None) -> bool:
        if num_iterations is None:
            num_iterations = self._get_niteration()

        ## X(1) is a identity matrix
        I = np.asarray(self.H.sum(axis=1)).ravel()
        I_v = I/self.I_max
        notidx = I_v<=Z
        idx = np.invert(notidx)
        C = self.H.copy()
        csr_zero_rows_inplace(C, notidx)
        csr_scal_cons_inplace(C, 0.5)
        CC = C + C.transpose()
        id = scipy.sparse.identity(self.K)

        I_all = np.sum(I_v[idx])/(Z*self.K)
        id.data = id.data * I_all
        CC = CC + id


        L_sum = scipy.sparse.csr_matrix((self.K,self.K))
        L_sum += CC
        d_sum = 0
        d_best = I_all*self.K - idx.sum()
        d_sum += d_best
        X_best = scipy.sparse.identity(self.K)

        return CC
        # for i in range(10):
        #     csr_expm_rank_dsketch(csr_scal_cons_inplace(CC,0.5),K=self.K,d=Z,r=Z)
        #     pass



    def set_st(self, state):
        self._process_state(state)
        self._compute_width_bound()

    @profile
    def _process_state(self, state):
        print(state.__class__)
        ss = state.toarray()
        asso = np.argmax(ss,axis=1)
        self.s_max = np.max(ss,axis=1)
        self.I_max = self.s_max/self.min_sinr-1.

        self.H = state[:,asso]
        self.H.setdiag(0)

        self.H_x, self.H_y = self.H.nonzero()
        # tmp = np.random.randn(self.K,10)
        # values = np.sum(tmp[self.H_x] * tmp[self.H_y], axis=-1)
        # C = scipy.sparse.coo_matrix((values, (self.H_x, self.H_y)),shape=(self.K, self.K))

    def _compute_width_bound(self):
        self.width_bound = 100

    def _get_niteration(self):
        return self.width_bound**2


if __name__ == '__main__':
    from sim_src.env.env import env
    e = env()
    print(e.min_sinr)
    a = mmwm_scipy(e.n_sta,e.min_sinr)
    a.set_st(e.rxpr)
    a.run_fc(10)
