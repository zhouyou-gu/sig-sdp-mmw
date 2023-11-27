import math

import numpy as np
import scipy

from sim_src.scipy_util import *
from sim_src.util import profile

from sim_src.alg.interface import feasibility_check_alg


class mmwm_scipy(feasibility_check_alg):
    ETA = 0.1
    OFFSET = 0.25
    def __init__(self, K, min_sinr):
        self.K = K
        self.min_sinr = min_sinr

        self.H = None
        self.H_x = None
        self.H_y = None
        self.s_max = None
        self.I_max = None

        self.width_bound = 0.

    @profile
    def run_fc_main(self, Z, num_iterations=None) -> bool:
        if num_iterations is None:
            num_iterations = self._get_niteration()

        d_sum = 0
        d_bst = np.infty
        X_bst = None

        G = scipy.sparse.csr_matrix((self.K, self.K))
        for i in range(num_iterations):
            # compute X from G
            G_2 = G.copy()
            G_2.data = G_2.data/2.
            e_half = np.asarray(scipy.sparse.linalg.expm(G_2.tocsc()).todense())

            X_mdiag_data = np.sum(e_half * e_half, axis=1)
            X_trace = np.sum(X_mdiag_data)/self.K
            X_mdiag_data = X_mdiag_data/X_trace
            X_mdiag = scipy.sparse.diags(X_mdiag_data).tocsr()
            X_mdiag.sort_indices()
            X_offdi_data = np.sum(e_half[self.H_x] * e_half[self.H_y], axis=1)/X_trace
            X_offdi = scipy.sparse.coo_matrix((X_offdi_data, (self.H_x, self.H_y)), shape=(self.K, self.K)).tocsr()
            X_offdi.sort_indices()

            X = X_offdi+X_mdiag

            # get L(X) and dX_L(X)
            # get LT(X) and dX_LT(X)
            T_violation_idx = (X_mdiag.data > 1.)
            T_violation_cnt = np.sum(T_violation_idx)
            T_violation_err = np.sum(X_mdiag.data[T_violation_idx] - 1.)
            T_dX_A = scipy.sparse.diags(T_violation_idx.astype(float)).tocsr()
            T_dX_I = scipy.sparse.diags(np.ones(self.K)*T_violation_cnt/self.K).tocsr()
            T_dX = T_dX_A - T_dX_I

            # get LI(X) and dX_LI(X)
            HX = X_offdi.copy()
            HX.data = ((X_offdi.data - 1.) * (Z - 1)) * self.H.data / Z
            I = np.asarray(HX.sum(axis=1)).ravel()
            I_max_I_sum = (self.I_max - (1.+ self.OFFSET)*self.I)
            I_violation_idx = (I > I_max_I_sum)
            I_violation_cnt = np.sum(I_violation_idx)
            I_violation_err = np.sum((I[I_violation_idx] - I_max_I_sum[I_violation_idx])/self.I[I_violation_idx])
            I_dX_A = csr_zero_rows_inplace(self.H.copy(),np.invert(I_violation_idx))
            I_dX_A = csr_scal_rows_inplace(I_dX_A,1./self.I)
            I_dX_A.data = I_dX_A.data*(Z-1.)/Z
            I_dX_A = I_dX_A + I_dX_A.transpose()
            I_dX_A.data = I_dX_A.data/2.
            I_dX_B = scipy.sparse.diags(np.ones(self.K)*np.sum((-self.I[I_violation_idx]*(Z-1)/Z - I_max_I_sum[I_violation_idx])/self.I[I_violation_idx])/self.K).tocsr()
            I_dX = I_dX_A+I_dX_B

            # get LX(X) and dX_LX(X)
            X_violation_idx = X_offdi.data < (-1. / (Z - 1.))
            X_violation_cnt = np.sum(X_violation_idx)
            X_violation_err = np.sum(X_offdi.data[X_violation_idx] * (1.-Z) - 1.)
            X_dX_A = X_offdi.copy()
            X_dX_A.data = np.zeros(X_dX_A.data.size)
            X_dX_A.data[X_violation_idx] = - (Z-1.)
            X_dX_A = X_dX_A + X_dX_A.transpose()
            X_dX_A.data = X_dX_A.data/2.
            X_dX_I = scipy.sparse.diags(np.ones(self.K)*X_violation_cnt/self.K).tocsr()
            X_dX = X_dX_A-X_dX_I

            sa_T, vh = scipy.sparse.linalg.eigsh(T_dX,k=1,which='LM')
            sa_I, vh = scipy.sparse.linalg.eigsh(I_dX,k=1,which='LM')
            sa_X, vh = scipy.sparse.linalg.eigsh(X_dX,k=1,which='LM')

            print(i,"PHO_T_I_X",sa_T[0],sa_I[0],sa_X[0])
            print(i,"ERR_T_I_X",T_violation_err,I_violation_err,X_violation_err)
            print(i,"ECT_T_I_X",T_violation_cnt,I_violation_cnt,X_violation_cnt)
            print(i,"ECT_T_I_X",T_violation_cnt/float(self.K),I_violation_cnt/float(self.K),X_violation_cnt/float(self.H.nnz))

            dLX = scipy.sparse.csr_matrix((self.K, self.K))
            LX = 0.
            if T_violation_err > 0:
                dLX = dLX + T_dX/np.abs(sa_T[0])
                LX += T_violation_err/np.abs(sa_T[0])

            if I_violation_err > 0:
                dLX = dLX + I_dX/np.abs(sa_I[0])
                LX += I_violation_err/np.abs(sa_I[0])

            if X_violation_err > 0:
                dLX = dLX + X_dX/np.abs(sa_X[0])
                LX += X_violation_err/np.abs(sa_X[0])

            sa, vh = scipy.sparse.linalg.eigsh(dLX,k=1,which='LM')
            tmp_PHO = np.abs(sa[0])

            print(i,"ERR_PHO_SUM",LX,tmp_PHO,"ERR/PHO", LX / tmp_PHO)

            # update G
            G = G  - (self.ETA / tmp_PHO) * dLX
            d_sum = d_sum + LX / tmp_PHO

            print(i,"ERR_AVG",d_sum/(i+1))
            print(i,"X_DIA",X_mdiag_data[0:5])
            print(i,"V_I_TRUE_CNT",np.sum((I[I_violation_idx]+self.I[I_violation_idx])/self.I_max[I_violation_idx] > 1.))
            if LX<d_bst:
                d_bst = LX

            print(i,"BEST_ERR",d_bst)


    @profile
    def run_fc_exphalffc_plug(self, Z, num_iterations=None, exphalffc=None) -> bool:
        assert exphalffc is not None
        if num_iterations is None:
            num_iterations = self._get_niteration()

        d_sum = 0

        G = scipy.sparse.csr_matrix((self.K, self.K))
        for i in range(num_iterations):
            # compute X from G
            G_2 = G.copy()
            G_2.data = G_2.data/2.
            e_half = exphalffc(G_2.copy())

            X_mdiag_data = np.sum(e_half * e_half, axis=1)
            X_trace = np.sum(X_mdiag_data)/self.K
            X_mdiag_data = X_mdiag_data/X_trace
            X_mdiag = scipy.sparse.diags(X_mdiag_data).tocsr()
            X_mdiag.sort_indices()
            X_offdi_data = np.sum(e_half[self.H_x] * e_half[self.H_y], axis=1)/X_trace
            X_offdi = scipy.sparse.coo_matrix((X_offdi_data, (self.H_x, self.H_y)), shape=(self.K, self.K)).tocsr()
            X_offdi.sort_indices()

            X = X_offdi+X_mdiag

            # get L(X) and dX_L(X)
            # get LT(X) and dX_LT(X)
            T_violation_idx = (X_mdiag.data > 1.)
            T_violation_cnt = np.sum(T_violation_idx)
            T_violation_err = np.sum(X_mdiag.data[T_violation_idx] - 1.)
            T_dX_A = scipy.sparse.diags(T_violation_idx.astype(float)).tocsr()
            T_dX_I = scipy.sparse.diags(np.ones(self.K)*T_violation_cnt/self.K).tocsr()
            T_dX = T_dX_A - T_dX_I

            # get LI(X) and dX_LI(X)
            HX = X_offdi.copy()
            HX.data = ((X_offdi.data - 1.) * (Z - 1)) * self.H.data / Z
            I = np.asarray(HX.sum(axis=1)).ravel()
            I_max_I_sum = (self.I_max - (1.+ self.OFFSET)*self.I)
            I_violation_idx = (I > I_max_I_sum)
            I_violation_cnt = np.sum(I_violation_idx)
            I_violation_err = np.sum((I[I_violation_idx] - I_max_I_sum[I_violation_idx])/self.I[I_violation_idx])
            I_dX_A = csr_zero_rows_inplace(self.H.copy(),np.invert(I_violation_idx))
            I_dX_A = csr_scal_rows_inplace(I_dX_A,1./self.I)
            I_dX_A.data = I_dX_A.data*(Z-1.)/Z
            I_dX_A = I_dX_A + I_dX_A.transpose()
            I_dX_A.data = I_dX_A.data/2.
            I_dX_B = scipy.sparse.diags(np.ones(self.K)*np.sum((-self.I[I_violation_idx]*(Z-1)/Z - I_max_I_sum[I_violation_idx])/self.I[I_violation_idx])/self.K).tocsr()
            I_dX = I_dX_A+I_dX_B

            # get LX(X) and dX_LX(X)
            X_violation_idx = X_offdi.data < (-1. / (Z - 1.))
            X_violation_cnt = np.sum(X_violation_idx)
            X_violation_err = np.sum(X_offdi.data[X_violation_idx] * (1.-Z) - 1.)
            X_dX_A = X_offdi.copy()
            X_dX_A.data = np.zeros(X_dX_A.data.size)
            X_dX_A.data[X_violation_idx] = - (Z-1.)
            X_dX_A = X_dX_A + X_dX_A.transpose()
            X_dX_A.data = X_dX_A.data/2.
            X_dX_I = scipy.sparse.diags(np.ones(self.K)*X_violation_cnt/self.K).tocsr()
            X_dX = X_dX_A-X_dX_I

            sa_T, vh = scipy.sparse.linalg.eigsh(T_dX,k=1,which='LM')
            sa_I, vh = scipy.sparse.linalg.eigsh(I_dX,k=1,which='LM')
            sa_X, vh = scipy.sparse.linalg.eigsh(X_dX,k=1,which='LM')

            print(i,"ERR_T_I_X",T_violation_err,I_violation_err,X_violation_err)
            print(i,"ECT_T_I_X",T_violation_cnt,I_violation_cnt,X_violation_cnt)
            print(i,"V_I_TRUE_CNT",np.sum((I[I_violation_idx]+self.I[I_violation_idx])/self.I_max[I_violation_idx] > 1.))

            dLX = scipy.sparse.csr_matrix((self.K, self.K))
            LX = 0.
            if T_violation_err > 0:
                dLX = dLX + T_dX/np.abs(sa_T[0])
                LX += T_violation_err/np.abs(sa_T[0])

            if I_violation_err > 0:
                dLX = dLX + I_dX/np.abs(sa_I[0])
                LX += I_violation_err/np.abs(sa_I[0])

            if X_violation_err > 0:
                dLX = dLX + X_dX/np.abs(sa_X[0])
                LX += X_violation_err/np.abs(sa_X[0])

            sa, vh = scipy.sparse.linalg.eigsh(dLX,k=1,which='LM')
            tmp_PHO = np.abs(sa[0])


            # update G
            G = G  - (self.ETA / tmp_PHO) * dLX
            d_sum = d_sum + LX / tmp_PHO


            e_half = np.asarray(scipy.sparse.linalg.expm(G_2.tocsc()).todense())

            X_mdiag_data = np.sum(e_half * e_half, axis=1)
            X_trace = np.sum(X_mdiag_data)/self.K
            X_mdiag_data = X_mdiag_data/X_trace
            X_mdiag = scipy.sparse.diags(X_mdiag_data).tocsr()
            X_mdiag.sort_indices()
            X_offdi_data = np.sum(e_half[self.H_x] * e_half[self.H_y], axis=1)/X_trace
            X_offdi = scipy.sparse.coo_matrix((X_offdi_data, (self.H_x, self.H_y)), shape=(self.K, self.K)).tocsr()
            X_offdi.sort_indices()

            X = X_offdi+X_mdiag

            # get L(X) and dX_L(X)
            # get LT(X) and dX_LT(X)
            T_violation_idx = (X_mdiag.data > 1.)
            T_violation_cnt = np.sum(T_violation_idx)
            T_violation_err = np.sum(X_mdiag.data[T_violation_idx] - 1.)
            T_dX_A = scipy.sparse.diags(T_violation_idx.astype(float)).tocsr()
            T_dX_I = scipy.sparse.diags(np.ones(self.K)*T_violation_cnt/self.K).tocsr()
            T_dX = T_dX_A - T_dX_I

            # get LI(X) and dX_LI(X)
            HX = X_offdi.copy()
            HX.data = ((X_offdi.data - 1.) * (Z - 1)) * self.H.data / Z
            I = np.asarray(HX.sum(axis=1)).ravel()
            I_max_I_sum = (self.I_max - (1.+ self.OFFSET)*self.I)
            I_violation_idx = (I > I_max_I_sum)
            I_violation_cnt = np.sum(I_violation_idx)
            I_violation_err = np.sum((I[I_violation_idx] - I_max_I_sum[I_violation_idx])/self.I[I_violation_idx])
            I_dX_A = csr_zero_rows_inplace(self.H.copy(),np.invert(I_violation_idx))
            I_dX_A = csr_scal_rows_inplace(I_dX_A,1./self.I)
            I_dX_A.data = I_dX_A.data*(Z-1.)/Z
            I_dX_A = I_dX_A + I_dX_A.transpose()
            I_dX_A.data = I_dX_A.data/2.
            I_dX_B = scipy.sparse.diags(np.ones(self.K)*np.sum((-self.I[I_violation_idx]*(Z-1)/Z - I_max_I_sum[I_violation_idx])/self.I[I_violation_idx])/self.K).tocsr()
            I_dX = I_dX_A+I_dX_B

            # get LX(X) and dX_LX(X)
            X_violation_idx = X_offdi.data < (-1. / (Z - 1.))
            X_violation_cnt = np.sum(X_violation_idx)
            X_violation_err = np.sum(X_offdi.data[X_violation_idx] * (1.-Z) - 1.)
            X_dX_A = X_offdi.copy()
            X_dX_A.data = np.zeros(X_dX_A.data.size)
            X_dX_A.data[X_violation_idx] = - (Z-1.)
            X_dX_A = X_dX_A + X_dX_A.transpose()
            X_dX_A.data = X_dX_A.data/2.
            X_dX_I = scipy.sparse.diags(np.ones(self.K)*X_violation_cnt/self.K).tocsr()
            X_dX = X_dX_A-X_dX_I

            print(i,"ERR_T_I_X",T_violation_err,I_violation_err,X_violation_err)
            print(i,"ECT_T_I_X",T_violation_cnt,I_violation_cnt,X_violation_cnt)
            print(i,"V_I_TRUE_CNT",np.sum((I[I_violation_idx]+self.I[I_violation_idx])/self.I_max[I_violation_idx] > 1.))

    def set_st(self, state):
        self._process_state(state)
        self._compute_width_bound()

    # @profile
    def _process_state(self, state):
        ss = state.toarray()
        asso = np.argmax(ss, axis=1)
        self.s_max = np.max(ss, axis=1)
        self.I_max = self.s_max / self.min_sinr - 1.

        self.H = state[:,asso]
        self.H.setdiag(0)
        self.H.eliminate_zeros()
        self.H = self.H.transpose()
        self.H = self.H.tocsr()
        self.H.sort_indices()
        self.H_x, self.H_y = self.H.nonzero()

        self.I = np.asarray(self.H.sum(axis=1)).ravel()
        self.I_I_max = self.I/self.I_max

    def _compute_width_bound(self):
        self.width_bound = 100

    def _get_niteration(self):
        return self.width_bound ** 2


if __name__ == '__main__':
    from sim_src.env.env import env

    e = env()
    print(e.min_sinr)
    a = mmwm_scipy(e.n_sta, e.min_sinr)
    a.set_st(e.rxpr)
    a.run_fc(10)
