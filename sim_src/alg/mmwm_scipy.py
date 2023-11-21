import math

import numpy as np
import scipy
from pylanczos import PyLanczos

from sim_src.scipy_util import *
from sim_src.util import profile

from sim_src.alg.interface import feasibility_check_alg


class mmwm_scipy(feasibility_check_alg):
    ETA = 0.1

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
    def run_fc(self, Z, num_iterations=None) -> bool:
        if num_iterations is None:
            num_iterations = self._get_niteration()

        ## X(1) is a identity matrix
        I_v = self.I / self.I_max
        notidx = I_v <= Z
        idx = np.invert(notidx)
        C = self.H.copy()
        csr_zero_rows_inplace(C, notidx)
        csr_scal_cons_inplace(C, 0.5)
        CC = C + C.transpose()
        id = scipy.sparse.identity(self.K)

        I_all = np.sum(I_v[idx]) / (Z * self.K)
        id.data = id.data * I_all
        CC = CC + id

        L_sum = scipy.sparse.csr_matrix((self.K, self.K))
        L_sum += CC
        d_sum = 0
        d_best = I_all * self.K - idx.sum()
        d_sum += d_best
        X_best = id
        X_next = id.copy()
        for i in range(num_iterations):

            e_half = csr_expm_rank_dsketch(csr_scal_cons_inplace(CC, -0.5 * self.ETA), K=self.K, d=Z, r=100)

            trace = np.sum(np.power(e_half, 2))
            print(trace, '+++++++++++')
            csr_scal_cons_inplace(e_half, np.sqrt(self.K / trace))

            X_trace = np.sum(e_half * e_half, axis=1)
            X_trace = scipy.sparse.diags(X_trace).tocsr()
            tmp_idx = (X_trace.data > 1.)
            C_T_e = np.sum(X_trace.data[tmp_idx] - 1.)
            tmp_cnt = tmp_idx.sum()
            tmp_data = -np.ones(self.K) * float(tmp_cnt) / float(self.K)
            X_trace.data = tmp_data
            X_trace.data[tmp_idx] += 1.
            C_T = X_trace

            X_offdi_selected = np.sum(e_half[self.H_x] * e_half[self.H_y], axis=1)
            X_offdi_selected = scipy.sparse.coo_matrix((X_offdi_selected, (self.H_x, self.H_y)), shape=(self.K, self.K))
            X_offdi_selected = X_offdi_selected.tocsr()
            X_offdi_selected.sort_indices()

            X_offdi_group = X_offdi_selected.copy()
            tmp_idx = X_offdi_group.data < (-1. / (Z - 1.))
            C_X_e = np.sum(X_offdi_group.data[tmp_idx] * (-(Z - 1.)) - 1.)
            tmp_cnt = tmp_idx.sum()
            tmp_data = np.zeros(X_offdi_group.getnnz())
            tmp_data[tmp_idx] = (-(Z - 1.))
            X_offdi_group.data = tmp_data
            X_offdi_group.eliminate_zeros()
            X_offdi_group = X_offdi_group + X_offdi_group.transpose()
            C_X = X_offdi_group - scipy.sparse.diags(np.ones(self.K) * tmp_cnt / self.K).tocsr()
            X_offdi_inter = X_offdi_selected.copy()
            X_offdi_inter.data = X_offdi_inter.data * self.H.data * (Z - 1)
            H = self.H.copy()
            I_IX = (np.asarray(X_offdi_inter.sum(axis=1)).ravel() + self.I) / self.I_max
            tmp_idx = I_IX > Z
            C_I_e = np.sum(I_IX[tmp_idx] / Z - 1.)
            tmp_cnt = tmp_idx.sum()
            tmp_I = np.sum(self.I[tmp_idx] / self.I_max[tmp_idx])
            tmp_data = np.ones(self.K) * (-tmp_cnt + tmp_I / Z) / self.K
            csr_zero_rows_inplace(H, np.invert(tmp_idx))
            H.data = H.data * (Z - 1.) / Z
            H = H + H.transpose()
            C_I = H + scipy.sparse.diags(tmp_data).tocsr()

            CC = C_T + C_X + C_I
            # print(C_X)
            L_sum += CC
            CC = L_sum.copy()
            d_total = C_T_e + C_X_e + C_I_e
            if d_total < d_best:
                X_best = X_offdi_selected
                d_best = d_total
            print(d_total, C_T_e, C_X_e, C_I_e)
            # print(CC)

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
            e_half = np.asarray(scipy.sparse.linalg.expm(G_2).todense())

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
            HX.data = (X_offdi.data * (Z - 1) + 1.) * self.H.data / Z
            I = np.asarray(HX.sum(axis=1)).ravel() / self.I_max
            I_violation_idx = (I > 1.)
            I_violation_cnt = np.sum(I_violation_idx)
            I_violation_err = np.sum(I[I_violation_idx] - 1.)
            I_dX_A = csr_zero_rows_inplace(self.H.copy(),np.invert(I_violation_idx))
            I_dX_A = csr_scal_rows_inplace(I_dX_A,1./self.I_max)
            I_dX_A.data = I_dX_A.data*(Z-1.)/Z
            I_dX_A = I_dX_A + I_dX_A.transpose()
            I_dX_A.data = I_dX_A.data/2.
            I_dX_B = scipy.sparse.diags(np.ones(self.K)*np.sum(self.I_I_max[I_violation_idx]/Z/self.K)).tocsr()
            I_dX_I = scipy.sparse.diags(np.ones(self.K)*I_violation_cnt/self.K).tocsr()
            I_dX = I_dX_A+I_dX_B-I_dX_I

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

            print("_______________",sa_T[0],sa_I[0],sa_X[0])
            print("+++++++++++++++",T_violation_err,I_violation_err,X_violation_err)

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
            dLX.data = dLX.data/tmp_PHO


            print(LX,tmp_PHO,np.mean(np.abs(dLX.diagonal())))
            # print(LX,tmp_PHO,np.mean(np.abs(dLX.diagonal())))
            # dLX = T_dX + I_dX + X_dX
            # LX = T_violation_err + I_violation_err + X_violation_err

            # sa, vh = scipy.sparse.linalg.eigsh(dLX,k=1,which='LM')
            # tmp_PHO = np.abs(sa[0])
            # dLX.data = dLX.data/tmp_PHO

            # np.sum(np.asarray(dLX.diagonal().todense()))

            # update G
            G = G  - self.ETA * dLX
            d_sum = d_sum + LX / tmp_PHO
            X_all = e_half @ e_half.T
            print(X_all[0:5,0:5]/X_trace)
            print("$$$$$$",np.min(X_all),np.max(np.triu(X_all,k=1)))

            if LX<d_bst or i % 50 == 0:
                sa, vh = scipy.sparse.linalg.eigsh(dLX,k=1,which='LM')
                print("++++++",sa[0],tmp_PHO)
                print(i,LX,LX/tmp_PHO/self.K,T_violation_err,I_violation_err,X_violation_err)
                print(i,self.K,self.H.nnz,T_violation_cnt,I_violation_cnt,X_violation_cnt)
                print(i,d_sum/i)
                X = X.todense()
                print(X[0:5,0:5])
                print(I[I_violation_idx])
                print(self.I_I_max[I_violation_idx]/I[I_violation_idx])

                if LX<d_bst:
                    d_bst = LX
    def run_fc_b(self, Z, num_iterations=None) -> bool:
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
            e_half = np.asarray(scipy.sparse.linalg.expm(G_2).todense())

            randv = np.random.randn(self.K,int(Z)*100)
            # randv = randv.T/np.linalg.norm(randv,axis=1)
            e_half = e_half @ randv
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
            HX.data = (X_offdi.data * (Z - 1) + 1.) * self.H.data / Z
            I = np.asarray(HX.sum(axis=1)).ravel() / self.I_max
            I_violation_idx = (I > 1.)
            I_violation_cnt = np.sum(I_violation_idx)
            I_violation_err = np.sum(I[I_violation_idx] - 1.)
            I_dX_A = csr_zero_rows_inplace(self.H.copy(),np.invert(I_violation_idx))
            I_dX_A = csr_scal_rows_inplace(I_dX_A,1./self.I_max)
            I_dX_A.data = I_dX_A.data*(Z-1.)/Z
            I_dX_A = I_dX_A + I_dX_A.transpose()
            I_dX_A.data = I_dX_A.data/2.
            I_dX_B = scipy.sparse.diags(np.ones(self.K)*np.sum(self.I_I_max[I_violation_idx]/Z/self.K)).tocsr()
            I_dX_I = scipy.sparse.diags(np.ones(self.K)*I_violation_cnt/self.K).tocsr()
            I_dX = I_dX_A+I_dX_B-I_dX_I

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

            dLX = T_dX + I_dX + X_dX
            LX = T_violation_err + I_violation_err + X_violation_err

            sa, vh = scipy.sparse.linalg.eigsh(dLX,k=1,which='LM')
            tmp_PHO = np.abs(sa[0])
            dLX.data = dLX.data/tmp_PHO


            # update G
            G = G  - self.ETA * dLX
            d_sum = d_sum + LX / tmp_PHO
            if LX<d_bst or i % 50 == 0:
                sa, vh = scipy.sparse.linalg.eigsh(dLX,k=1,which='LM')
                print("++++++",sa[0],tmp_PHO)
                print(i,LX,LX/tmp_PHO/self.K,T_violation_err,I_violation_err,X_violation_err)
                print(i,self.K,self.H.nnz,T_violation_cnt,I_violation_cnt,X_violation_cnt)
                print(i,d_sum/i)
                X = X.todense()
                print(X[0:5,0:5])
                print(I[I_violation_idx])
                print(self.I_I_max[I_violation_idx]/I[I_violation_idx])
                X_all = e_half @ e_half.T
                print(X_all[0:5,0:5]/X_trace)
                if LX<d_bst:
                    d_bst = LX
    def run_fc_a(self, Z, num_iterations=None) -> bool:
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
            e_half = np.asarray(scipy.sparse.linalg.expm(G_2).todense())
            print("??????",np.linalg.matrix_rank(e_half))
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
            T_dX.data = T_dX.data
            T_violation_err = T_violation_err

            sa_T, vh = scipy.sparse.linalg.eigsh(T_dX,k=1,which='LM')


            # get LI(X) and dX_LI(X)
            H = self.H.copy()
            DHX = np.asarray(H @ X_mdiag_data).ravel() + np.asarray(H.sum(axis=1)).ravel() * X_mdiag_data
            HX = X_offdi.copy()
            HX.data = - X_offdi.data * self.H.data
            FHX = np.asarray(HX.sum(axis=1)).ravel() * 2
            LapX = (DHX+FHX) * (Z-1.)/Z/2.
            Is_Im = self.I - self.I_max
            LapX = LapX
            I_violation_idx = (LapX < Is_Im)
            I_violation_cnt = np.sum(I_violation_idx)
            I_violation_err = np.sum(Is_Im[I_violation_idx]-LapX[I_violation_idx])

            I_dX_A = csr_zero_rows_inplace(self.H.copy(),np.invert(I_violation_idx))
            # I_dX_A = csr_scal_rows_inplace(I_dX_A,1./Is_Im)
            I_dX_A.data = I_dX_A.data*(Z-1.)/Z/2.
            I_dX_A = I_dX_A + I_dX_A.transpose()

            I_dX_B = csr_zero_rows_inplace(self.H.copy(),np.invert(I_violation_idx))
            I_dX_B = I_dX_B.transpose()
            I_dX_B = np.asarray(I_dX_B.sum(axis=1)).ravel()
            I_dX_C = csr_zero_rows_inplace(self.H.copy(),np.invert(I_violation_idx))
            # I_dX_C = csr_scal_rows_inplace(I_dX_C,1./Is_Im)
            I_dX_C = np.asarray(I_dX_C.sum(axis=1)).ravel()

            I_dX_T = -(I_dX_B + I_dX_C)*(Z-1.)/Z/2.
            I_dX_T = scipy.sparse.diags(I_dX_T).tocsr()
            I_dX_I = scipy.sparse.diags(-np.zeros(self.K)*np.sum(Is_Im[I_violation_idx])/self.K).tocsr()
            I_dX = I_dX_A+I_dX_T-I_dX_I


            sa_I, vh = scipy.sparse.linalg.eigsh(I_dX,k=1,which='LM')


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

            sa_X, vh = scipy.sparse.linalg.eigsh(X_dX,k=1,which='LM')

            print("_______________",sa_T[0],sa_I[0],sa_X[0])
            print("+++++++++++++++",T_violation_err,I_violation_err,X_violation_err)

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
            dLX.data = dLX.data/tmp_PHO


            # update G
            G = G  - self.ETA * dLX
            d_sum = d_sum + LX / tmp_PHO
            if LX<d_bst or i % 50 == 0:
                sa, vh = scipy.sparse.linalg.eigsh(dLX,k=1,which='LM')
                print("++++++",sa[0],tmp_PHO)
                print(i,LX,LX/tmp_PHO/self.K,T_violation_err,I_violation_err,X_violation_err)
                print(i,self.K,self.H.nnz,T_violation_cnt,I_violation_cnt,X_violation_cnt)
                print(i,d_sum/i)
                X = X.todense()
                print(X[0:5,0:5])
                print(LapX[I_violation_idx])
                print((LapX[I_violation_idx]-self.I[I_violation_idx])/self.I_max[I_violation_idx])
                X_all = e_half @ e_half.T
                print(X_all[0:5,0:5]/X_trace)
                if LX<d_bst:
                    d_bst = LX
    def run_fc_z(self, Z, num_iterations=None) -> bool:
        if num_iterations is None:
           num_iterations = self._get_niteration()

        d_sum = 0
        d_bst = np.infty
        X_bst = None

        G = scipy.sparse.csr_matrix((self.K, self.K))
        RND = scipy.sparse.csr_matrix((self.K, self.K))
        for i in range(num_iterations):
        # compute X from G
            G_2 = G.copy()
            G_2.data = G_2.data/2.
            e_half = np.asarray(scipy.sparse.linalg.expm(G_2).todense())
            print("??????",np.linalg.matrix_rank(e_half))

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
            T_dX.data = T_dX.data
            T_violation_err = T_violation_err

            sa_T, vh = scipy.sparse.linalg.eigsh(T_dX,k=1,which='LM')


            # get LI(X) and dX_LI(X)
            H = self.H.copy()
            DHX = np.asarray(H @ X_mdiag_data).ravel() + np.asarray(H.sum(axis=1)).ravel() * X_mdiag_data
            HX = X_offdi.copy()
            HX.data = - X_offdi.data * self.H.data
            FHX = np.asarray(HX.sum(axis=1)).ravel() * 2
            LapX = (DHX+FHX) * (Z-1.)/Z/2.
            Is_Im = self.I - self.I_max
            LapX = LapX
            I_violation_idx = (LapX < Is_Im)
            I_violation_cnt = np.sum(I_violation_idx)
            I_violation_err = np.sum(Is_Im[I_violation_idx]-LapX[I_violation_idx])

            I_dX_A = csr_zero_rows_inplace(self.H.copy(),np.invert(I_violation_idx))
            # I_dX_A = csr_scal_rows_inplace(I_dX_A,1./Is_Im)
            I_dX_A.data = I_dX_A.data*(Z-1.)/Z/2.
            I_dX_A = I_dX_A + I_dX_A.transpose()

            I_dX_B = csr_zero_rows_inplace(self.H.copy(),np.invert(I_violation_idx))
            I_dX_B = I_dX_B.transpose()
            I_dX_B = np.asarray(I_dX_B.sum(axis=1)).ravel()
            I_dX_C = csr_zero_rows_inplace(self.H.copy(),np.invert(I_violation_idx))
            # I_dX_C = csr_scal_rows_inplace(I_dX_C,1./Is_Im)
            I_dX_C = np.asarray(I_dX_C.sum(axis=1)).ravel()

            I_dX_T = -(I_dX_B + I_dX_C)*(Z-1.)/Z/2.
            I_dX_T = scipy.sparse.diags(I_dX_T).tocsr()
            I_dX_I = scipy.sparse.diags(-np.zeros(self.K)*np.sum(Is_Im[I_violation_idx])/self.K).tocsr()
            I_dX = I_dX_A+I_dX_T-I_dX_I


            sa_I, vh = scipy.sparse.linalg.eigsh(I_dX,k=1,which='LM')


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

            sa_X, vh = scipy.sparse.linalg.eigsh(X_dX,k=1,which='LM')

            print("_______________",sa_T[0],sa_I[0],sa_X[0])
            print("+++++++++++++++",T_violation_err,I_violation_err,X_violation_err)

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
            dLX.data = dLX.data/tmp_PHO



            # update G
            G = G  - self.ETA * dLX
            d_sum = d_sum + LX / tmp_PHO

            if LX<d_bst or i % 50 == 0:
                sa, vh = scipy.sparse.linalg.eigsh(dLX,k=1,which='LM')
                print("++++++",sa[0],tmp_PHO)
                print(i,LX,LX/tmp_PHO/self.K,T_violation_err,I_violation_err,X_violation_err)
                print(i,self.K,self.H.nnz,T_violation_cnt,I_violation_cnt,X_violation_cnt)
                print(i,d_sum/i)
                X = X.todense()
                print(X[0:5,0:5])
                print(LapX[I_violation_idx])
                print((LapX[I_violation_idx]-self.I[I_violation_idx])/self.I_max[I_violation_idx])
                X_all = e_half @ e_half.T
                print(X_all[0:5,0:5]/X_trace)
                if LX<d_bst:
                    d_bst = LX
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
