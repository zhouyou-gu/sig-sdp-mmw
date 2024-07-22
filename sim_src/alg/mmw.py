import math

import numpy as np
import scipy

from sim_src.alg.sdp_solver import sdp_solver
from sim_src.linalg_util import generate_rand_regular_simplex_with_Z_vertices
from sim_src.scipy_util import csr_scal_rows_inplace
from sim_src.util import STATS_OBJECT, profile


class mmw(STATS_OBJECT,sdp_solver):
    def __init__(self, nit=100, rank_radio=2, alpha=1., eta=0.1, log_gap=False):
        sdp_solver.__init__(self, nit=nit, rank_radio=rank_radio, alpha=alpha)
        self.eta = eta
        self.LOG_GAP = log_gap

    def run_with_state(self, bs_iteration, Z, state):
        tic = self._get_tic()
        ret = self._run(Z, state)
        tim = self._get_tim(tic)
        K = state[0].shape[0]
        self._add_np_log("mmw_all_it",bs_iteration,np.array([Z,K,tim]))
        return ret

    def _process_state(self, Z, S_gain, Q_asso, h_max):
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

        return S_gain_T_no_asso_no_diag, s_max, Q_asso, h_max, S_sum, norm_H

    # @profile
    def _run(self,Z,state):
        ### state process
        sp_tic = self._get_tic()
        K = state[0].shape[0]

        S_gain_T_no_asso_no_diag, s_max, Q_asso, h_max, S_sum, norm_H = self._process_state(Z,state[0],state[1],state[2])


        SSS = S_gain_T_no_asso_no_diag + S_gain_T_no_asso_no_diag.transpose()
        SSS = scipy.sparse.triu(SSS).tocsr()
        SSS.eliminate_zeros()
        SSS.sort_indices()
        nz_idx_gain_x_ut, nz_idx_gain_y_ut = SSS.nonzero()
        nz_idx_asso_x_ut, nz_idx_asso_y_ut = scipy.sparse.triu(Q_asso,k=1).tocsr().nonzero()

        E_asso = int(Q_asso.getnnz()/2)
        C = E_asso+2*K

        Y = np.ones(C)/C

        e_accu = np.zeros(C)
        L_accu = scipy.sparse.csr_matrix((K, K))

        X_mdiag = scipy.sparse.diags(np.ones(K)).tocsr()
        X_offdi = scipy.sparse.csr_matrix((K,K))
        tim = self._get_tim(sp_tic)
        self._add_np_log("mmw_state_process",0,np.array([Z,K,tim]))

        X_avgd = scipy.sparse.csr_matrix((K,K))
        Y_avgd = np.zeros(C)
        self.N_STEP = 0
        for i in range(self.nit):
            self.N_STEP += 1
            X_avgd += (X_offdi + X_mdiag)
            Y_avgd += Y
            if self.LOG_GAP:
                X_avgd_this =  X_avgd.copy()
                X_avgd_this.data =  X_avgd_this.data/(self.N_STEP)
                X_avgd_this_mdiag = X_avgd_this.diagonal()
                X_avgd_this.setdiag(0)
                X_avgd_this_offdi = X_avgd_this
                ## AD, X -> eD
                eD = (X_avgd_this_mdiag-1.)/(1.-1./K)
                ## AF, X -> eF
                XXX = np.asarray(X_avgd_this_offdi[nz_idx_asso_x_ut,nz_idx_asso_y_ut]).ravel()
                eF = (XXX+1./(Z-1))/(1./(K*(Z-1))+1./2.)
                ## AH, X -> eH
                AHX = S_gain_T_no_asso_no_diag*X_avgd_this_offdi
                eH = ((np.asarray(AHX.sum(axis=1)).ravel()*(Z-1)/Z) - (h_max-(1/Z * S_sum)))/norm_H
                ## eD, eF, eH, e_accu -> YD, YF, YH, e_accu
                e_this_max = np.max(np.hstack((eD,eF,eH)))

                ### compute L
                YY = Y_avgd.copy()/(self.N_STEP)
                YD = YY[0:K]
                YF = YY[K:K+E_asso]
                YH = YY[K+E_asso:2*K+E_asso]
                ## YD, AD -> LD
                LD = (scipy.sparse.diags(YD)-np.sum(YD)/K*scipy.sparse.diags(np.ones(K)))/(1.-1./K)
                ## YF, AF -> LF
                YF_m = scipy.sparse.coo_matrix((YF, (nz_idx_asso_x_ut, nz_idx_asso_y_ut)), shape=(K, K)).tocsr()
                YF_m = (YF_m + YF_m.transpose())/2.
                LF = (YF_m+np.sum(YF)/(K*(Z-1))*scipy.sparse.diags(np.ones(K)))/(1./2.+1./(K*(Z-1)))
                ## YH, AH -> LH
                LH = S_gain_T_no_asso_no_diag.copy()
                LH = csr_scal_rows_inplace(LH,YH)
                LH = csr_scal_rows_inplace(LH,1./norm_H)
                LH = (LH + LH.transpose())*(Z-1)/(2*Z)
                LH = LH - np.sum((1./K*h_max-1/(K*Z)*S_sum)*YH/norm_H)*scipy.sparse.diags(np.ones(K))

                ## LD, LF, LH, -> lamda_min_L
                s, v = scipy.sparse.linalg.eigsh((LD + LF + LH),k=1,which='SA')
                lamda_min_L = s[0]*K
                self._add_np_log("gap",i,np.array([e_this_max,lamda_min_L,e_this_max-lamda_min_L]))

            tic_per_it = self._get_tic()
            X_print = X_mdiag + X_offdi
            Y_print = scipy.sparse.coo_matrix((Y[K:K+E_asso], (nz_idx_asso_x_ut, nz_idx_asso_y_ut)), shape=(K, K)).tocsr()
            Y_print = Y_print + scipy.sparse.diags(Y[K+E_asso:2*K+E_asso])
            if True:
                ### compute dual
                tic_dual = self._get_tic()
                ## AD, X -> eD
                eD = (X_mdiag.data-1.)/(1.-1./K)

                ## AF, X -> eF
                XXX = np.asarray(X_offdi[nz_idx_asso_x_ut,nz_idx_asso_y_ut]).ravel()
                eF = (XXX+1./(Z-1))/(1./(K*(Z-1))+1./2.)
                ## AH, X -> eH
                AHX = S_gain_T_no_asso_no_diag*X_offdi
                eH = ((np.asarray(AHX.sum(axis=1)).ravel()*(Z-1)/Z) - (h_max-(1/Z * S_sum)))/norm_H
                ## eD, eF, eH, e_accu -> YD, YF, YH, e_accu
                e_this = np.hstack((eD,eF,eH))
                e_accu += e_this*self.eta

                Y = scipy.special.softmax(e_accu)

                tim = self._get_tim(tic_dual)
                self._add_np_log("mmw_dual",i,np.array([Z,K,tim]))

            if True:
                ### compute L
                tic_loss = self._get_tic()
                YY = Y.copy()
                YD = YY[0:K]
                YF = YY[K:K+E_asso]
                YH = YY[K+E_asso:2*K+E_asso]
                ## YD, AD -> LD
                LD = (scipy.sparse.diags(YD)-np.sum(YD)/K*scipy.sparse.diags(np.ones(K)))/(1.-1./K)

                ## YF, AF -> LF
                YF_m = scipy.sparse.coo_matrix((YF, (nz_idx_asso_x_ut, nz_idx_asso_y_ut)), shape=(K, K)).tocsr()
                YF_m = (YF_m + YF_m.transpose())/2.
                LF = (YF_m+np.sum(YF)/(K*(Z-1))*scipy.sparse.diags(np.ones(K)))/(1./2.+1./(K*(Z-1)))

                ## YH, AH -> LH
                LH = S_gain_T_no_asso_no_diag.copy()
                LH = csr_scal_rows_inplace(LH,YH)
                LH = csr_scal_rows_inplace(LH,1./norm_H)
                LH = (LH + LH.transpose())*(Z-1)/(2*Z)
                LH = LH - np.sum((1./K*h_max-1/(K*Z)*S_sum)*YH/norm_H)*scipy.sparse.diags(np.ones(K))

                ## LD, LF, LH, L_accu -> L_accu
                L_accu = L_accu - (LD + LF + LH)*self.eta

                tim = self._get_tim(tic_loss)
                self._add_np_log("mmw_loss",i,np.array([Z,K,tim]))

            if True:
                ### compute X
                tic_expm = self._get_tic()

                ## L_accu -> X_half, X

                L_half = L_accu.copy()
                L_half.data = L_half.data/2.
                X_half = mmw.expm_half_randsk(L_half.copy(),Z*self.rank_radio)
                # X_half = X_half/np.linalg.norm(X_half, axis=1, keepdims=True)
                X_mdiag_data = np.sum(X_half * X_half, axis=1)
                X_trace = np.sum(X_mdiag_data)/K
                X_mdiag_data = X_mdiag_data/X_trace
                X_mdiag = scipy.sparse.diags(X_mdiag_data).tocsr()
                X_mdiag.sort_indices()
                X_offdi_data = np.sum(X_half[nz_idx_gain_x_ut] * X_half[nz_idx_gain_y_ut], axis=1)/X_trace
                X_offdi_S = scipy.sparse.coo_matrix((X_offdi_data, (nz_idx_gain_x_ut,nz_idx_gain_y_ut)), shape=(K, K)).tocsr()
                X_offdi_data = np.sum(X_half[nz_idx_asso_x_ut] * X_half[nz_idx_asso_y_ut], axis=1)/X_trace
                X_offdi_Q = scipy.sparse.coo_matrix((X_offdi_data, (nz_idx_asso_x_ut,nz_idx_asso_y_ut)), shape=(K, K)).tocsr()
                X_offdi = X_offdi_S + X_offdi_Q
                X_offdi = X_offdi + X_offdi.transpose()
                X_offdi.eliminate_zeros()
                X_offdi.sort_indices()

                tim = self._get_tim(tic_expm)
                self._add_np_log("mmw_expm",i,np.array([Z,K,tim]))

            tim = self._get_tim(tic_per_it)
            self._add_np_log("mmw_per_it",i,np.array([Z,K,tim]))

        tic_xavg = self._get_tic()
        X_avgd.data = X_avgd.data/(self.nit)
        s, v = scipy.sparse.linalg.eigsh(-X_avgd,k=1,which='LA')
        self._print("SS###############\n",S_gain_T_no_asso_no_diag[nz_idx_gain_x_ut[0:self.PRINT_DIM],nz_idx_gain_y_ut[0:self.PRINT_DIM]],1/(Z-1))
        self._print("XAVG_nz_S_idx###########\n",np.vstack((nz_idx_gain_x_ut[0:self.PRINT_DIM],nz_idx_gain_y_ut[0:self.PRINT_DIM])),1/(Z-1))
        self._print("XAVG_nz_S###############\n",X_avgd[nz_idx_gain_x_ut[0:self.PRINT_DIM],nz_idx_gain_y_ut[0:self.PRINT_DIM]],1/(Z-1))
        self._print("XAVG_nz_Q_idx###########\n",np.vstack((nz_idx_asso_x_ut[0:self.PRINT_DIM],nz_idx_asso_y_ut[0:self.PRINT_DIM])),1/(Z-1))
        self._print("XAVG_nz_Q###############\n",X_avgd[nz_idx_asso_x_ut[0:self.PRINT_DIM],nz_idx_asso_y_ut[0:self.PRINT_DIM]],1/(Z-1))
        self._print("XAVG_nz_Sort############\n",np.sort(np.asarray(X_avgd[nz_idx_gain_x_ut,nz_idx_gain_y_ut]).flatten())[:10])
        self._print("XAVG_lam_min############\n",-s)

        rank = np.min([K-1 , (Z-1)*self.rank_radio])
        # X_avgd[nz_idx_asso_x_ut,nz_idx_asso_y_ut] = 0.
        u, s, vT = scipy.sparse.linalg.svds(X_avgd, k=rank)
        X_half = np.matmul(u, np.diag(np.sqrt(s)))
        # X_half = X_half/np.linalg.norm(X_half, axis=1, keepdims=True)
        XX = np.matmul(X_half[0:self.PRINT_DIM],X_half[0:self.PRINT_DIM].transpose())
        self._print("X_half_ret############\n",XX[0:self.PRINT_DIM,0:self.PRINT_DIM])
        tim = self._get_tim(tic_xavg)
        self._add_np_log("mmw_xavg",0,np.array([Z,K,tim]))
        return True, X_half

    @staticmethod
    def expm_half_randsk(L,D):
        randv = np.random.randn(L.shape[0],D)/math.sqrt(float(D))
        randv = randv/np.linalg.norm(randv,axis=1)[:,None]
        ret = scipy.sparse.linalg.expm_multiply(L.copy(),randv)
        return ret


if __name__ == '__main__':
    row = np.array([0, 0, 1, 2, 2, 2])
    col = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([1, 2, 3, 4, 5, 6])
    a = scipy.sparse.csr_matrix((data, (row, col)), shape=(3, 3)).toarray()
    print(a[row,col])

    e = scipy.sparse.csr_matrix((10,10))
    v = np.random.randn(10)
    print(scipy.sparse.linalg.expm_multiply(e,v),v)