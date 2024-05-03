import math

import numpy as np
import scipy

from sim_src.alg.sdp_solver import sdp_solver
from sim_src.scipy_util import csr_scal_rows_inplace
from sim_src.util import STATS_OBJECT, profile


class mmw(STATS_OBJECT,sdp_solver):
    def __init__(self, nit=100, D=5, alpha=1., eta=0.1):
        self.nit = nit
        self.D = D
        self.alpha = alpha
        self.eta = eta

    def run_with_state(self, iteration, Z, state):
        tic = self._get_tic()
        ret = self._run(Z, state)
        tim = self._get_tim(tic)
        K = state[0].shape[0]
        self._add_np_log("mmw_all_it",iteration,np.array([Z,K,tim]))
        return ret

    def _process_state(self, Z, S_gain, Q_asso, h_max):
        K = S_gain.shape[0]
        S_gain_T_no_diag = S_gain.copy().transpose()
        S_gain_T_no_diag.setdiag(0)
        S_sum = np.asarray(S_gain_T_no_diag.sum(axis=1)).ravel()

        s_max = S_gain.diagonal()
        S_gain_T_no_diag_square = S_gain_T_no_diag.copy()
        S_gain_T_no_diag_square.data = S_gain_T_no_diag_square.data ** 2
        norm_H = np.sqrt(np.asarray(S_gain_T_no_diag_square.sum(axis=1)).ravel()) * (Z-1)/(2*Z) + np.abs(1/K*h_max-1/K/Z*S_sum)

        return S_gain_T_no_diag, s_max, Q_asso, h_max, S_sum, norm_H

    @profile
    def _run(self,Z,state):
        ### state process
        sp_tic = self._get_tic()
        K = state[0].shape[0]

        S_gain_T_no_diag, s_max, Q_asso, h_max, S_sum, norm_H = self._process_state(Z,state[0],state[1],state[2])


        SSS = S_gain_T_no_diag + S_gain_T_no_diag.transpose()
        SSS = scipy.sparse.triu(SSS).tocsr()
        SSS.eliminate_zeros()
        SSS.sort_indices()
        nz_idx_gain_x_ut, nz_idx_gain_y_ut = SSS.nonzero()
        nz_idx_asso_x, nz_idx_asso_y = scipy.sparse.triu(Q_asso,k=1).tocsr().nonzero()

        E_asso = int(Q_asso.getnnz()/2)
        C = E_asso+2*K

        YD = np.ones(K)/float(C)
        YF = np.ones(E_asso)/float(C)
        YH = np.ones(K)/float(C)

        e_accu = np.zeros(C)
        L_accu = scipy.sparse.csr_matrix((K, K))
        e_weighted = 0.
        tim = self._get_tim(sp_tic)
        self._add_np_log("mmw_state_process",0,np.array([Z,K,tim]))

        X_mdiag_avg_data = np.zeros(K)
        X_offdi_avg_data = np.zeros(nz_idx_gain_x_ut.size)

        for i in range(self.nit):

            tic_per_it = self._get_tic()

            ### compute L
            tic_loss = self._get_tic()

            ## YD, AD -> LD
            LD = (scipy.sparse.diags(YD)-np.sum(YD)/K*scipy.sparse.diags(np.ones(K)))/(1.-1./K)

            ## YF, AF -> LF
            YF_m = scipy.sparse.coo_matrix((YF, (nz_idx_asso_x, nz_idx_asso_y)), shape=(K, K)).tocsr()
            YF_m = (YF_m + YF_m.transpose())/2.
            LF = (YF_m+np.sum(YF)/(K*(Z-1))*scipy.sparse.diags(np.ones(K)))/(1./2.+1./(K*(Z-1)))

            ## YH, AH -> LH
            LH = S_gain_T_no_diag.copy()
            LH = csr_scal_rows_inplace(LH,YH)
            LH = csr_scal_rows_inplace(LH,1./norm_H)
            LH = (LH + LH.transpose())*(Z-1)/(2*Z)
            LH = LH - np.sum((1./K*h_max-1/(K*Z)*S_sum)*YH/norm_H)*scipy.sparse.diags(np.ones(K))

            ## LD, LF, LH, L_accu -> L_accu
            L_accu = L_accu - (LD + LF + LH)*self.eta

            tim = self._get_tim(tic_loss)
            self._add_np_log("mmw_loss",i,np.array([Z,K,tim]))

            ### compute X
            tic_expm = self._get_tic()

            ## L_accu -> X_half, X

            L_half = L_accu.copy()
            L_half.data = L_half.data/2.
            X_half = mmw.expm_half_randsk(L_half.copy(),self.D)
            X_mdiag_data = np.sum(X_half * X_half, axis=1)
            X_trace = np.sum(X_mdiag_data)/K
            X_mdiag_data = X_mdiag_data/X_trace
            X_mdiag = scipy.sparse.diags(X_mdiag_data).tocsr()
            X_mdiag.sort_indices()
            X_offdi_data = np.sum(X_half[nz_idx_gain_x_ut] * X_half[nz_idx_gain_y_ut], axis=1)/X_trace
            X_offdi = scipy.sparse.coo_matrix((X_offdi_data, (nz_idx_gain_x_ut,nz_idx_gain_y_ut)), shape=(K, K)).tocsr()
            X_offdi = X_offdi + X_offdi.transpose()
            X_offdi.eliminate_zeros()
            X_offdi.sort_indices()

            tim = self._get_tim(tic_expm)
            self._add_np_log("mmw_expm",i,np.array([Z,K,tim]))


            ### compute dual
            tic_dual = self._get_tic()
            ## AD, X -> eD
            eD = (X_mdiag.data-1.)/(1.-1./K)

            ## AF, X -> eF
            XXX = np.asarray(X_offdi[nz_idx_asso_x,nz_idx_asso_y]).ravel()
            eF = (XXX+1./(Z-1))/(1./(K*(Z-1))+1./2.)
            ## AH, X -> eH
            AHX = S_gain_T_no_diag*X_offdi
            eH = ((np.asarray(AHX.sum(axis=1)).ravel()*(Z-1)/Z) - (h_max-(1/Z * S_sum)))/norm_H


            ## eD, eF, eH, e_accu -> YD, YF, YH, e_accu
            e_accu[0:K] += eD*self.eta
            e_accu[K:K+E_asso] += eF*self.eta
            e_accu[K+E_asso:2*K+E_asso] += eH*self.eta

            e_weighted += np.sum(eD*YD) + np.sum(eF*YF) + np.sum(eH*YH)

            Y = scipy.special.softmax(e_accu)
            YD = Y[0:K]
            YF = Y[K:K+E_asso]
            YH = Y[K+E_asso:2*K+E_asso]

            tim = self._get_tim(tic_dual)
            self._add_np_log("mmw_dual",i,np.array([Z,K,tim]))

            tim = self._get_tim(tic_per_it)
            self._add_np_log("mmw_per_it",i,np.array([Z,K,tim]))


        return True, X_half/np.linalg.norm(X_half,axis=1,keepdims=True)

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