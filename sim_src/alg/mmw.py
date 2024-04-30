import math

import numpy as np
import scipy

from sim_src.scipy_util import csr_scal_rows_inplace


class mmw:
    def __init__(self, nit=100, D=10, alpha=1., eta=0.1):
        self.nit = nit
        self.D = D
        self.alpha = alpha
        self.eta = eta

    def run_with_state(self, Z, state):
        return self._run(Z, state)

    def rounding(self, Z, gX):
        pass
    def _process_state(self, Z, S_gain, Q_asso, h_max):
        K = S_gain.shape[0]
        S_gain_T_no_diag = S_gain.copy().transponse().setdiag(0)
        S_sum = S_gain_T_no_diag.sum(1)

        s_max = S_gain.diagonal()
        S_gain_T_no_diag.data = S_gain_T_no_diag.data ** 2
        norm_H = S_gain_T_no_diag.sum(1) ** (1/2) * (Z-1)/(2*Z) + np.abs(1/K*h_max-1/K/Z*S_sum)

        return S_gain_T_no_diag, s_max, Q_asso, h_max, S_sum, norm_H

    def _run(self,Z,state):
        K = state.shape[0]

        S_gain_T_no_diag, s_max, Q_asso, h_max, S_sum, norm_H = self._process_state(Z,state[0],state[1],state[0])

        nz_idx_gain_x, nz_idx_gain_y = S_gain_T_no_diag.nonzero()
        nz_idx_asso_x, nz_idx_asso_y = Q_asso.nonzero()

        E_asso = Q_asso.getnnz()/2
        C = E_asso+2*K

        YD = np.ones(K)/float(C)
        YF = np.ones(E_asso)/float(C)
        YH = np.ones(E_asso)/float(C)

        e_accu = np.zeros(C)
        L_accu = scipy.sparse.csr_matrix((K, K))

        for i in range(self.nit):


            ## YD, AD -> LD
            LD = (scipy.sparse.diags(YD)-np.sum(YD)/K*scipy.sparse.diags(np.ones(K)))/(1.-1./K)

            ## YF, AF -> LF
            YF_m = scipy.sparse.coo_matrix((YF, (nz_idx_asso_x, nz_idx_asso_y)), shape=(K, K)).tocsr()
            YF_m = YF_m + YF_m.transpose()
            LF = (YF_m-np.sum(YF)/(K*(Z-1))*scipy.sparse.diags(np.ones(K)))/(1./2.+1./(K(Z-1)))

            ## YH, AH -> LH
            LH = S_gain_T_no_diag.copy()
            LH = csr_scal_rows_inplace(LH,YH)
            LH = LH + LH.transpose()
            LH.data = LH.data*(Z-1)/(2*Z)
            ## todo LH


            ## LD, LF, LH, L_accu -> L_accu
            L_accu = L_accu + (LD + LF + LH)*self.eta


            ## L_accu -> X_half, X
            L_half = L_accu.copy()
            L_half.data = L_half.data/2.
            X_half = mmw.expm_half_randsk(L_half.copy(),self.D)
            X_mdiag_data = np.sum(X_half * X_half, axis=1)
            X_trace = np.sum(X_mdiag_data)/K
            X_mdiag_data = X_mdiag_data/X_trace
            X_mdiag = scipy.sparse.diags(X_mdiag_data).tocsr()
            X_mdiag.sort_indices()
            X_offdi_data = np.sum(X_half[self.H_x] * X_half[self.H_y], axis=1)/X_trace
            X_offdi = scipy.sparse.coo_matrix((X_offdi_data, (self.H_x, self.H_y)), shape=(self.K, self.K)).tocsr()
            X_offdi.sort_indices()


            ## AD, X -> eD
            eD = (X_mdiag-1.)/(1.-1./K)

            ## AF, X -> eF
            eF = (X_offdi+1./(Z-1))/(1./(K*(Z-1))+1./2.)

            ## AH, X -> eH
            eH = X_offdi

            ## eD, eF, eH, e_accu -> YD, YF, YH, e_accu
            e_accu = e_accu + (eD + eF + eH)*self.eta
            Y = scipy.special.softmax(e_accu)
            YD = Y[0:K]
            YF = Y[K:K+E_asso]
            YH = Y[K+E_asso:2*K+K+E_asso]


    @staticmethod
    def expm_half_randsk(L,D):
        randv = np.random.randn(L.shape[0],D)/math.sqrt(float(D))
        randv = randv/np.linalg.norm(randv,axis=1)[:,None]
        ret = scipy.sparse.linalg.expm_multiply(L.copy(),randv)
        return ret


if __name__ == '__main__':
    pass