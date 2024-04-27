import math

import numpy as np
import scipy

from sim_src.scipy_util import csr_scal_rows_inplace


class mmw:
    def __init__(self, D = 10, nit = 100):
        self.nit = nit
        self.D = D
        pass

    def run_with_state(self, Z, state):
        pass


    def rounding(self,Z,gX):
        pass

    def _run(self,Z,state):
        K = state[0].shape[0]

        S_gain = scipy.sparse.csr_matrix((K, K))
        Q_asso = scipy.sparse.csr_matrix((K, K))
        nz_idx_x_gain = state[0]
        nz_idx_y_gain = state[0]
        nz_idx_x_asso = state[0]
        nz_idx_y_asso = state[0]

        E_asso = self._get_nE_asso(state)
        C = E_asso+2*K

        AD = scipy.sparse.diags(np.zeros(K)).tocsr()
        AF = scipy.sparse.csr_matrix((K, K))
        AH = scipy.sparse.csr_matrix((K, K))

        YD = np.ones(K)/float(C)
        YF = np.ones(E_asso)/float(C)
        YH = np.ones(E_asso)/float(C)

        e_accu = np.zeros(C)
        L_accu = scipy.sparse.csr_matrix((K, K))

        for i in range(self.nit):


            ## YD, AD -> LD
            LD = (scipy.sparse.diags(YD)-np.sum(YD)/K*scipy.sparse.diags(np.ones(K)))/(1.-1./K)

            ## YF, AF -> LF
            YF_m = scipy.sparse.coo_matrix((YF, (nz_idx_x_asso, nz_idx_y_asso)), shape=(K, K)).tocsr()
            YF_m = YF_m + YF_m.transpose()
            LF = (YF_m-np.sum(YF)/(K*(Z-1))*scipy.sparse.diags(np.ones(K)))/(1./2.+1./(K(Z-1)))

            ## YH, AH -> LH
            YH_m = scipy.sparse.coo_matrix((YH, (nz_idx_x_asso, nz_idx_y_asso)), shape=(K, K)).tocsr()
            LH = S_gain.copy()
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
            eH = (X_offdi+1./(Z-1))/(1./(K*(Z-1))+1./2.)

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
