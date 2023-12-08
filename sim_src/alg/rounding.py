import scipy
import numpy as np
import math
class rand_rounding:
    @staticmethod
    def get_rand_Z_simplex_in_RK(K,Z):
        assert K >= Z
        base = np.zeros((int(K),int(Z)))
        np.fill_diagonal(base,1.)
        bm = np.mean(base,axis=1)
        # print(bm)
        base = base - bm[:,np.newaxis]
        base = base / np.linalg.norm(base,axis=0)
        rota = scipy.stats.special_ortho_group.rvs(K)
        base = rota @ base
        # print(base[:,0].T @ base[:,1].T)
        # print(base[:,1].T @ base[:,1].T)
        return base

    @staticmethod
    def get_group_vec_using_ehalf_nattempt(Z,A,rxpr,I_max,nattempt=50):
        # A = scipy.sparse.csc_matrix(A.shape)
        best = None
        best_pct = np.infty

        ss = np.asarray(rxpr.todense())
        asso = np.argmax(ss, axis=1)
        H = ss[:,asso]
        H = H.transpose()
        np.fill_diagonal(H, 0.)

        a = np.argmax(rxpr,axis=1)
        for i in range(nattempt):
            g = rand_rounding.get_group_vec_using_ehalf(Z,A)
            I = rand_rounding.get_interference(H,g)
            p = rand_rounding.get_violation_pct(I,I_max)
            # print(p,best_pct)

            if p<=best_pct:
                best_pct=p
                best = g
        print(best_pct)
        return best, best_pct

    @staticmethod
    def get_group_vec_using_ehalf(Z,A):
        K = A.shape[0]
        randv = np.random.randn(K,int(Z))/math.sqrt(float(Z))
        # print(np.linalg.norm(randv,axis=0).shape)
        randv = randv/np.linalg.norm(randv,axis=0)
        # randv = np.asarray(scipy.sparse.linalg.expm(A.copy()).todense()) @ randv
        # randv = rand_rounding.get_rand_Z_simplex_in_RK(K,Z)
        randv = scipy.sparse.linalg.expm_multiply(A.copy(),randv)
        return np.argmax(randv,axis=1)
    @staticmethod
    def get_interference(H,g):
        x = (g[:, np.newaxis] == g[np.newaxis, :])
        x = x.astype(float)
        HH = H * x
        return np.sum(HH,axis=1)

    @staticmethod
    def get_violation_pct(I,I_max):
        idx = I>I_max
        K = I.size
        return np.sum(idx.astype(float))/K



