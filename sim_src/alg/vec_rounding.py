import numpy
import scipy
import numpy as np
import math
class vec_rand_rounding:
    @staticmethod
    def get_group_vec_using_ehalf_nattempt(Z,A,rxpr,I_max,nattempt=1):
        best = None
        best_pct = np.infty
        best_Z = np.infty
        ss = np.asarray(rxpr.todense())
        asso = np.argmax(ss, axis=1)
        H = ss[:,asso]
        H = H.transpose()
        np.fill_diagonal(H, 0.)

        for i in range(nattempt):
            g, Z = vec_rand_rounding.get_group_vec_using_ehalf(Z,A,H,I_max)
            I = vec_rand_rounding.get_interference(H,g)
            p = vec_rand_rounding.get_violation_pct(I,I_max)
            if Z <= best_Z:
                best_Z = Z
                best_pct = p
        print(best_Z,best_pct)
        return best, best_pct

    @staticmethod
    def get_group_vec_using_ehalf(Z,A,H,I_max):
        K = A.shape[0]
        assigned = np.zeros(K, dtype=bool)
        grp_idx = np.zeros(K)
        ZZ = 0
        for z in range(int(Z-1)):
            ZZ = ZZ + 1
            idx_best = None
            K_best = 0
            I_best = 0
            for t in range(100):
                randv = np.random.randn(K,1)
                randv = np.asarray(scipy.sparse.linalg.expm_multiply(A.copy(),randv)).ravel()
                rank = np.argsort(randv[np.invert(assigned)])
                # add mask
                # idx = None
                skip_mask = np.ones(rank.size, dtype=bool)
                for i in range(rank.size):
                    skip_mask[i] = False
                    # do interference check
                    idx = rank[np.invert(skip_mask)]
                    # print(idx)
                    HH = H[:,np.invert(assigned)]
                    I = np.asarray(HH[:,idx].sum(axis=1)).ravel()
                    # print(idx.size,I[np.invert(assigned)][idx],I_max[np.invert(assigned)][idx])
                    vio = I[np.invert(assigned)][idx] > I_max[np.invert(assigned)][idx]
                    if np.all(vio == False):
                        continue
                    else:
                        skip_mask[i] = True
                        break
                idx = rank[np.invert(skip_mask)]
                if K_best < idx.size:
                    idx_best = idx
                    K_best = idx.size

            tmp = np.argwhere(np.invert(assigned)).ravel()
            # print(tmp,idx_best,"+++++++++++++++++++++++++++++")
            idx_best = tmp[idx_best]
            # print(idx)
            grp_idx[idx_best] = z
            assigned[idx_best] = True
            # print(assigned)

            if np.all(assigned == True):
                break
        grp_idx[np.invert(assigned)] = Z-1
        return grp_idx, ZZ+1


    @staticmethod
    def get_group_vec_using_ehalf_tmp(Z,A,H,I_max, th=0.):
        K = A.shape[0]
        assigned = np.zeros(K, dtype=bool)
        grp_idx = np.zeros(K)
        for z in range(int(Z)):
            randv = np.random.randn(K,1)
            randv = randv/np.linalg.norm(randv)
            randv = np.asarray(scipy.sparse.linalg.expm_multiply(A.copy(),randv)).ravel()
            idx = randv > th
            grp_idx[idx] = z
        return grp_idx, Z

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
