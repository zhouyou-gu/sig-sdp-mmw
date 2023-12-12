import numpy
import scipy
import numpy as np
import math

from sim_src.scipy_util import csr_scal_rows_inplace
from sim_src.util import profile


class vec_rand_rounding:
    @staticmethod
    def get_group_vec_using_ehalf_nattempt(Z,A,rxpr,I_max,nattempt=10):
        ss = np.asarray(rxpr.todense())
        asso = np.argmax(ss, axis=1)
        H = ss[:,asso]
        H = H.transpose()
        np.fill_diagonal(H, 0.)
        H = scipy.sparse.csr_matrix(H)

        gr, Z = vec_rand_rounding.get_group_vec_using_ehalf(Z,A,H,I_max,nattempt)
        I = vec_rand_rounding.get_interference(H,gr)
        p = vec_rand_rounding.get_violation_pct(I,I_max)

        return Z, p, gr

    @staticmethod
    def get_group_vec_using_ehalf(Z,A,H,I_max,nattempt):
        K = A.shape[0]
        not_assigned = np.ones(K, dtype=bool)
        grp_idx = np.zeros(K)
        ZZ = 0
        AA = np.asarray(scipy.sparse.linalg.expm(A.copy().tocsc()).todense())
        X_norm = np.linalg.norm(AA,axis=1)
        for z in range(int(Z)):
            ZZ += 1
            idx_best = None
            K_best = 0
            for t in range(nattempt):
                randv = np.random.randn(K,1)
                randv = np.asarray(scipy.sparse.linalg.expm_multiply(A.copy(),randv)).ravel()
                randv = randv/X_norm
                tmp_rank = np.argsort(randv[not_assigned])
                not_assigned_idx = np.argwhere(not_assigned).ravel()
                not_assigned_idx_rank = not_assigned_idx[tmp_rank]
                # print(rank)
                # add mask
                check_mask = np.zeros(not_assigned_idx_rank.size, dtype=bool)
                for i in range(not_assigned_idx_rank.size):
                    check_mask[i] = True
                    # do interference check
                    idx = not_assigned_idx_rank[check_mask]
                    HH = H[idx,:].tocsc()
                    HH = HH[:,idx].tocsr()
                    I = np.asarray(HH.sum(axis=1)).ravel()
                    # print(idx.size,I,I_max[idx])
                    vio = I > I_max[idx]
                    if np.all(vio == False):
                        continue
                    else:
                        check_mask[i] = False
                        break
                idx = not_assigned_idx_rank[check_mask]
                if K_best < idx.size:
                    idx_best = idx
                    K_best = idx.size

            # print(idx_best)
            grp_idx[idx_best] = z
            not_assigned[idx_best] = False
            # print(not_assigned)
            if np.all(not_assigned == False):
                break
        # if not np.all(not_assigned == False):
        #     grp_idx[not_assigned] = Z-1

        if not np.all(not_assigned == False):
            grp_idx[not_assigned] = np.random.randint(Z,size = int(not_assigned.sum()))



        return grp_idx, ZZ

    @staticmethod
    def get_interference(H,g):
        x = (g[:, np.newaxis] == g[np.newaxis, :])
        x = x.astype(float)
        HH = np.asarray(H.todense()) * x
        return np.sum(HH,axis=1)

    @staticmethod
    def get_violation_pct(I,I_max):
        idx = I>I_max
        K = I.size
        return np.sum(idx.astype(float))/K
