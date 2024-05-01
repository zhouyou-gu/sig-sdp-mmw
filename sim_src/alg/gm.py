from sim_src.util import STATS_OBJECT


class heuristic_method(STATS_OBJECT):

    def rounding(self,Z,gX,state,nattempt=1):
        K = gX.shape[0]
        D = gX.shape[1]
        S_gain = state[0]
        Q_asso = state[1]
        h_max = state[3]
        S_gain_T_no_diag = S_gain.transpose()
        S_gain_T_no_diag.setdiag(0)
        not_assigned = np.ones(K, dtype=bool)

        z_vec = np.zeros(K)
        for z in range(Z):
            k_list_z = []
            for n in range(nattempt):
                k_list = []
                randv = np.random.randn(D,1)
                randv = np.asarray(scipy.sparse.linalg.expm_multiply(gX[not_assigned],randv)).ravel()
                kindx = np.arange(K)[not_assigned]
                krank = kindx[np.argsort(randv[not_assigned])]
                for i in range(krank.size):
                    tmp = k_list.copy()
                    tmp.append(krank[i])
                    # do interference check
                    SS = S_gain_T_no_diag[tmp,:].tocsc()
                    SS = SS[:,tmp].tocsr()
                    h = np.asarray(SS.sum(axis=1)).ravel()
                    vio = h > h_max[tmp]
                    if np.any(vio == True):
                        continue

                    # do association check
                    QQ = Q_asso[tmp,:].tocsc()
                    QQ = QQ[:,tmp].tocsr()
                    q = np.asarray(QQ.sum(axis=1)).ravel()
                    if np.any(q >= 1.):
                        continue
                    k_list.append(krank[i])

                if len(k_list) > len(k_list_z):
                    k_list_z = k_list

            z_vec[k_list_z] = z
            not_assigned[k_list_z] = False

        if not np.all(not_assigned == False):
            z_vec[not_assigned] = np.random.randint(Z,size = int(not_assigned.sum()))

        return z_vec, Z