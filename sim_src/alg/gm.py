import numpy as np

from sim_src.util import STATS_OBJECT


class MAX_GAIN(STATS_OBJECT):

    @staticmethod
    def run(Z,state,nattempt=1):
        K = state[0].shape[0]
        S_gain = state[0]
        Q_asso = state[1]
        h_max = state[2]
        S_gain_T_no_diag = S_gain.transpose()
        S_gain_T_no_diag.setdiag(0)
        not_assigned = np.ones(K, dtype=bool)

        S_sum = np.asarray(S_gain_T_no_diag.sum(axis=1)).ravel()

        z_vec = np.zeros(K)
        ZZ = 0
        for z in range(Z):
            ZZ += 1
            tmp_gain_sum = np.zeros(K)
            tmp_asso_sum = np.zeros(K)
            k_list_z = []
            for n in range(nattempt):
                k_list = []
                kindx = np.arange(K)[not_assigned]
                krank = kindx[np.argsort(-S_sum[not_assigned])]
                for i in range(krank.size):
                    tmp = k_list.copy()
                    tmp.append(krank[i])
                    # do interference check
                    tmp_h = np.asarray(S_gain_T_no_diag[krank[i]].toarray()).ravel()
                    vio = (tmp_gain_sum[tmp] + tmp_h[tmp]) > h_max[tmp]
                    if np.any(vio == True):
                        continue

                    # do association check
                    tmp_a = np.asarray(Q_asso[krank[i]].toarray()).ravel()
                    vio = (tmp_asso_sum[tmp] + tmp_a[tmp]) >= 1

                    if np.any(vio == True):
                        continue

                    tmp_gain_sum += tmp_h
                    tmp_asso_sum += tmp_a
                    k_list.append(krank[i])

                if len(k_list) > len(k_list_z):
                    k_list_z = k_list
            z_vec[k_list_z] = z
            not_assigned[k_list_z] = False
            if np.all(not_assigned == False):
                break

        if not np.all(not_assigned == False):
            z_vec[not_assigned] = np.random.randint(Z,size = int(not_assigned.sum()))

        return z_vec, ZZ, np.sum(not_assigned)


class MAX_ASSO(STATS_OBJECT):

    @staticmethod
    def run(Z,state,nattempt=1):
        K = state[0].shape[0]
        S_gain = state[0]
        Q_asso = state[1]
        h_max = state[2]
        S_gain_T_no_diag = S_gain.transpose()
        S_gain_T_no_diag.setdiag(0)
        not_assigned = np.ones(K, dtype=bool)

        A_sum = np.asarray(Q_asso.sum(axis=1)).ravel()

        z_vec = np.zeros(K)
        ZZ = 0
        for z in range(Z):
            ZZ += 1
            tmp_gain_sum = np.zeros(K)
            tmp_asso_sum = np.zeros(K)
            k_list_z = []
            for n in range(nattempt):
                k_list = []
                kindx = np.arange(K)[not_assigned]
                krank = kindx[np.argsort(-A_sum[not_assigned])]
                for i in range(krank.size):
                    tmp = k_list.copy()
                    tmp.append(krank[i])
                    # do interference check
                    tmp_h = np.asarray(S_gain_T_no_diag[krank[i]].toarray()).ravel()
                    vio = (tmp_gain_sum[tmp] + tmp_h[tmp]) > h_max[tmp]
                    if np.any(vio == True):
                        continue

                    # do association check
                    tmp_a = np.asarray(Q_asso[krank[i]].toarray()).ravel()
                    vio = (tmp_asso_sum[tmp] + tmp_a[tmp]) >= 1

                    if np.any(vio == True):
                        continue

                    tmp_gain_sum += tmp_h
                    tmp_asso_sum += tmp_a
                    k_list.append(krank[i])

                if len(k_list) > len(k_list_z):
                    k_list_z = k_list
            z_vec[k_list_z] = z
            not_assigned[k_list_z] = False
            if np.all(not_assigned == False):
                break

        if not np.all(not_assigned == False):
            z_vec[not_assigned] = np.random.randint(Z,size = int(not_assigned.sum()))

        return z_vec, ZZ, np.sum(not_assigned)