import numpy as np

from sim_src.util import STATS_OBJECT


class MAX_GAIN(STATS_OBJECT):

    @staticmethod
    def run(Z,state,nattempt=1):
        K = state[0].shape[0]
        S_gain = state[0].copy()
        Q_asso = state[1]
        h_max = state[2]
        S_gain.setdiag(0)
        S_gain_T_no_diag = S_gain.transpose()
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
                    tmp_h = np.asarray(S_gain[krank[i]].toarray()).ravel()
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
        S_gain = state[0].copy()
        Q_asso = state[1]
        h_max = state[2]
        S_gain.setdiag(0)
        S_gain_T_no_diag = S_gain.transpose()
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
                    tmp_h = np.asarray(S_gain[krank[i]].toarray()).ravel()
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

class MAX_RAND(STATS_OBJECT):

    @staticmethod
    def run(Z,state,nattempt=1):
        K = state[0].shape[0]
        S_gain = state[0].copy()
        Q_asso = state[1]
        h_max = state[2]
        S_gain.setdiag(0)
        S_gain.eliminate_zeros()
        S_gain_T_no_diag = S_gain.transpose()

        S_gain_index = np.split(S_gain.indices, S_gain.indptr)[1:-1]
        Q_asso_index = np.split(Q_asso.indices, S_gain.indptr)[1:-1]

        not_assigned = np.ones(K, dtype=bool)

        inprod = np.random.randn(Z,K)
        sorted_indices = np.argsort(-inprod, axis=0)
        rank = np.argsort(np.random.randn(K))

        z_vec = np.zeros(K)

        gain_sum = []
        asso_sum = []
        slot_asn = []
        for z in range(Z):
            gain_sum.append(np.zeros(K))
            asso_sum.append(np.zeros(K))
            slot_asn.append([])

        user_sum = 0
        for kk in range(K):
            k = rank[kk]
            for zz in range(Z):
                z = sorted_indices[zz,k]

                if not not_assigned[k]:
                    break

                # do interference check
                neighbor_index = np.intersect1d(np.array(slot_asn[z]),S_gain_index[k])
                neighbor_index = np.append(neighbor_index,k).astype(int)
                tmp_h = np.asarray(S_gain[k].toarray()).ravel()
                vio = (gain_sum[z][neighbor_index] + tmp_h[neighbor_index]) > h_max[neighbor_index]
                if np.any(vio == True):
                    continue

                # do association check
                neighbor_index = np.intersect1d(np.array(slot_asn[z]),Q_asso_index[k])
                neighbor_index = np.append(neighbor_index,k).astype(int)
                tmp_a = np.asarray(Q_asso[k].toarray()).ravel()
                vio = (asso_sum[z][neighbor_index] + tmp_a[neighbor_index]) >= 1
                if np.any(vio == True):
                    continue

                gain_sum[z] += np.asarray(S_gain[k].toarray()).ravel()
                asso_sum[z] += np.asarray(Q_asso[k].toarray()).ravel()
                slot_asn[z].append(k)

                user_sum += 1
                not_assigned[k] = False
                z_vec[k] = z
                break


        if not np.all(not_assigned == False):
            z_vec[not_assigned] = np.random.randint(Z,size = int(not_assigned.sum()))
            print(z_vec[not_assigned])
        return z_vec, Z, np.sum(not_assigned)
