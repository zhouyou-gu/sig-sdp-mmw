import numpy as np
import scipy


class power_adaption:
    @staticmethod
    def get_power_adaption(g, rxpr, min_snr, max_iter=10):
        ss = np.asarray(rxpr.todense())
        asso = np.argmax(ss, axis=1)
        s_max = np.max(ss, axis=1)
        H = ss[:,asso]
        H = H.transpose()
        np.fill_diagonal(H, 0.)
        x = (g[:, np.newaxis] == g[np.newaxis, :])
        x = x.astype(float)
        H = H * x
        H = scipy.sparse.csr_matrix(H)

        K = H.shape[0]
        power = np.ones(K)

        sinr = (power*s_max)/((H @ power) + 1)
        idx = sinr<=min_snr
        ratio = min_snr/sinr + 1e-5

        # print("before",np.sum(idx.astype(float))/K)
        for i in range(max_iter):
            sinr = (power*s_max)/((H @ power) + 1)
            ratio = min_snr/sinr + 1e-5
            power[sinr > min_snr] *= ratio[sinr > min_snr]
        sinr = (power*s_max)/((H @ power) + 1)
        idx = sinr<min_snr
        # print("after",np.sum(idx.astype(float))/K)
        # print("after_remain",np.sum((ratio<1.).astype(float))/K, np.mean(ratio[ratio<1.]))

        return np.sum(idx.astype(float))/K, np.percentile(power,85), np.mean(ratio[ratio<1.])