import math

import numpy as np
import scipy
class env():
    C = 299792458.0
    PI = 3.14159265358979323846
    HIDDEN_LOSS = 200.
    NOISE_FLOOR_DBM = -94.
    BOLTZMANN = 1.3803e-23
    NOISEFIGURE = 13
    def __init__(self, cell_edge = 20., cell_size = 20, sta_density_per_1m2 = 5e-3, fre_Hz = 4e9, txp_dbm_hi = 5., txp_offset = 2., min_s_n_ratio = 0.1, packet_bit = 800, bandwidth = 5e6, slot_time=1.25e-4, max_err = 1e-5, seed=1):
        self.rand_gen_loc = np.random.default_rng(seed)
        self.rand_gen_fad = np.random.default_rng(seed)
        self.rand_gen_mob = np.random.default_rng(seed)

        self.cell_edge = cell_edge
        self.cell_size = cell_size

        self.grid_edge = self.cell_edge * self.cell_size

        self.n_ap = int(self.cell_size ** 2)
        self.ap_offset = self.cell_edge / 2.

        self.sta_density_per_1m2 = sta_density_per_1m2
        self.sta_density_per_grid = self.sta_density_per_1m2 * self.cell_edge ** 2
        self.n_sta = int(self.cell_size**2 * self.sta_density_per_grid)

        self.fre_Hz = fre_Hz
        self.lam = self.C / self.fre_Hz
        self.txp_dbm_hi = txp_dbm_hi
        self.txp_offset = txp_offset
        self.min_s_n_ratio = min_s_n_ratio
        self.packet_bit = packet_bit
        self.bandwidth = bandwidth
        self.slot_time = slot_time
        self.max_err = max_err

        self.ap_locs = None
        self.sta_locs = None
        self.sta_dirs = None

        self.min_sinr = None
        self.loss = None

        self._config_ap_locs()
        self._config_sta_locs()
        self._config_sta_dirs()

        print(self._compute_min_sinr())

    def _config_ap_locs(self):
        x=np.linspace(0 + self.ap_offset, self.grid_edge - self.ap_offset, self.cell_size)
        y=np.linspace(0 + self.ap_offset, self.grid_edge - self.ap_offset, self.cell_size)
        xx,yy=np.meshgrid(x,y)
        self.ap_locs = np.array((xx.ravel(), yy.ravel())).T

    def _config_sta_locs(self):
        self.sta_locs = self.rand_gen_loc.uniform(low=0.,high=self.grid_edge,size=(self.n_sta,2))

    def _config_sta_dirs(self):
        dd = self.rand_gen_mob.standard_normal(size=(self.n_sta,2))
        self.sta_dirs = dd/np.linalg.norm(dd,axis=1,keepdims=True)

    def _get_random_dir(self):
        dd = self.rand_gen_mob.standard_normal(2)
        return dd/np.linalg.norm(dd)

    def _compute_min_sinr(self):
        min_sinr_db = env.bisection_method(self.packet_bit,self.bandwidth,self.slot_time, self.max_err)
        self.min_sinr = env.db_to_dec(min_sinr_db)
        return self.min_sinr

    def rand_user_mobility(self, mobility_in_meter_s = 0., t_s = 0, resolution_s = 1e-1):

        if mobility_in_meter_s == 0. or t_s == 0.:
            return
        n_step = int(t_s/resolution_s)
        for n in range(n_step):
            for i in range(self.n_sta):
                dd = self.sta_dirs[i] * mobility_in_meter_s * resolution_s
                x = self.sta_locs[i][0] + dd[0]
                y = self.sta_locs[i][1] + dd[1]
                if np.linalg.norm(np.array((x,y)),np.inf) <= self.grid_edge:
                    self.sta_locs[i] = np.array([x,y])
                else:
                    self.sta_dirs[i] = self._get_random_dir()

    @classmethod
    def bandwidth_txpr_to_noise_dBm(cls, B):
        return env.NOISE_FLOOR_DBM

    @staticmethod
    def fre_dis_to_loss_dB(fre_Hz, dis):
        L = 20. * math.log10(fre_Hz/1e6) + 16 - 28
        loss = L + 28 * np.log10(dis+1) # at least one-meter distance
        return loss

    @staticmethod
    def db_to_dec(snr_db):
        return 10.**(snr_db/10.)

    @staticmethod
    def dec_to_db(snr_dec):
        return 10.* math.log10(snr_dec)

    @staticmethod
    def polyanskiy_model(snr_dec, L, B, T):
        nu = - L * math.log(2.) + B * T * math.log(1+snr_dec)
        do = math.sqrt(B * T * (1. - 1./((1.+snr_dec)**2)))
        return scipy.stats.norm.sf(nu/do)

    @staticmethod
    def err(x, L, B, T, max_err):
        snr = env.db_to_dec(x)
        return env.polyanskiy_model(snr, L, B, T)/max_err - 1.

    @staticmethod
    def bisection_method(L, B, T, max_err=1e-5, a=-5., b=30., tol=0.1):

        if env.err(a, L, B, T, max_err) * env.err(b, L, B, T, max_err) >= 0:
            print("Bisection method fails.")
            return None

        while (env.err(a, L, B, T, max_err) - env.err(b, L, B, T, max_err)) > tol:
            midpoint = (a + b) / 2
            if env.err(midpoint, L, B, T, max_err) == 0:
                return midpoint
            elif env.err(a, L, B, T, max_err) * env.err(midpoint, L, B, T, max_err) < 0:
                b = midpoint
            else:
                a = midpoint

        return (a + b) / 2

    def _compute_txp(self):
        dis = scipy.spatial.distance.cdist(self.sta_locs,self.ap_locs)
        gain = -env.fre_dis_to_loss_dB(self.fre_Hz,dis)
        smax = np.max(gain, axis=1)
        t = env.dec_to_db(self._compute_min_sinr()) - (smax - self.bandwidth_txpr_to_noise_dBm(self.bandwidth))
        ret = np.reshape(t + env.dec_to_db(self.txp_offset),(self.n_sta,-1))
        return ret

    def _compute_state(self):
        dis = scipy.spatial.distance.cdist(self.sta_locs,self.ap_locs)
        self.loss = env.fre_dis_to_loss_dB(self.fre_Hz,dis)

        rxpr_db = self._compute_txp() - self.loss - self.bandwidth_txpr_to_noise_dBm(self.bandwidth)
        rxpr_hi = 10 ** (rxpr_db / 10.)

        rxpr_hi[rxpr_hi < self.min_s_n_ratio] = 0.

        rxpr_hi = scipy.sparse.csr_matrix(rxpr_hi)

        return rxpr_hi

    def _compute_state_real(self):
        dis = scipy.spatial.distance.cdist(self.sta_locs,self.ap_locs)
        self.loss = env.fre_dis_to_loss_dB(self.fre_Hz,dis)

        rxpr_db = self._compute_txp() - self.loss - self.bandwidth_txpr_to_noise_dBm(self.bandwidth)
        rxpr_hi = 10 ** (rxpr_db / 10.)

        rxpr_hi = scipy.sparse.csr_matrix(rxpr_hi)

        return rxpr_hi

    def generate_S_Q_hmax(self,real=False)-> (scipy.sparse.csr_matrix, scipy.sparse.csr_matrix):
        if real:
            rxpr = self._compute_state_real()
        else:
            rxpr = self._compute_state()

        K = rxpr.shape[0]
        A = rxpr.shape[1]
        ss = rxpr.tolil().toarray()
        asso = np.argmax(ss, axis=1)
        asso_indicator = np.zeros((K,A))
        asso_indicator[np.arange(K), asso] = 1

        asso_indicator = asso_indicator.astype(bool)
        Q_asso = scipy.sparse.lil_matrix((K, K))

        for a in range(A):
            idx = np.where(asso_indicator[:,a] == 1)[0]
            Q_asso[np.ix_(idx, idx)] = 1
        Q_asso = Q_asso.tocsr()
        Q_asso.setdiag(0.)
        Q_asso.sort_indices()
        Q_asso.eliminate_zeros()
        S_gain = rxpr[:, asso]
        S_gain.eliminate_zeros()
        S_gain.sort_indices()

        h_max = S_gain.diagonal()/self._compute_min_sinr() - 1.
        return S_gain, Q_asso, h_max

    def evaluate_sinr(self,z,Z):
        rxpr = self._compute_state_real()
        S_gain, Q_asso, _ = self.generate_S_Q_hmax(real=True)
        S_gain = np.array(S_gain.tolil().toarray())
        S_gain_T_no_diag = S_gain.copy().transpose()
        np.fill_diagonal(S_gain_T_no_diag, 0)

        K = rxpr.shape[0]
        sinr = np.zeros(K)+1e-3
        for zz in range(Z):
            kidx = z==zz
            kidx = np.arange(K)[kidx]
            signal = S_gain.diagonal()[kidx]
            interference = np.asarray(S_gain_T_no_diag[kidx][:,kidx].sum(axis=1)).ravel()
            sinr[kidx] = signal/(interference+1)
        print(sinr)
        # ss = rxpr.tolil().toarray()
        # asso = np.argmax(ss, axis=1)
        #
        # A = ss.shape[1]
        # for a in range(A):
        #     for zz in range(Z):
        #         kidx = np.logical_and(asso == a , z==zz)
        #         max_sinr = np.max(np.asarray(sinr[kidx]))
        #         max_sinr_idx = np.argmax(np.asarray(sinr[kidx]))
        #         sinr[kidx] = 0
        #         np.asarray(sinr[kidx])[max_sinr_idx] = max_sinr
        return sinr

    def evaluate_bler(self,z,Z):
        sinr = self.evaluate_sinr(z,Z)
        K = sinr.size
        bler = np.zeros(K)
        for k in range(K):
            bler[k] = env.polyanskiy_model(sinr[k],self.packet_bit,self.bandwidth,self.slot_time)
        return bler
    def evaluate_pckl(self,z,Z):
        bler = self.evaluate_bler(z,Z)
        K = bler.size
        pckl = np.random.choice([0, 1], size=(K,), p=[bler, 1-bler])
        return pckl


    def check_cell_edge_snr_err(self):
        l = env.fre_dis_to_loss_dB(e.fre_Hz,self.cell_edge/2*math.sqrt(2))
        s_db = self.txp_dbm_hi - l - env.bandwidth_txpr_to_noise_dBm(e.bandwidth)
        s_dec = e.db_to_dec(s_db)
        err = env.polyanskiy_model(s_dec,self.packet_bit,self.bandwidth,self.slot_time)
        print("snr_db", s_db, "snr_dec", s_dec, "err", err)
        return

if __name__ == '__main__':

    dd = np.random.randn(10,2)
    dd = dd/np.linalg.norm(dd,axis=1,keepdims=True)
    print(dd)

    # exit(0)
    e = env(cell_size=5,seed=2)
    print(e.ap_locs)
    print(e.sta_locs)
    print("n_sta",e.n_sta)
    print("n_ap",e.n_ap)



    print(e._compute_state())

    print(e._compute_min_sinr())
    exit(0)
    e.check_cell_edge_snr_err()
    a = e.sta_locs[0].copy()
    print(e.sta_locs[0])
    e.rand_user_mobility(mobility_in_meter_s=1,t_s=1)
    b = e.sta_locs[0].copy()
    print(e.sta_locs[0])
    print(np.linalg.norm(a-b),a-b)
