import math

import numpy as np
import scipy
class env():
    C = 299792458.0
    PI = 3.14159265358979323846
    HIDDEN_LOSS = 200.
    NOISE_FLOOR_DBM = -94.
    def __init__(self, cell_edge = 20., cell_size = 20, sta_density_per_1m2 = 1e-2, fre_Hz = 4e9, txp_dbm = 0., min_s_n_ratio = 0.1, packet_bit = 400, bandwidth = 2e6, slot_time=1.25e-4, max_err = 1e-5, seed=1):
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
        self.txp_dbm = txp_dbm
        self.min_s_n_ratio = min_s_n_ratio
        self.packet_bit = packet_bit
        self.bandwidth = bandwidth
        self.slot_time = slot_time
        self.max_err = max_err

        self.ap_locs = None
        self.sta_locs = None

        self.min_sinr = None
        self.loss = None
        self.rxpr = None

        self._config_ap_locs()
        self._config_sta_locs()

        self._compute_min_sinr()

        self._compute_state()

    def _config_ap_locs(self):
        x=np.linspace(0 + self.ap_offset, self.grid_edge - self.ap_offset, self.cell_size)
        y=np.linspace(0 + self.ap_offset, self.grid_edge - self.ap_offset, self.cell_size)
        xx,yy=np.meshgrid(x,y)
        self.ap_locs = np.array((xx.ravel(), yy.ravel())).T

    def _config_sta_locs(self):
        self.sta_locs = self.rand_gen_loc.uniform(low=0.,high=self.grid_edge,size=(self.n_sta,2))


    def _compute_min_sinr(self):
        def db_to_dec(snr_db):
            return 10.**(snr_db/10.)

        def polyanskiy_model(snr, L, B, T):
            nu = - L * math.log(2.) + B * T * math.log(1+snr)
            do = math.sqrt(B * T * (1. - 1./((1.+snr)**2)))
            return scipy.stats.norm.sf(nu/do)

        def bisection_method(L, B, T, max_err, a=-5., b=30., tol=0.1):
            def err(x, L, B, T):
                snr = db_to_dec(x)
                return polyanskiy_model(snr, L, B, T)/max_err - 1.

            if err(a, L, B, T) * err(b, L, B, T) >= 0:
                print("Bisection method fails.")
                return None

            while (err(a, L, B, T) - err(b, L, B, T)) > tol:
                midpoint = (a + b) / 2
                if err(midpoint, L, B, T) == 0:
                    return midpoint
                elif err(a, L, B, T) * err(midpoint, L, B, T) < 0:
                    b = midpoint
                else:
                    a = midpoint

            return (a + b) / 2

        self.min_sinr = bisection_method(self.packet_bit,self.bandwidth,self.slot_time, self.max_err)

        return self.min_sinr

    def _compute_state(self):
        dis = scipy.spatial.distance.cdist(self.sta_locs,self.ap_locs)
        L = 20. * math.log10(self.fre_Hz/1e6) + 16 - 28
        loss = L + 28 * np.log10(dis+1) # at least one-meter distance

        self.loss = loss

        rxpr_db = self.txp_dbm - self.loss - self.NOISE_FLOOR_DBM
        self.rxpr = 10 ** (rxpr_db/10.)

        self.rxpr[self.rxpr<0.1] = 0.

        self.rxpr = scipy.sparse.csr_matrix(self.rxpr)

        return self.loss


if __name__ == '__main__':
    e = env()
    print(e.ap_locs)
    print(e.sta_locs)
    print("n_sta",e.n_sta)
    print("n_ap",e.n_ap)
    print(e._compute_state())
    # print(np.linalg.norm(e.ap_locs[3]-e.sta_locs[6]))
    # print(20. * math.log10(4e9/1e6) + 16 - 28 + 28 * np.log10(10*1.47))
    # print(10 * math.log10(1e6))
    print(e._compute_min_sinr())

    def polyanskiy_model(snr, L, B, T):
        nu = - L * math.log(2.) + B * T * math.log(1+snr)
        do = math.sqrt(B * T * (1. - 1./((1.+snr)**2)))
        return scipy.stats.norm.sf(nu/do)

    print(polyanskiy_model(10.**(4.67/10.),400,2e6,1.25e-4))