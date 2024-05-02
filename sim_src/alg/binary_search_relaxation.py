import math
import numpy as np

from sim_src.alg import alg_interface
from sim_src.util import STATS_OBJECT


class binary_search_relaxation(alg_interface,STATS_OBJECT):
    def __init__(self):
        self.feasibility_check_alg = None

    def set_bounds(self,state):
        S_gain = state[0].copy()
        S = S_gain + S_gain.transpose()
        S.setdiag(0)
        S.sort_indices()
        nnz_per_row = np.diff(S.indptr)
        ub = np.max(nnz_per_row)+1
        nnz_per_row = np.diff(state[1].indptr)
        lb = np.max(nnz_per_row)+1
        return  lb, ub

    def run(self,state):
        bd_tic = self._get_tic()
        left, right = self.set_bounds(state)
        tim = self._get_tim(bd_tic)
        self._add_np_log("bs_iters",0,np.array([left,right,tim]))

        bs_tic = self._get_tic()
        Z, gX, it = self.search(left, right, state)
        tim = self._get_tim(bs_tic)
        self._add_np_log("bs_search",0,np.array([left,right,Z,it,tim]))

        rd_tic = self._get_tic()
        z_vec, Z_fin, remainder = self.feasibility_check_alg.rounding(Z,gX,state)
        tim = self._get_tim(rd_tic)
        self._add_np_log("bs_round",0,np.array([left,right,Z_fin,tim]))

        return z_vec, Z_fin, remainder

    def search(self, left, right, state):
        it = 0
        while True:
            it += 1
            mid = math.floor(float(left+right)/2.)
            f, gX = self.feasibility_check_alg.run_with_state(it,mid,state)
            if not f:
                left = mid+1
            else:
                right = mid
            print(left,right)
            if left >= right:
                break

        return right, gX, it