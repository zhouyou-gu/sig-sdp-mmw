import math

from sim_src.alg import alg_interface
from sim_src.util import TIMED_OBJECT


class binary_search_relaxation(alg_interface,TIMED_OBJECT):
    def __init__(self):
        TIMED_OBJECT.__init__(self)

        self.feasibility_check_alg = None

    def set_bounds(self,state):

        return  0, 0

    def run(self,state):
        tic = self._get_tic()
        left, right = self.set_bounds(state)
        tim = self._get_tim(tic)

        tic = self._get_tic()
        Z, gX, it = self.search(left, right, state)
        tim = self._get_tim(tic)

        tic = self._get_tic()
        Z_fin , z_vec = self.feasibility_check_alg.rounding(Z,gX)
        tim = self._get_tim(tic)
        return Z_fin, z_vec

    def search(self, left, right, state):
        it = 0
        while True:
            it += 1
            mid = math.floor(float(left+right)/2.)
            f, gX = self.feasibility_check_alg.run_with_state(mid,state)
            if not f:
                left = mid+1
            else:
                right = mid
            if left >= right:
                break

        return right, gX, it