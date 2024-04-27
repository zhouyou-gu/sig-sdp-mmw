from sim_src.util import TIMED_OBJECT


class controller(TIMED_OBJECT):
    def __init__(self):
        self.env = None
        self.alg = None

    def run(self):
        state = self.env.get_state()
        tic = self._get_tic()
        Z, z_vec = self.alg.run(state)
        tim = self._get_tim(tic)
        res = self.env.step(Z, z_vec, tim)