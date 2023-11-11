import numpy as np
from scipy import linalg

from sim_src.util import StatusObject, counted


class mmwm(StatusObject):
    def __init__(self, eta = 0.1, T=1000, D=10):
        self.eta = eta
        self.T = T
        self.D = D

        self.P_t = np.identity(self.D)
        self.W_t = np.identity(self.D)
        self.M_t = np.identity(self.D)
        self.M_t_sum = np.zeros((self.D, self.D))


    def run_alg(self):
        for i in range(self.T):
            if self.step():
                continue
            else:
                break


    @counted
    def step(self):
        self.P_t = self.W_t/np.trace(self.W_t)
        success, self.M_t = self.get_cost_M()
        if not success:
            return False
        assert self.M_t.shape == (self.D,self.D)
        self.M_t_sum = self.M_t_sum + self.M_t

        self.W_t = self.matrix_exp(self.M_t_sum, self.eta)
        return True


    def get_cost_M(self):
        return True, np.identity(self.D)

    def matrix_exp(self, M_sum, eta):
        return linalg.expm(-eta*M_sum)


if __name__ == '__main__':
    O = mmwm()
    O.run_alg()