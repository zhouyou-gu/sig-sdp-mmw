import numpy as np

from mmwm_alg import mmwm


class mmwm_sdp_alg(mmwm):
    def __init__(self,R=10, rho=10, eta=0.1, T=1000, D=10):
        mmwm.__init__(self,eta=eta,T=T,D=D)
        self.R = R
        self.rho = rho

        self.X_t = np.identity(self.D)
        self.y_t = None
        self.y_t_sum = None
        self.C = np.zeros((self.D,self.D))
        self.A_list = []

    def add_A(self,A):
        self.A_list.append(A)

    def set_C(self,C):
        self.C = C
    def get_cost_M(self):
        self.X_t = self.R * self.P_t
        success, self.y_t = self.oracle_func(self.X_t)
        if not success:
            return success, None

        assert np.size(self.y_t) == len(self.A_list)
        Ay_sum = np.sum([self.A_list[i]*self.y_t[i] for i in range(len(self.A_list))])
        self.M_t = 1/self.rho*(Ay_sum-self.C)
        self.y_t_sum = self.y_t_sum + self.y_t

    def oracle_func(self,X) -> (bool,np.ndarray):
        return False, None
