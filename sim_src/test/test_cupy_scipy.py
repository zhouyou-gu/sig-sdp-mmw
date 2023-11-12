from sim_src.alg.mmwm_scipy import mmwm_scipy
from sim_src.env.env import env
from sim_src.scipy_util import csr_expm_rank_dsketch

e = env()
print(e.min_sinr)
a = mmwm_scipy(e.n_sta,e.min_sinr)
a.set_st(e.rxpr)
CC = a.run_fc(10)

csr_expm_rank_dsketch(CC,K=a.K,d=10,r=10)
