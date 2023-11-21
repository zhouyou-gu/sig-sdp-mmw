import math
import time

from sim_src.alg.mmwm_scipy import mmwm_scipy
from sim_src.env.env import env
from sim_src.scipy_util import csr_expm_rank_dsketch
import numpy
e = env(cell_size=5,seed=int(time.time()))
print(e.min_sinr,e.n_sta,e.rxpr)
a = mmwm_scipy(e.n_sta,e.min_sinr)
a.set_st(e.rxpr)
CC = a.run_fc_main(6., num_iterations=10000 * int(math.log(e.n_sta)))

