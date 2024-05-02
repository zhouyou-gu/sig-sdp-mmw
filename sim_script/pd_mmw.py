import os

import math
import time

import numpy as np
import scipy.sparse.linalg

from sim_src.alg.binary_search_relaxation import binary_search_relaxation
from sim_src.alg.gm import MAX_ASSO, MAX_GAIN
from sim_src.alg.mmw import mmw
from sim_src.alg.mmwm_scipy import mmwm_scipy
from sim_src.env.env import env
from sim_src.util import GLOBAL_PROF_ENABLER, plot_a_array

# GLOBAL_PROF_ENABLER.DISABLE()
e = env(cell_size=10,seed=int(time.time()))
bs = binary_search_relaxation()

alg = mmw(nit=100, D=1, alpha=1., eta=0.1)

bs.feasibility_check_alg = alg
e.generate_S_Q_hmax()
z_vec, Z_fin, remainder = bs.run(e.generate_S_Q_hmax())
print(Z_fin,remainder,"mmw")
# Z_fin = 50
z_vec, Z_fin, remainder  = MAX_GAIN.run(Z_fin,state=(e.generate_S_Q_hmax()))
print(Z_fin,remainder,"mgain")

z_vec, Z_fin, remainder  = MAX_ASSO.run(Z_fin,state=(e.generate_S_Q_hmax()))
print(Z_fin,remainder,"masso")




