import os

import math
import time

import numpy as np
import scipy.sparse.linalg

from sim_src.alg.binary_search_relaxation import binary_search_relaxation
from sim_src.alg.gm import MAX_ASSO, MAX_GAIN, MAX_RAND
from sim_src.alg.mmw import mmw, mmw_vec_rounding, mmw_gc, mmw_vec_rounding_slot_based
from sim_src.alg.mmwm_scipy import mmwm_scipy
from sim_src.env.env import env
from sim_src.util import GLOBAL_PROF_ENABLER, plot_a_array

# GLOBAL_PROF_ENABLER.DISABLE()

np.set_printoptions(threshold=10)
np.set_printoptions(linewidth=1000)

e = env(cell_size=10,seed=int(time.time()))
print(e.n_sta)
bs = binary_search_relaxation()
bs.force_lower_bound = True
alg = mmw_vec_rounding_slot_based(nit=1000, D=1, alpha=1., eta=0.05)
alg.DEBUG=True
bs.feasibility_check_alg = alg
e.generate_S_Q_hmax()
z_vec, Z_fin, remainder = bs.run(e.generate_S_Q_hmax())
print(Z_fin,remainder,"mmw")
# Z_fin = 50
z_vec, Z_fin, remainder  = MAX_GAIN.run(Z_fin,state=(e.generate_S_Q_hmax()))
print(Z_fin,remainder,"mgain")

z_vec, Z_fin, remainder  = MAX_ASSO.run(Z_fin,state=(e.generate_S_Q_hmax()))
print(Z_fin,remainder,"masso")

z_vec, Z_fin, remainder  = MAX_RAND.run(Z_fin,state=(e.generate_S_Q_hmax()))
print(Z_fin,remainder,"mrand")




