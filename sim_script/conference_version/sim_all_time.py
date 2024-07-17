import os

import math
import time

import numpy as np
import scipy.sparse.linalg

from sim_src.alg.binary_search_relaxation import binary_search_relaxation
from sim_src.alg.gm import MAX_ASSO, MAX_GAIN, MAX_RAND
from sim_src.alg.lrp import lrp_solver
from sim_src.alg.mmw import mmw
from sim_src.alg.sdp_solver import admm_sdp_solver
from sim_src.env.env import env
from sim_src.util import GLOBAL_PROF_ENABLER, plot_a_array, GET_LOG_PATH_FOR_SIM_SCRIPT, CSV_WRITER_OBJECT

# GLOBAL_PROF_ENABLER.DISABLE()

log_path = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)
print(log_path)
log = CSV_WRITER_OBJECT(path=log_path)

np.set_printoptions(threshold=10)
np.set_printoptions(linewidth=1000)

REPEAT =  20

RHO =  5e-3

for CELL_SIZE in [6,8,10]:
    for seed in range(REPEAT):
        e = env(cell_size=CELL_SIZE,sta_density_per_1m2=RHO,seed=seed)
        bs = binary_search_relaxation()
        bs.force_lower_bound = False
        alg = admm_sdp_solver(nit=100)
        bs.feasibility_check_alg = alg
        z_vec, Z_fin, remainder = bs.run(e.generate_S_Q_hmax())

        time_list = []

        tic = bs._get_tic()
        alg = admm_sdp_solver(nit=100)
        _, gX = alg.run_with_state(0,Z_fin,e.generate_S_Q_hmax())
        z_vec, _, _ = alg.rounding(Z_fin,gX,e.generate_S_Q_hmax())
        tim = bs._get_tim(tic)
        time_list.append(tim)

        tic = bs._get_tic()
        alg = lrp_solver(nit=100)
        _, gX = alg.run_with_state(0,Z_fin,e.generate_S_Q_hmax())
        z_vec, _, _ = alg.rounding(Z_fin,gX,e.generate_S_Q_hmax())
        tim = bs._get_tim(tic)
        time_list.append(tim)

        tic = bs._get_tic()
        z_vec, _, _ = MAX_GAIN.run(Z_fin,state=(e.generate_S_Q_hmax()))
        tim = bs._get_tim(tic)
        time_list.append(tim)

        tic = bs._get_tic()
        z_vec, _, _ = MAX_ASSO.run(Z_fin,state=(e.generate_S_Q_hmax()))
        tim = bs._get_tim(tic)
        time_list.append(tim)
        log.log_mul_scalar(data_name="time-"+str(CELL_SIZE)+"-"+str(int(RHO*10000)),iteration=seed,values=time_list)
