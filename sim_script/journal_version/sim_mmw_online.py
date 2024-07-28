import os

import math
import time

import numpy as np
import scipy.sparse.linalg

from sim_src.alg.binary_search_relaxation import binary_search_relaxation
from sim_src.alg.gm import MAX_ASSO, MAX_GAIN, MAX_RAND
from sim_src.alg.lrp import lrp_solver
from sim_src.alg.mmw import mmw
from sim_src.alg.sdp_solver import admm_sdp_solver, rand_sdp_solver
from sim_src.env.env import env
from sim_src.env.mob_env import mob_env
from sim_src.util import GLOBAL_PROF_ENABLER, plot_a_array, GET_LOG_PATH_FOR_SIM_SCRIPT, CSV_WRITER_OBJECT

# GLOBAL_PROF_ENABLER.DISABLE()

log_path = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)
print(log_path)
log = CSV_WRITER_OBJECT(path=log_path)

np.set_printoptions(threshold=10)
np.set_printoptions(linewidth=1000)

REPEAT = 100

RHO =  75e-4

N_SPEED = 11
for CELL_SIZE in [10]:
    for seed in range(REPEAT):
        e = mob_env(cell_size=CELL_SIZE,sta_density_per_1m2=RHO,seed=seed)
        bs = binary_search_relaxation()
        tic = bs._get_tic()
        alg = mmw(nit=150,eta=0.04)
        bs.feasibility_check_alg = alg
        z_vec, Z_fin, remainder = bs.run(e.generate_S_Q_hmax())
        _, gX = alg.run_with_state(0,Z_fin,e.generate_S_Q_hmax())
        tim = bs._get_tim(tic)
        print(tim,Z_fin)
        for i in range(N_SPEED):
            bler = e.evaluate_bler(z_vec, Z_fin)
            log.log_mul_scalar(data_name="online-mmw-"+str(i)+"-"+str(150)+"-"+str(CELL_SIZE)+"-"+str(int(RHO*10000)),iteration=seed,values=bler.tolist())
            e.step_time(tim,mob_spd_meter_s=0.1)
            z_vec, _, _ = alg.rounding(Z_fin,gX,e.generate_S_Q_hmax())

        e = mob_env(cell_size=CELL_SIZE,sta_density_per_1m2=RHO,seed=seed)
        bs = binary_search_relaxation()
        tic = bs._get_tic()
        alg = mmw(nit=75,eta=0.04)
        bs.feasibility_check_alg = alg
        z_vec, Z_fin, remainder = bs.run(e.generate_S_Q_hmax())
        _, gX = alg.run_with_state(0,Z_fin,e.generate_S_Q_hmax())
        tim = bs._get_tim(tic)
        print(tim,Z_fin)
        for i in range(N_SPEED):
            bler = e.evaluate_bler(z_vec, Z_fin)
            log.log_mul_scalar(data_name="online-mmw75-"+str(i)+"-"+str(150)+"-"+str(CELL_SIZE)+"-"+str(int(RHO*10000)),iteration=seed,values=bler.tolist())
            e.step_time(tim,mob_spd_meter_s=0.1)
            z_vec, _, _ = alg.rounding(Z_fin,gX,e.generate_S_Q_hmax())


        e = mob_env(cell_size=CELL_SIZE,sta_density_per_1m2=RHO,seed=seed)
        bs = binary_search_relaxation()
        tic = bs._get_tic()
        alg = mmw(nit=150,eta=0.04)
        bs.feasibility_check_alg = alg
        z_vec, Z_fin, remainder = bs.run(e.generate_S_Q_hmax())
        _, gX = alg.run_with_state(0,Z_fin,e.generate_S_Q_hmax())
        tim = bs._get_tim(tic)
        print(tim,Z_fin)
        for i in range(N_SPEED):
            bler = e.evaluate_bler(z_vec, Z_fin)
            log.log_mul_scalar(data_name="online-ideal-"+str(i)+"-"+str(150)+"-"+str(CELL_SIZE)+"-"+str(int(RHO*10000)),iteration=seed,values=bler.tolist())
            e.step_time(0,mob_spd_meter_s=0.1)
            z_vec, _, _ = alg.rounding(Z_fin,gX,e.generate_S_Q_hmax())

        e = mob_env(cell_size=CELL_SIZE,sta_density_per_1m2=RHO,seed=seed)
        bs = binary_search_relaxation()
        tic = bs._get_tic()
        z_vec, Z_fin, _ = MAX_GAIN.run(-1,state=e.generate_S_Q_hmax(),not_Z_bound=True)
        tim = bs._get_tim(tic)
        print(tim,Z_fin)
        for i in range(N_SPEED):
            bler = e.evaluate_bler(z_vec, Z_fin)
            log.log_mul_scalar(data_name="online-mgain-"+str(i)+"-"+str(150)+"-"+str(CELL_SIZE)+"-"+str(int(RHO*10000)),iteration=seed,values=bler.tolist())
            e.step_time(tim,mob_spd_meter_s=0.1,resolution_us=10000.)
