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
from sim_src.util import GLOBAL_PROF_ENABLER, plot_a_array, GET_LOG_PATH_FOR_SIM_SCRIPT, CSV_WRITER_OBJECT

# GLOBAL_PROF_ENABLER.DISABLE()

log_path = GET_LOG_PATH_FOR_SIM_SCRIPT(__file__)
print(log_path)
log = CSV_WRITER_OBJECT(path=log_path)

np.set_printoptions(threshold=10)
np.set_printoptions(linewidth=1000)

REPEAT =  100

RHO =  75e-4

for CELL_SIZE in range(5,16):
    for seed in range(REPEAT):
        e = env(cell_size=CELL_SIZE,sta_density_per_1m2=RHO,seed=seed)
        bs = binary_search_relaxation()
        alg = mmw(nit=150,eta=0.04)
        bs.feasibility_check_alg = alg
        z_vec, Z_fin, remainder = bs.run(e.generate_S_Q_hmax())
        bler = e.evaluate_bler(z_vec, Z_fin)
        mbler = np.mean(bler)
        wbler = np.max(bler)
        log.log_mul_scalar(data_name="mmw-"+str(CELL_SIZE)+"-"+str(int(RHO*10000)),iteration=seed,values=[Z_fin] + bler.tolist())

        alg = rand_sdp_solver()
        _, gX = alg.run_with_state(0,Z_fin,e.generate_S_Q_hmax())
        z_vec, Z_tmp, _ = alg.rounding(Z_fin,gX,e.generate_S_Q_hmax())
        bler = e.evaluate_bler(z_vec, Z_fin)
        mbler = np.mean(bler)
        wbler = np.max(bler)
        log.log_mul_scalar(data_name="rand-"+str(CELL_SIZE)+"-"+str(int(RHO*10000)),iteration=seed,values=[Z_fin] + bler.tolist())
        print(Z_tmp,Z_fin,"rand")

        alg = lrp_solver(nit=100)
        _, gX = alg.run_with_state(0,Z_fin,e.generate_S_Q_hmax())
        z_vec, Z_tmp, _ = alg.rounding(Z_fin,gX,e.generate_S_Q_hmax())
        bler = e.evaluate_bler(z_vec, Z_fin)
        mbler = np.mean(bler)
        wbler = np.max(bler)
        log.log_mul_scalar(data_name="ladmm-"+str(CELL_SIZE)+"-"+str(int(RHO*10000)),iteration=seed,values=[Z_fin] + bler.tolist())
        print(Z_tmp,Z_fin,"ladmm")

        z_vec, Z_tmp, _ = MAX_GAIN.run(Z_fin,state=(e.generate_S_Q_hmax()))
        bler = e.evaluate_bler(z_vec, Z_fin)
        mbler = np.mean(bler)
        wbler = np.max(bler)
        log.log_mul_scalar(data_name="mgain-"+str(CELL_SIZE)+"-"+str(int(RHO*10000)),iteration=seed,values=[Z_fin] + bler.tolist())
        print(Z_tmp,Z_fin,"mgain")

        z_vec, Z_tmp, _ = MAX_ASSO.run(Z_fin,state=(e.generate_S_Q_hmax()))
        bler = e.evaluate_bler(z_vec, Z_fin)
        mbler = np.mean(bler)
        wbler = np.max(bler)
        log.log_mul_scalar(data_name="masso-"+str(CELL_SIZE)+"-"+str(int(RHO*10000)),iteration=seed,values=[Z_fin] + bler.tolist())
        print(Z_tmp,Z_fin,"masso")
