import math

import numpy as np

from sim_src.env.env import env
from sim_src.util import STATS_OBJECT


class mob_env(env):
    def step_a_period(self,Z,mob_spd_meter_s):
        self.rand_user_mobility(mob_spd_meter_s,self.get_period_time_us(Z))
        return

    def get_n_period(self,Z,time_us):
        return math.ceil(time_us/(Z*self.slot_time*1e6))

    def get_period_time_us(self,Z):
        return Z*self.slot_time*1e6

    def step_time(self,time_us,mob_spd_meter_s):
        self.rand_user_mobility(mob_spd_meter_s,time_us,resolution_us=100000.)

class controller(STATS_OBJECT):
    def __init__(self):
        self.env = None
        self.alg = None

    def run(self,total_time_s,mob_spd_meter_s):
        past_time_us = 0.
        process_time_us = []
        total_pck = 0
        total_packet_loss = 0

        tic = self._get_tic()
        z_vec, Z_fin, remainder = self.alg.run(self.env.generate_S_Q_hmax())
        tim_us = self._get_tim(tic)
        for i in range(self.env.get_n_period(Z_fin,tim_us)):
            self.env.step_a_period(Z_fin,mob_spd_meter_s)


        while True:
            tic = self._get_tic()
            z_vec_next, Z_fin_next, remainder_next = self.alg.run(self.env.generate_S_Q_hmax())
            tim_us = self._get_tim(tic)
            process_time_us.append(tim_us)

            for i in range(self.env.get_n_period(Z_fin,tim_us)): # time to user current z_vec, Z_fin is based on the running time of solver for next solutions
                pckl = self.env.evaluate_pckl(z_vec, Z_fin) # evaluation use current z_vec, Z_fin
                total_packet_loss += np.sum(pckl)
                total_pck += self.env.n_sta
                self.env.step_a_period(Z_fin,mob_spd_meter_s) # step use current z_vec, Z_fin, assume users are fixed in each period

                past_time_us += self.env.get_period_time_us(Z_fin)

            if total_time_s*1e6<past_time_us:
                break

            z_vec = z_vec_next
            Z_fin = Z_fin_next
            remainder = remainder_next


        return total_packet_loss, total_pck, past_time_us, process_time_us
