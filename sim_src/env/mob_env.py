from sim_src.env.env import env


class mob_env(env):
    def step(self,Z,z_vec,time_us,mob_spd):
        #get slot duration
        self.rand_user_mobility(mob_spd,time_us/1e6)
        #evaluate packet loss
        #return number of packets loss and total number of packets
        return

