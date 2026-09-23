import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pandapower as pp
import gym
from gym import spaces
from collections import deque
import random
import matplotlib.pyplot as plt
import rl_utils
import torch.nn.functional as F
from torch.distributions import Normal

import time
import os

NUM_CON = 18
NUM_GEN = 2
NUM_SVC = 4
NUM_BSS = 4
NUM_WT = 4
NUM_PV = 4
NUM_BUS = 119

np.random.seed(0)


class PowerSystemEnv(gym.Env):
    def __init__(self):
        super(PowerSystemEnv, self).__init__()

        self.net = pp.create_empty_network()

        self._adjust_network_119()

        pp.runpp(self.net)

        total_load = sum(self.net.load.p_mw)
        total_loss = self.net.res_line.pl_mw.sum()
        balance_error = abs(-total_load - total_loss)
        print(balance_error)

        self._adjust_network()

        self.num_load_buses = len(self.net.load)

        self.ori_load = self.net.load.p_mw.values.copy()
        self.ori_load_Q = self.net.load.q_mvar.values.copy()

        self.gen_min = 0
        self.gen_max = 1
        self.gen_ramp = 0.4

        self.SOC_min = 0.125
        self.SOC_max = 2.375
        self.SOC_scale = 0.5

        self.load_min = 0.6
        self.load_max = 1.2

        self.load_min_Q = 0.6
        self.load_max_Q = 1.2

        self.pv_min = 0
        self.pv_max = 3

        self.wind_min = 0
        self.wind_max = 2

        self.svc_min = -0.3
        self.svc_max = 0.3
        self.svc_scale = 0.3

        self.action_space = spaces.Box(
            low=np.array([-1] * NUM_CON),
            high=np.array([1] * NUM_CON),
            dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(1 + self.num_load_buses + self.num_load_buses + NUM_WT + NUM_PV + NUM_GEN + NUM_BSS,),
            dtype=np.float32)

    def _adjust_network_119(self):
        self.net.sn_mva = 1

        bus0 = pp.create_bus(self.net, vn_kv=12.66, zone=1.0, max_vm_pu=1.06, min_vm_pu=0.94, name="0")
        pp.create_ext_grid(self.net, bus=bus0, vm_pu=1.0, va_degree=0, max_p_mw=1000, min_p_mw=-1000, max_q_mvar=1000, min_q_mvar=-1000)
        self.net.ext_grid['vm_pu'] = 1

        for i in range(118):
            pp.create_bus(self.net, vn_kv=12.66, zone=1.0, max_vm_pu=1.06, min_vm_pu=0.94, name='{}'.format(i + 1))

        list_pq = [[0, 0],
                   [133.84, 101.14], [16.214, 11.292], [34.315, 21.845], [73.016, 63.602], [144.2, 68.604],
                   [104.47, 61.725], [28.547, 11.503], [87.56, 51.073], [198.2, 106.77], [146.8, 75.995],
                   [26.04, 18.687], [52.1, 23.22], [141.9, 117.5], [21.87, 28.79], [33.37, 26.45],
                   [32.43, 25.23], [20.234, 11.906], [156.94, 78.523], [546.29, 351.4], [180.31, 164.2],
                   [93.167, 54.594], [85.18, 39.65], [168.1, 95.178], [125.11, 150.22], [16.03, 24.62],
                   [26.03, 24.62], [594.56, 522.62], [120.62, 59.117], [102.38, 99.554], [513.4, 318.5],
                   [475.25, 456.14], [151.43, 136.79], [205.38, 83.302], [131.6, 93.082], [448.4, 369.79],
                   [440.52, 321.64], [112.54, 55.134], [53.963, 38.998], [393.05, 342.6], [326.74, 278.56],
                   [536.26, 240.24], [76.247, 66.562], [53.52, 39.76], [40.328, 31.964], [39.653, 20.758],
                   [66.195, 42.361], [73.904, 51.653], [114.77, 57.965], [918.37, 1205.1], [210.3, 146.66],
                   [66.68, 56.608], [42.207, 40.184], [433.74, 283.41], [62.1, 26.86], [92.46, 88.38],
                   [85.188, 55.436], [345.3, 332.4], [22.5, 16.83], [80.551, 49.156], [95.86, 90.758],
                   [62.92, 47.7], [478.8, 463.74], [120.94, 52.006], [139.11, 100.34], [391.78, 193.5],
                   [27.741, 26.713], [52.814, 25.257], [66.89, 38.713], [467.5, 395.14], [594.85, 239.74],
                   [132.5, 84.363], [52.699, 22.482], [869.79, 614.775], [31.349, 29.817], [192.39, 122.43],
                   [65.75, 45.37], [238.15, 223.22], [294.55, 162.47], [485.57, 437.92], [243.53, 183.03],
                   [243.53, 183.03], [134.25, 119.29], [22.71, 27.96], [49.513, 26.515], [383.78, 257.16],
                   [49.64, 20.6], [22.473, 11.806], [62.93, 42.96], [30.67, 34.93], [62.53, 66.79],
                   [114.57, 81.748], [81.292, 66.526], [31.733, 15.96], [33.32, 60.48], [531.28, 224.85],
                   [507.03, 367.42], [26.39, 11.7], [45.99, 30.392], [100.66, 47.572], [456.48, 350.3],
                   [522.56, 449.29], [408.43, 168.46], [141.48, 134.25], [104.43, 66.024], [96.793, 83.647],
                   [493.92, 419.34], [225.38, 135.88], [509.21, 387.21], [188.5, 173.46], [918.03, 898.55],
                   [305.08, 215.37], [54.38, 40.97], [211.14, 192.9], [67.009, 53.336], [162.07, 90.321],
                   [48.785, 29.156], [33.9, 18.98]]
        for i in range(118):
            list_pq[i][0] = list_pq[i][0] / 1000 * 0.5
            list_pq[i][1] = list_pq[i][1] / 1000 * 0.5

        for i in range(118):
            pp.create_load(self.net, bus=i+1, p_mw=list_pq[i][0], q_mvar=list_pq[i][1], type=None, controllable=False)

        list_rx = [(0, 1, 0.001, 0.001), (1, 2, 0.036, 0.01296), (2, 3, 0.033, 0.01188), (2, 4, 0.045, 0.0162),
                   (4, 5, 0.015, 0.054), (5, 6, 0.015, 0.054), (6, 7, 0.015, 0.0125), (7, 8, 0.018, 0.014),
                   (8, 9, 0.021, 0.063), (2, 10, 0.166, 0.1344), (10, 11, 0.112, 0.0789), (11, 12, 0.187, 0.313),
                   (12, 13, 0.142, 0.1512), (13, 14, 0.18, 0.118), (14, 15, 0.15, 0.045), (15, 16, 0.16, 0.18),
                   (16, 17, 0.157, 0.171), (11, 18, 0.218, 0.285), (18, 19, 0.118, 0.185), (19, 20, 0.16, 0.196),

                   (20, 21, 0.12, 0.189), (21, 22, 0.12, 0.0789), (22, 23, 1.41, 0.723), (23, 24, 0.293, 0.1348),
                   (24, 25, 0.133, 0.104), (25, 26, 0.178, 0.134), (26, 27, 0.178, 0.134), (4, 28, 0.015, 0.0296),
                   (28, 29, 0.012, 0.0276), (29, 30, 0.12, 0.2766), (30, 31, 0.21, 0.243), (31, 32, 0.12, 0.054),
                   (32, 33, 0.178, 0.234), (33, 34, 0.178, 0.234), (34, 35, 0.154, 0.162), (30, 36, 0.187, 0.261),
                   (36, 37, 0.133, 0.099), (29, 38, 0.33, 0.194), (38, 39, 0.31, 0.194), (39, 40, 0.13, 0.194),

                   (40, 41, 0.28, 0.15), (41, 42, 1.18, 0.85), (42, 43, 0.42, 0.2436), (43, 44, 0.27, 0.0972),
                   (44, 45, 0.339, 0.1221), (45, 46, 0.27, 0.1779), (35, 47, 0.21, 0.1383), (47, 48, 0.12, 0.0789),
                   (48, 49, 0.15, 0.0987), (49, 50, 0.15, 0.0987), (50, 51, 0.24, 0.1581), (51, 52, 0.12, 0.0789),
                   (52, 53, 0.405, 0.1458), (53, 54, 0.405, 0.1458), (29, 55, 0.391, 0.141), (55, 56, 0.406, 0.1461),
                   (56, 57, 0.406, 0.1461), (57, 58, 0.706, 0.5461), (58, 59, 0.338, 0.1218), (59, 60, 0.338, 0.1218),

                   (60, 61, 0.207, 0.0747), (61, 62, 0.247, 0.8922), (1, 63, 0.028, 0.0418), (63, 64, 0.117, 0.2016),
                   (64, 65, 0.255, 0.0918), (65, 66, 0.21, 0.0759), (66, 67, 0.383, 0.138), (67, 68, 0.504, 0.3303),
                   (68, 69, 0.406, 0.1461), (69, 70, 0.962, 0.761), (70, 71, 0.165, 0.06), (71, 72, 0.303, 0.1092),
                   (72, 73, 0.303, 0.1092), (73, 74, 0.206, 0.144), (74, 75, 0.233, 0.084), (75, 76, 0.591, 0.1773),
                   (76, 77, 0.126, 0.0453), (64, 78, 0.559, 0.3687), (78, 79, 0.186, 0.1227), (79, 80, 0.186, 0.1227),

                   (80, 81, 0.26, 0.139), (81, 82, 0.154, 0.148), (82, 83, 0.23, 0.128), (83, 84, 0.252, 0.106),
                   (84, 85, 0.18, 0.148), (79, 86, 0.16, 0.182), (86, 87, 0.2, 0.23), (87, 88, 0.16, 0.393),
                   (65, 89, 0.669, 0.2412), (89, 90, 0.266, 0.1227), (90, 91, 0.266, 0.1227), (91, 92, 0.266, 0.1227),
                   (92, 93, 0.266, 0.1227), (93, 94, 0.233, 0.115), (94, 95, 0.496, 0.138), (91, 96, 0.196, 0.18),
                   (96, 97, 0.196, 0.18), (97, 98, 0.1866, 0.122), (98, 99, 0.0746, 0.318), (1, 100, 0.0625, 0.0265),

                   (100, 101, 0.1501, 0.234), (101, 102, 0.1347, 0.0888), (102, 103, 0.2307, 0.1203), (103, 104, 0.447, 0.1608),
                   (104, 105, 0.1632, 0.0588), (105, 106, 0.33, 0.099), (106, 107, 0.156, 0.0561), (107, 108, 0.3819, 0.1374),
                   (108, 109, 0.1626, 0.0585), (109, 110, 0.3819, 0.1374), (110, 111, 0.2445, 0.0879), (110, 112, 0.2088, 0.0753),
                   (112, 113, 0.2301, 0.0828), (100, 114, 0.6102, 0.2196), (114, 115, 0.1866, 0.127), (115, 116, 0.3732, 0.246),
                   (116, 117, 0.405, 0.367), (117, 118, 0.489, 0.438)]
        for i in range(118):
            pp.create_line_from_parameters(self.net, from_bus=list_rx[i][0], to_bus=list_rx[i][1], length_km=1,
                                           r_ohm_per_km=list_rx[i][2], x_ohm_per_km=list_rx[i][3],
                                           c_nf_per_km=0, type='ol', max_i_ka=99999.0, max_loading_percent=100.0)

    def _adjust_network(self):
        pp.create_sgen(self.net, bus=28, p_mw=0, q_mvar=0, name='DG_29')
        pp.create_sgen(self.net, bus=63, p_mw=0, q_mvar=0, name='DG_64')

        pp.create_sgen(self.net, bus=19, p_mw=0, q_mvar=0, name='SVC_20')
        pp.create_sgen(self.net, bus=55, p_mw=0, q_mvar=0, name='SVC_56')
        pp.create_sgen(self.net, bus=78, p_mw=0, q_mvar=0, name='SVC_79')
        pp.create_sgen(self.net, bus=104, p_mw=0, q_mvar=0, name='SVC_105')

        pp.create_sgen(self.net, bus=11, p_mw=0, q_mvar=0, name='BSS_12')
        pp.create_sgen(self.net, bus=33, p_mw=0, q_mvar=0, name='BSS_34')
        pp.create_sgen(self.net, bus=67, p_mw=0, q_mvar=0, name='BSS_68')
        pp.create_sgen(self.net, bus=102, p_mw=0, q_mvar=0, name='BSS_103')

        pp.create_sgen(self.net, bus=10, p_mw=0, q_mvar=0, name='WT_11')
        pp.create_sgen(self.net, bus=31, p_mw=0, q_mvar=0, name='WT_32')
        pp.create_sgen(self.net, bus=65, p_mw=0, q_mvar=0, name='WT_66')
        pp.create_sgen(self.net, bus=100, p_mw=0, q_mvar=0, name='WT_101')

        pp.create_sgen(self.net, bus=21, p_mw=0, q_mvar=0, name='PV_22')
        pp.create_sgen(self.net, bus=35, p_mw=0, q_mvar=0, name='PV_36')
        pp.create_sgen(self.net, bus=69, p_mw=0, q_mvar=0, name='PV_70')
        pp.create_sgen(self.net, bus=106, p_mw=0, q_mvar=0, name='PV_107')

    def _create_load_profile(self):
        a0_1 = np.random.uniform(0.6, 0.8, (12, self.num_load_buses))
        a0_2 = np.random.uniform(0.8, 1, (12, self.num_load_buses))
        a0_3 = np.random.uniform(1, 1.2, (12, self.num_load_buses))
        a0_4 = np.random.uniform(0.8, 1, (12, self.num_load_buses))
        a1 = np.vstack([a0_1, a0_2, a0_3, a0_4])

        a2 = []
        for i in range(24 * 2):
            a2.append(a1[i] * self.ori_load)
        a2 = np.array(a2)

        a0_1_Q = np.random.uniform(0.6, 0.8, (12, self.num_load_buses))
        a0_2_Q = np.random.uniform(0.8, 1, (12, self.num_load_buses))
        a0_3_Q = np.random.uniform(1, 1.2, (12, self.num_load_buses))
        a0_4_Q = np.random.uniform(0.8, 1, (12, self.num_load_buses))
        a1_Q = np.vstack([a0_1_Q, a0_2_Q, a0_3_Q, a0_4_Q])

        a2_Q = []
        for i in range(24 * 2):
            a2_Q.append(a1_Q[i] * self.ori_load_Q)
        a2_Q = np.array(a2_Q)

        return a2, a2_Q

    def _create_renew_profile(self):
        ori_pv_22 = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.08555, 0.2146, 0.28005, 0.34255, 0.4083,
            0.4965, 0.57495, 0.65735, 0.7352, 0.77635,
            0.7934, 0.76625, 0.7384, 0.66305, 0.5944,
            0.5039, 0.4242, 0.34665, 0.27375, 0.21285,
            0.15455, 0.112, 0.0765, 0.04975, 0.03195,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0
        ]
        ori_pv_36 = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.0988, 0.14345, 0.1949, 0.2572, 0.33485,
            0.43135, 0.5385, 0.61845, 0.69505, 0.7613,
            0.7733, 0.7947, 0.7868, 0.7421, 0.6802,
            0.61125, 0.5389, 0.46325, 0.3659, 0.2736,
            0.2012, 0.14325, 0.09855, 0.06755, 0.045,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0
        ]
        ori_pv_70 = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0876, 0.1305, 0.19295, 0.2675,
            0.3532, 0.4657, 0.5539, 0.65265,
            0.7311, 0.7817, 0.79875, 0.76765,
            0.7246, 0.64585, 0.53615, 0.43295,
            0.33835, 0.26345, 0.1897, 0.12735,
            0.08535, 0.0548, 0.03135, 0.0107,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0
        ]
        ori_pv_107 = [
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0, 0.0, 0.0, 0.0,
            0.06405, 0.1636, 0.2201, 0.2948, 0.3683,
            0.44055, 0.52885, 0.62395, 0.69795,
            0.75565, 0.7707, 0.79765, 0.7864,
            0.7322, 0.6563, 0.5836, 0.49325,
            0.40825, 0.32125, 0.2513, 0.1813,
            0.1302, 0.085, 0.0585, 0.0386,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
            0.0, 0.0
        ]
        pv_22_avail_list = []
        pv_36_avail_list = []
        pv_70_avail_list = []
        pv_107_avail_list = []
        for i in range(24 * 2):
            pv_22_avail_list.append(ori_pv_22[i] * np.random.uniform(0.8, 1.2) * 3)
            pv_36_avail_list.append(ori_pv_36[i] * np.random.uniform(0.8, 1.2) * 3)
            pv_70_avail_list.append(ori_pv_70[i] * np.random.uniform(0.8, 1.2) * 3)
            pv_107_avail_list.append(ori_pv_107[i] * np.random.uniform(0.8, 1.2) * 3)
        pv_22_avail_list = np.array(pv_22_avail_list)
        pv_36_avail_list = np.array(pv_36_avail_list)
        pv_70_avail_list = np.array(pv_70_avail_list)
        pv_107_avail_list = np.array(pv_107_avail_list)

        return {
            'wind_11': np.random.uniform(1.2, 1.8, 24 * 2),
            'wind_32': np.random.uniform(0.8, 1.2, 24 * 2),
            'wind_66': np.random.uniform(0.8, 1.2, 24 * 2),
            'wind_101': np.random.uniform(1.2, 1.8, 24 * 2),
            'pv_22': pv_22_avail_list,
            'pv_36': pv_36_avail_list,
            'pv_70': pv_70_avail_list,
            'pv_107': pv_107_avail_list,
        }

    def reset(self):
        self.hour = 0

        self.SOC_12 = 1.25
        self.SOC_34 = 1.25
        self.SOC_68 = 1.25
        self.SOC_103 = 1.25

        gen_init = self.gen_min
        self.net.sgen.loc[0, 'p_mw'] = gen_init
        self.net.sgen.loc[1, 'p_mw'] = gen_init

        self.load_profile, self.load_profile_Q = self._create_load_profile()
        self.renew_profile = self._create_renew_profile()

        state = self._get_state()

        return state

    def new_reset(self, list_feng_11, list_feng_32, list_feng_66, list_feng_101,
                  list_pv_22, list_pv_36, list_pv_70, list_pv_107,
                  list_fu_he, list_fu_he_Q):
        self.hour = 0

        self.SOC_12 = 1.25
        self.SOC_34 = 1.25
        self.SOC_68 = 1.25
        self.SOC_103 = 1.25

        gen_init = self.gen_min
        self.net.sgen.loc[0, 'p_mw'] = gen_init
        self.net.sgen.loc[1, 'p_mw'] = gen_init

        self.load_profile = list_fu_he
        self.load_profile_Q = list_fu_he_Q

        self.renew_profile = {
            'wind_11': list_feng_11,
            'wind_101': list_feng_101,
            'wind_32': list_feng_32,
            'wind_66': list_feng_66,
            'pv_22': list_pv_22,
            'pv_36': list_pv_36,
            'pv_70': list_pv_70,
            'pv_107': list_pv_107,
        }

        state = self._get_state()

        return state

    def _get_state(self):
        load = self.load_profile[self.hour].copy()
        load_Q = self.load_profile_Q[self.hour].copy()

        renew_pred = [self.renew_profile['wind_11'][self.hour],
                      self.renew_profile['wind_101'][self.hour],
                      self.renew_profile['wind_32'][self.hour],
                      self.renew_profile['wind_66'][self.hour],
                      self.renew_profile['pv_22'][self.hour],
                      self.renew_profile['pv_36'][self.hour],
                      self.renew_profile['pv_70'][self.hour],
                      self.renew_profile['pv_107'][self.hour]].copy()

        gen_p = [self.net.sgen.p_mw.values[0],
                 self.net.sgen.p_mw.values[1]].copy()

        BSS_p = [self.SOC_12,
                 self.SOC_34,
                 self.SOC_68,
                 self.SOC_103].copy()

        t_normalized = self.hour / 47

        BSS_p_12_normalized = (BSS_p[0] - self.SOC_min) / (self.SOC_max - self.SOC_min + 1e-6)
        BSS_p_34_normalized = (BSS_p[1] - self.SOC_min) / (self.SOC_max - self.SOC_min + 1e-6)
        BSS_p_68_normalized = (BSS_p[2] - self.SOC_min) / (self.SOC_max - self.SOC_min + 1e-6)
        BSS_p_103_normalized = (BSS_p[3] - self.SOC_min) / (self.SOC_max - self.SOC_min + 1e-6)

        gen_p_29_normalized = (gen_p[0] - self.gen_min) / (self.gen_max - self.gen_min + 1e-6)
        gen_p_64_normalized = (gen_p[1] - self.gen_min) / (self.gen_max - self.gen_min + 1e-6)

        load_normalized = ((load - self.ori_load * self.load_min) /
                           (self.ori_load * self.load_max - self.ori_load * self.load_min + 1e-6))
        load_Q_normalized = ((load_Q - self.ori_load_Q * self.load_min_Q) /
                           (self.ori_load_Q * self.load_max_Q - self.ori_load_Q * self.load_min_Q + 1e-6))

        self.wind_11_normalized = (renew_pred[0] - self.wind_min) / (self.wind_max - self.wind_min + 1e-6)
        self.wind_32_normalized = (renew_pred[1] - self.wind_min) / (self.wind_max - self.wind_min + 1e-6)
        self.wind_66_normalized = (renew_pred[2] - self.wind_min) / (self.wind_max - self.wind_min + 1e-6)
        self.wind_101_normalized = (renew_pred[3] - self.wind_min) / (self.wind_max - self.wind_min + 1e-6)
        self.pv_22_normalized = (renew_pred[4] - self.pv_min) / (self.pv_max - self.pv_min + 1e-6)
        self.pv_36_normalized = (renew_pred[5] - self.pv_min) / (self.pv_max - self.pv_min + 1e-6)
        self.pv_70_normalized = (renew_pred[6] - self.pv_min) / (self.pv_max - self.pv_min + 1e-6)
        self.pv_107_normalized = (renew_pred[7] - self.pv_min) / (self.pv_max - self.pv_min + 1e-6)

        state = np.concatenate([
            [t_normalized],
            load_normalized,
            load_Q_normalized,
            [self.wind_11_normalized, self.wind_32_normalized, self.wind_66_normalized, self.wind_101_normalized],
            [self.pv_22_normalized, self.pv_36_normalized, self.pv_70_normalized, self.pv_107_normalized],
            [gen_p_29_normalized, gen_p_64_normalized],
            [BSS_p_12_normalized, BSS_p_34_normalized, BSS_p_68_normalized, BSS_p_103_normalized]
        ])

        return state.astype(np.float32)

    def step(self, action):
        get_action = action[:NUM_CON]

        gen_29_delta = get_action[0] * self.gen_ramp / 2
        gen_64_delta = get_action[1] * self.gen_ramp / 2
        new_gen_29_p = self.net.sgen.p_mw[0] + gen_29_delta
        new_gen_64_p = self.net.sgen.p_mw[1] + gen_64_delta
        new_gen_29_p = np.clip(new_gen_29_p, self.gen_min, self.gen_max)
        new_gen_64_p = np.clip(new_gen_64_p, self.gen_min, self.gen_max)

        new_SVC_20 = get_action[2] * self.svc_scale
        new_SVC_56 = get_action[3] * self.svc_scale
        new_SVC_79 = get_action[4] * self.svc_scale
        new_SVC_105 = get_action[5] * self.svc_scale

        SOC_12_delta = get_action[6] * self.SOC_scale / 2
        SOC_34_delta = get_action[7] * self.SOC_scale / 2
        SOC_68_delta = get_action[8] * self.SOC_scale / 2
        SOC_103_delta = get_action[9] * self.SOC_scale / 2
        new_SOC_12 = self.SOC_12 + SOC_12_delta
        new_SOC_34 = self.SOC_34 + SOC_34_delta
        new_SOC_68 = self.SOC_68 + SOC_68_delta
        new_SOC_103 = self.SOC_103 + SOC_103_delta
        new_SOC_12 = np.clip(new_SOC_12, self.SOC_min, self.SOC_max)
        new_SOC_34 = np.clip(new_SOC_34, self.SOC_min, self.SOC_max)
        new_SOC_68 = np.clip(new_SOC_68, self.SOC_min, self.SOC_max)
        new_SOC_103 = np.clip(new_SOC_103, self.SOC_min, self.SOC_max)

        if self.SOC_12 - new_SOC_12 >= 0:
            new_BSS_12_p = (self.SOC_12 - new_SOC_12) * 0.98 * 2
        else:
            new_BSS_12_p = (self.SOC_12 - new_SOC_12) * 1.02 * 2
        if self.SOC_34 - new_SOC_34 >= 0:
            new_BSS_34_p = (self.SOC_34 - new_SOC_34) * 0.98 * 2
        else:
            new_BSS_34_p = (self.SOC_34 - new_SOC_34) * 1.02 * 2
        if self.SOC_68 - new_SOC_68 >= 0:
            new_BSS_68_p = (self.SOC_68 - new_SOC_68) * 0.98 * 2
        else:
            new_BSS_68_p = (self.SOC_68 - new_SOC_68) * 1.02 * 2
        if self.SOC_103 - new_SOC_103 >= 0:
            new_BSS_103_p = (self.SOC_103 - new_SOC_103) * 0.98 * 2
        else:
            new_BSS_103_p = (self.SOC_103 - new_SOC_103) * 1.02 * 2

        WT_11_pu = np.clip(self.wind_11_normalized.copy(), 0.0, 1.0)
        new_WT_11_Q = get_action[10] * np.sqrt(np.maximum(0.0, 1.0 - WT_11_pu ** 2)) * self.wind_max
        WT_32_pu = np.clip(self.wind_32_normalized.copy(), 0.0, 1.0)
        new_WT_32_Q = get_action[11] * np.sqrt(np.maximum(0.0, 1.0 - WT_32_pu ** 2)) * self.wind_max
        WT_66_pu = np.clip(self.wind_66_normalized.copy(), 0.0, 1.0)
        new_WT_66_Q = get_action[12] * np.sqrt(np.maximum(0.0, 1.0 - WT_66_pu ** 2)) * self.wind_max
        WT_101_pu = np.clip(self.wind_101_normalized.copy(), 0.0, 1.0)
        new_WT_101_Q = get_action[13] * np.sqrt(np.maximum(0.0, 1.0 - WT_101_pu ** 2)) * self.wind_max

        PV_22_pu = np.clip(self.pv_22_normalized.copy(), 0.0, 1.0)
        new_PV_22_Q = get_action[14] * np.sqrt(np.maximum(0.0, 1.0 - PV_22_pu ** 2)) * self.pv_max
        PV_36_pu = np.clip(self.pv_36_normalized.copy(), 0.0, 1.0)
        new_PV_36_Q = get_action[15] * np.sqrt(np.maximum(0.0, 1.0 - PV_36_pu ** 2)) * self.pv_max
        PV_70_pu = np.clip(self.pv_70_normalized.copy(), 0.0, 1.0)
        new_PV_70_Q = get_action[16] * np.sqrt(np.maximum(0.0, 1.0 - PV_70_pu ** 2)) * self.pv_max
        PV_107_pu = np.clip(self.pv_107_normalized.copy(), 0.0, 1.0)
        new_PV_107_Q = get_action[17] * np.sqrt(np.maximum(0.0, 1.0 - PV_107_pu ** 2)) * self.pv_max

        wind_11_avail = self.renew_profile['wind_11'][self.hour].copy()
        wind_32_avail = self.renew_profile['wind_32'][self.hour].copy()
        wind_66_avail = self.renew_profile['wind_66'][self.hour].copy()
        wind_101_avail = self.renew_profile['wind_101'][self.hour].copy()
        pv_22_avail = self.renew_profile['pv_22'][self.hour].copy()
        pv_36_avail = self.renew_profile['pv_36'][self.hour].copy()
        pv_70_avail = self.renew_profile['pv_70'][self.hour].copy()
        pv_107_avail = self.renew_profile['pv_107'][self.hour].copy()

        load = self.load_profile[self.hour].copy()
        load_Q = self.load_profile_Q[self.hour].copy()

        self.net.sgen.p_mw = np.hstack(([new_gen_29_p, new_gen_64_p],
                                        [0, 0, 0, 0],
                                        [new_BSS_12_p, new_BSS_34_p, new_BSS_68_p, new_BSS_103_p],
                                        [wind_11_avail, wind_32_avail, wind_66_avail, wind_101_avail],
                                        [pv_22_avail, pv_36_avail, pv_70_avail, pv_107_avail]))

        self.net.sgen.q_mvar = np.hstack(([0, 0],
                                          [new_SVC_20, new_SVC_56, new_SVC_79, new_SVC_105],
                                          [0, 0, 0, 0],
                                          [new_WT_11_Q, new_WT_32_Q, new_WT_66_Q, new_WT_101_Q],
                                          [new_PV_22_Q, new_PV_36_Q, new_PV_70_Q, new_PV_107_Q]))

        self.SOC_12 = new_SOC_12.copy()
        self.SOC_34 = new_SOC_34.copy()
        self.SOC_68 = new_SOC_68.copy()
        self.SOC_103 = new_SOC_103.copy()

        self.net.load.p_mw = load
        self.net.load.q_mvar = load_Q

        pp.runpp(self.net)

        total_gen = sum(self.net.sgen.p_mw)
        total_load = sum(self.net.load.p_mw)
        total_loss = self.net.res_line.pl_mw.sum()

        Price_high = 185
        Price_median = 123
        Price_low = 64
        if self.hour < 12:
            Price_current = Price_low
        elif self.hour >= 24 and self.hour < 36:
            Price_current = Price_high
        else:
            Price_current = Price_median
        balance_error = total_load + total_loss - total_gen
        balance_penalty = (Price_current * max(0, balance_error)) / 2

        p_29 = self.net.sgen.p_mw[0]
        cost_29 = (0.00240 * p_29 ** 2 + 12.3299 * p_29 + 0) / 2
        p_64 = self.net.sgen.p_mw[1]
        cost_64 = (0.00240 * p_64 ** 2 + 12.3299 * p_64 + 0) / 2
        cost = cost_29 + cost_64

        reward = - (cost + balance_penalty) / 100

        done = self.hour >= 47

        info = {
            'total_loss':total_loss,
            'sgen_p': self.net.sgen.p_mw.copy(),
            'sgen_q': self.net.sgen.q_mvar.copy(),
            'load_p': self.load_profile[self.hour].copy(),
            'total_sgen': sum(self.net.sgen.p_mw)
        }

        if not done:
            self.hour += 1

        return self._get_state(), reward, done, info

    def step_1(self, action):
        get_action = action[:NUM_CON]

        gen_29_delta = get_action[0] * self.gen_ramp / 2
        gen_64_delta = get_action[1] * self.gen_ramp / 2
        new_gen_29_p = self.net.sgen.p_mw[0] + gen_29_delta
        new_gen_64_p = self.net.sgen.p_mw[1] + gen_64_delta
        new_gen_29_p = np.clip(new_gen_29_p, self.gen_min, self.gen_max)
        new_gen_64_p = np.clip(new_gen_64_p, self.gen_min, self.gen_max)

        new_SVC_20 = get_action[2] * self.svc_scale
        new_SVC_56 = get_action[3] * self.svc_scale
        new_SVC_79 = get_action[4] * self.svc_scale
        new_SVC_105 = get_action[5] * self.svc_scale

        SOC_12_delta = get_action[6] * self.SOC_scale / 2
        SOC_34_delta = get_action[7] * self.SOC_scale / 2
        SOC_68_delta = get_action[8] * self.SOC_scale / 2
        SOC_103_delta = get_action[9] * self.SOC_scale / 2
        new_SOC_12 = self.SOC_12 + SOC_12_delta
        new_SOC_34 = self.SOC_34 + SOC_34_delta
        new_SOC_68 = self.SOC_68 + SOC_68_delta
        new_SOC_103 = self.SOC_103 + SOC_103_delta
        new_SOC_12 = np.clip(new_SOC_12, self.SOC_min, self.SOC_max)
        new_SOC_34 = np.clip(new_SOC_34, self.SOC_min, self.SOC_max)
        new_SOC_68 = np.clip(new_SOC_68, self.SOC_min, self.SOC_max)
        new_SOC_103 = np.clip(new_SOC_103, self.SOC_min, self.SOC_max)

        if self.SOC_12 - new_SOC_12 >= 0:
            new_BSS_12_p = (self.SOC_12 - new_SOC_12) * 0.98 * 2
        else:
            new_BSS_12_p = (self.SOC_12 - new_SOC_12) * 1.02 * 2
        if self.SOC_34 - new_SOC_34 >= 0:
            new_BSS_34_p = (self.SOC_34 - new_SOC_34) * 0.98 * 2
        else:
            new_BSS_34_p = (self.SOC_34 - new_SOC_34) * 1.02 * 2
        if self.SOC_68 - new_SOC_68 >= 0:
            new_BSS_68_p = (self.SOC_68 - new_SOC_68) * 0.98 * 2
        else:
            new_BSS_68_p = (self.SOC_68 - new_SOC_68) * 1.02 * 2
        if self.SOC_103 - new_SOC_103 >= 0:
            new_BSS_103_p = (self.SOC_103 - new_SOC_103) * 0.98 * 2
        else:
            new_BSS_103_p = (self.SOC_103 - new_SOC_103) * 1.02 * 2

        WT_11_pu = np.clip(self.wind_11_normalized.copy(), 0.0, 1.0)
        new_WT_11_Q = get_action[10] * np.sqrt(np.maximum(0.0, 1.0 - WT_11_pu ** 2)) * self.wind_max
        WT_32_pu = np.clip(self.wind_32_normalized.copy(), 0.0, 1.0)
        new_WT_32_Q = get_action[11] * np.sqrt(np.maximum(0.0, 1.0 - WT_32_pu ** 2)) * self.wind_max
        WT_66_pu = np.clip(self.wind_66_normalized.copy(), 0.0, 1.0)
        new_WT_66_Q = get_action[12] * np.sqrt(np.maximum(0.0, 1.0 - WT_66_pu ** 2)) * self.wind_max
        WT_101_pu = np.clip(self.wind_101_normalized.copy(), 0.0, 1.0)
        new_WT_101_Q = get_action[13] * np.sqrt(np.maximum(0.0, 1.0 - WT_101_pu ** 2)) * self.wind_max

        PV_22_pu = np.clip(self.pv_22_normalized.copy(), 0.0, 1.0)
        new_PV_22_Q = get_action[14] * np.sqrt(np.maximum(0.0, 1.0 - PV_22_pu ** 2)) * self.pv_max
        PV_36_pu = np.clip(self.pv_36_normalized.copy(), 0.0, 1.0)
        new_PV_36_Q = get_action[15] * np.sqrt(np.maximum(0.0, 1.0 - PV_36_pu ** 2)) * self.pv_max
        PV_70_pu = np.clip(self.pv_70_normalized.copy(), 0.0, 1.0)
        new_PV_70_Q = get_action[16] * np.sqrt(np.maximum(0.0, 1.0 - PV_70_pu ** 2)) * self.pv_max
        PV_107_pu = np.clip(self.pv_107_normalized.copy(), 0.0, 1.0)
        new_PV_107_Q = get_action[17] * np.sqrt(np.maximum(0.0, 1.0 - PV_107_pu ** 2)) * self.pv_max

        wind_11_avail = self.renew_profile['wind_11'][self.hour].copy()
        wind_32_avail = self.renew_profile['wind_32'][self.hour].copy()
        wind_66_avail = self.renew_profile['wind_66'][self.hour].copy()
        wind_101_avail = self.renew_profile['wind_101'][self.hour].copy()
        pv_22_avail = self.renew_profile['pv_22'][self.hour].copy()
        pv_36_avail = self.renew_profile['pv_36'][self.hour].copy()
        pv_70_avail = self.renew_profile['pv_70'][self.hour].copy()
        pv_107_avail = self.renew_profile['pv_107'][self.hour].copy()

        load = self.load_profile[self.hour].copy()
        load_Q = self.load_profile_Q[self.hour].copy()

        self.net.sgen.p_mw = np.hstack(([new_gen_29_p, new_gen_64_p],
                                        [0, 0, 0, 0],
                                        [new_BSS_12_p, new_BSS_34_p, new_BSS_68_p, new_BSS_103_p],
                                        [wind_11_avail, wind_32_avail, wind_66_avail, wind_101_avail],
                                        [pv_22_avail, pv_36_avail, pv_70_avail, pv_107_avail]))

        self.net.sgen.q_mvar = np.hstack(([0, 0],
                                          [new_SVC_20, new_SVC_56, new_SVC_79, new_SVC_105],
                                          [0, 0, 0, 0],
                                          [new_WT_11_Q, new_WT_32_Q, new_WT_66_Q, new_WT_101_Q],
                                          [new_PV_22_Q, new_PV_36_Q, new_PV_70_Q, new_PV_107_Q]))

        self.SOC_12 = new_SOC_12.copy()
        self.SOC_34 = new_SOC_34.copy()
        self.SOC_68 = new_SOC_68.copy()
        self.SOC_103 = new_SOC_103.copy()

        self.net.load.p_mw = load
        self.net.load.q_mvar = load_Q

        pp.runpp(self.net)

        list_1 = list(self.net.res_bus['vm_pu'].copy())

        cost_c = 0
        for i in range(len(list_1)):
            if list_1[i] > 1.06:
                cost_c += list_1[i] - 1.06
            elif list_1[i] < 0.94:
                cost_c += 0.94 - list_1[i]

        total_gen = sum(self.net.sgen.p_mw)
        total_load = sum(self.net.load.p_mw)
        total_loss = self.net.res_line.pl_mw.sum()

        Price_high = 185
        Price_median = 123
        Price_low = 64
        if self.hour < 12:
            Price_current = Price_low
        elif self.hour >= 24 and self.hour < 36:
            Price_current = Price_high
        else:
            Price_current = Price_median
        balance_error = total_load + total_loss - total_gen
        balance_penalty = (Price_current * max(0, balance_error)) / 2

        p_29 = self.net.sgen.p_mw[0]
        cost_29 = (0.00240 * p_29 ** 2 + 12.3299 * p_29 + 0) / 2
        p_64 = self.net.sgen.p_mw[1]
        cost_64 = (0.00240 * p_64 ** 2 + 12.3299 * p_64 + 0) / 2
        cost = cost_29 + cost_64

        reward = - (cost + balance_penalty) / 100

        done = self.hour >= 47

        info = {
            'total_loss': total_loss,
            'sgen_p': self.net.sgen.p_mw.copy(),
            'sgen_q': self.net.sgen.q_mvar.copy(),
            'load_p': self.load_profile[self.hour].copy(),
            'total_sgen': sum(self.net.sgen.p_mw)
        }

        if not done:
            self.hour += 1

        return self._get_state(), reward, cost_c, done, info


class PolicyNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, num_con):
        super().__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)

        self.fc_con_mu = nn.Linear(hidden_dim, num_con)
        self.fc_con_std = nn.Linear(hidden_dim, num_con)

    def forward(self, x, stochastic=True):
        x = F.relu(self.fc1(x))

        con_mu = self.fc_con_mu(x)

        con_std = F.softplus(self.fc_con_std(x))

        con_dist = Normal(con_mu, con_std)
        if not stochastic:
            con_sample = con_mu
        else:
            con_sample = con_dist.rsample()

        con_action = torch.tanh(con_sample)

        con_log_prob = con_dist.log_prob(con_sample) - torch.log(1 - con_action.pow(2) + 1e-6)

        log_prob = con_log_prob.sum(dim=1, keepdim=True)

        return con_action, log_prob


class QValueNetContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(QValueNetContinuous, self).__init__()

        layers = (state_dim+action_dim, hidden_dim, 1)
        weight_dims = list(zip(layers[1:], layers))
        self.As = nn.ParameterList()
        self.Ws = nn.ParameterList()
        self.bs = nn.ParameterList()
        first_idim = weight_dims[0][1]
        self.layers = layers
        self.activ_id = "relu"

        self.activ = nn.ReLU()
        for odim, idim in weight_dims:
            self.As.append(nn.Parameter(torch.tensor(np.random.normal(size=(odim, first_idim)),dtype=torch.float)))
            self.Ws.append(
                nn.Parameter(torch.tensor(np.random.uniform(size=(odim, idim), low=0, high=1),dtype=torch.float)))
            self.bs.append(nn.Parameter(torch.tensor(np.random.normal(size=(odim,)),dtype=torch.float)))

    def forward(self, x, a):
        z=torch.cat([x, a], dim=1)
        z0 = z.clone()
        layers = list(zip(self.As, self.Ws, self.bs))
        for (A, W, b) in layers[:-1]:
            z = self.activ(z0 @ torch.t(A) + z @ torch.t(W) + b)
        out_A, out_W, out_b = layers[-1]
        z = z0 @ torch.t(out_A) + z @ torch.t(out_W) + out_b
        return z

    def project_ws(self):
        for w in self.Ws:
            w.data = torch.clamp(w.data, 0, np.inf)


class SACContinuous:
    def __init__(self, state_dim, hidden_dim, action_dim,
                 actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
                 device):
        self.actor = PolicyNetContinuous(state_dim, hidden_dim, num_con=NUM_CON
                                         ).to(device)
        self.critic_1 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)
        self.critic_2 = QValueNetContinuous(state_dim, hidden_dim,
                                            action_dim).to(device)

        self.target_critic_1 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)
        self.target_critic_2 = QValueNetContinuous(state_dim,
                                                   hidden_dim, action_dim).to(
                                                       device)

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(),
                                                   lr=critic_lr)
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(),
                                                   lr=critic_lr)

        self.log_alpha = torch.tensor(np.log(0.01), dtype=torch.float)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha],
                                                    lr=alpha_lr)
        self.target_entropy = target_entropy
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def take_action(self, state, stochastic=True):
        state = torch.tensor([state], dtype=torch.float).to(self.device)

        action, _ = self.actor(state,stochastic=stochastic)

        return action.cpu().detach().numpy()[0]

    def calc_target(self, rewards, next_states, dones):
        next_actions, log_prob = self.actor(next_states)
        entropy = -log_prob
        q1_value = self.target_critic_1(next_states, next_actions)
        q2_value = self.target_critic_2(next_states, next_actions)
        next_value = torch.min(q1_value,
                               q2_value) + self.log_alpha.exp() * entropy
        td_target = rewards + self.gamma * next_value * (1 - dones)

        return td_target

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(),
                                       net.parameters()):
            param_target.data.copy_(param_target.data * (1.0 - self.tau) +
                                    param.data * self.tau)

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],
                               dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        td_target = self.calc_target(rewards, next_states, dones)

        critic_1_loss = torch.mean(
            F.mse_loss(self.critic_1(states, actions), td_target.detach()))
        critic_2_loss = torch.mean(
            F.mse_loss(self.critic_2(states, actions), td_target.detach()))

        self.critic_1_optimizer.zero_grad()

        critic_1_loss.backward()

        self.critic_1_optimizer.step()

        self.critic_1.project_ws()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        self.critic_2.project_ws()

        new_actions, log_prob = self.actor(states)
        entropy = -log_prob
        q1_value = self.critic_1(states, new_actions)
        q2_value = self.critic_2(states, new_actions)

        actor_loss = torch.mean(-self.log_alpha.exp() * entropy -
                                torch.min(q1_value, q2_value))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = torch.mean(
            (entropy - self.target_entropy).detach() * self.log_alpha.exp())
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)


def _test_agent(env, agent, list_feng_11, list_feng_32, list_feng_66, list_feng_101,
                list_pv_22, list_pv_36, list_pv_70, list_pv_107,
                list_fu_he, list_fu_he_Q, model_path='model.pth'):
    agent.actor.load_state_dict(torch.load(model_path))

    agent.actor.eval()

    total_cost = 0
    t1 = 0

    state = env.new_reset(list_feng_11, list_feng_32, list_feng_66, list_feng_101,
                        list_pv_22, list_pv_36, list_pv_70, list_pv_107,
                        list_fu_he, list_fu_he_Q)

    done = False

    start_time = time.time()

    while not done:
        action = agent.take_action(state, stochastic=True)
        next_state, reward, done, info = env.step(action)

        cost = - reward

        total_cost += cost

        state = next_state

        t1 += 1

    print("", total_cost)

    end_time = time.time()
    print("Time taken:", end_time - start_time, "seconds")


def _test_agent_1(env, agent, list_feng_11, list_feng_32, list_feng_66, list_feng_101,
                list_pv_22, list_pv_36, list_pv_70, list_pv_107,
                list_fu_he, list_fu_he_Q, model_path='model.pth'):
    agent.actor.load_state_dict(torch.load(model_path))

    agent.actor.eval()

    total_cost = 0
    t1 = 0

    total_cost_c = 0

    state = env.new_reset(list_feng_11, list_feng_32, list_feng_66, list_feng_101,
                        list_pv_22, list_pv_36, list_pv_70, list_pv_107,
                        list_fu_he, list_fu_he_Q)

    done = False

    start_time = time.time()

    while not done:
        action = agent.take_action(state, stochastic=True)

        next_state, reward, cost_c, done, info = env.step_1(action)

        cost = - reward

        total_cost += cost

        total_cost_c += cost_c

        state = next_state

        t1 += 1

    print("", total_cost * 100)
    print("C", total_cost_c)

    end_time = time.time()
    print("Time taken:", end_time - start_time, "seconds")

env = PowerSystemEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

actor_lr = 3e-4
critic_lr = 3e-3
alpha_lr = 3e-4
num_episodes = 20000

hidden_dim = 128
gamma = 0.99
tau = 0.005
buffer_size = 100000
minimal_size = 1000
batch_size = 64

target_entropy = -env.action_space.shape[0]

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cpu")
print(device)

replay_buffer = rl_utils.ReplayBuffer(buffer_size)
agent = SACContinuous(state_dim, hidden_dim, action_dim,
                      actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                      gamma, device)

seedlist = [0,5,7,9,14,17,18,19,20,21,22,24,25,26,28,29,30,33,34,35,37,38,42,43,44,47,48,49,50,52,56,57,58,60,61,62,66,68,71,72,73,75,76,80,81,83,85,88,89,91]

for seed in seedlist:
    print("seed:", seed)
    load_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "119"))
    load_path = os.path.join(load_dir, "results_{}.npz".format(seed))

    data = np.load(load_path)

    list_feng_11 = data["wind_11"]
    list_feng_32 = data["wind_32"]
    list_feng_66 = data["wind_66"]
    list_feng_101 = data["wind_101"]
    list_pv_22 = data["pv_22"]
    list_pv_36 = data["pv_36"]
    list_pv_70 = data["pv_70"]
    list_pv_107 = data["pv_107"]
    list_fu_he = data["load"]
    list_fu_he_Q = data["load_Q"]

    _test_agent_1(env, agent, list_feng_11, list_feng_32, list_feng_66, list_feng_101,
                  list_pv_22, list_pv_36, list_pv_70, list_pv_107,
                  list_fu_he, list_fu_he_Q)

kk = -1

if kk == 0:
    list_feng_11 = np.array([1.2015582314483295, 1.6289121946824336, 1.5853376612749095, 1.544623139630153, 1.5893201592495778, 1.5963203388007097, 1.6581304463251083, 1.5635644349830977, 1.4871270602756694, 1.5548239307964755, 1.544473752749698, 1.3260475351260994, 1.6156632115609058, 1.3574558333096083, 1.499473826754254, 1.5553549590122644, 1.4968821470582867, 1.596410961379744, 1.6360902220705158, 1.3770249851852983, 1.4835213676380445, 1.594064418219532, 1.4182156280915719, 1.6592033226837666])
    list_feng_101 = np.array([1.6522857402166602, 1.406878121565222, 1.5574312892798785, 1.4959963513484942, 1.4788435174605195, 1.3832873499119545, 1.5503550725259791, 1.6291286905154885, 1.611774884062957, 1.4814151895238035, 1.4123253699227256, 1.5124057188292266, 1.445142649356595, 1.456656093066398, 1.5097506126777431, 1.4922322961318977, 1.5978728370157596, 1.6275248008552066, 1.3967979548059095, 1.5055893237533282, 1.567085598971824, 1.4190972970735116, 1.4940088937437308, 1.5321059511379662])
    list_feng_32 = np.array([1.0524072018657056, 1.0058708370549598, 1.029741924728592, 1.0111863914004278, 1.018011948218538, 0.9144720616808762, 0.9445885914549832, 1.0254825163239993, 0.961068791210204, 1.110437530569681, 0.9326621586859364, 1.0357708004261386, 0.9781691964012871, 0.9125683129666573, 1.0281770279878726, 0.9735630827473793, 0.9192672642546793, 0.9351075793311214, 1.1001420041751926, 0.9790400079510208, 1.0741389489116566, 1.069531559412505, 1.1039040217685465, 1.021614777466901])
    list_feng_66 = np.array([0.9464656114052233, 1.0478371313814194, 0.8508773026532637, 1.091714138005791, 1.0801948309883855, 0.918032160100247, 1.0880010585006485, 0.9152623020058124, 0.9948934703672401, 0.9710158898155785, 0.9961214568141774, 0.9460427546773262, 1.0193710605956467, 0.9488493633858236, 1.0038002639800152, 0.9998148343655697, 0.9133816691681169, 1.034047000176895, 0.8568535177895777, 1.116508455563514, 1.0024601925926202, 0.953070965575547, 1.0352741645005654, 1.0379074479290082])
    list_fu_he = np.array([
        [0.        , 0.03308731, 0.00534764, 0.00737528, 0.02299429,
       0.03595899, 0.02907057, 0.00854553, 0.02896715, 0.04724294,
       0.03637039, 0.00711512, 0.01725221, 0.03310925, 0.00460969,
       0.00973943, 0.00822898, 0.0043284 , 0.03850256, 0.15158943,
       0.05853304, 0.02538678, 0.0266929 , 0.03692311, 0.03982839,
       0.00368518, 0.00560488, 0.15335844, 0.03995674, 0.02868763,
       0.12461774, 0.13972013, 0.04972403, 0.05134542, 0.03956701,
       0.12047271, 0.1460516 , 0.02702719, 0.01745146, 0.11188498,
       0.0980035 , 0.1336772 , 0.01676536, 0.01396333, 0.01119814,
       0.00803011, 0.01544812, 0.02346903, 0.03737325, 0.24286656,
       0.05478898, 0.01498026, 0.01016596, 0.10849823, 0.01311566,
       0.02165511, 0.02723324, 0.10943606, 0.00606376, 0.02424227,
       0.02737885, 0.01458613, 0.14949087, 0.02917364, 0.04617306,
       0.08410236, 0.00675326, 0.01510897, 0.01714675, 0.11684761,
       0.12279696, 0.03731188, 0.01407542, 0.19822375, 0.0098502 ,
       0.04662629, 0.01568527, 0.05895839, 0.06771872, 0.11072466,
       0.0504134 , 0.05001101, 0.02896292, 0.00480415, 0.01529674,
       0.10946197, 0.0140893 , 0.00647915, 0.01958023, 0.00748462,
       0.01360273, 0.03811282, 0.02659222, 0.00676736, 0.00843558,
       0.15299481, 0.14299603, 0.00577578, 0.01064419, 0.02556735,
       0.12775085, 0.17128421, 0.11350572, 0.03737052, 0.02458727,
       0.02213932, 0.09922383, 0.06058682, 0.1531325 , 0.06025351,
       0.28841213, 0.07747678, 0.01425127, 0.05102482, 0.01655218,
       0.05118332, 0.01051442, 0.00771179],[0.        , 0.04290021, 0.00489246, 0.00820728, 0.01540149,
       0.03873154, 0.02969453, 0.00649489, 0.02200477, 0.04474132,
       0.04103852, 0.00662962, 0.01305587, 0.03607798, 0.0066344 ,
       0.00735805, 0.00768251, 0.00503293, 0.04414096, 0.16682614,
       0.03960489, 0.02169266, 0.02034409, 0.05061198, 0.0345394 ,
       0.00351398, 0.00629724, 0.12022678, 0.03107736, 0.02757531,
       0.15911088, 0.11470183, 0.03610555, 0.05249204, 0.04034577,
       0.12123019, 0.09524291, 0.0284244 , 0.0161239 , 0.11443287,
       0.10686103, 0.12230357, 0.0177615 , 0.01106455, 0.0093967 ,
       0.01070161, 0.01928216, 0.01885715, 0.03319443, 0.24373528,
       0.05600538, 0.01591433, 0.01356315, 0.12791344, 0.01469972,
       0.02576046, 0.0270397 , 0.08654971, 0.00641464, 0.01702416,
       0.02503354, 0.0184921 , 0.13653301, 0.02969272, 0.03490851,
       0.09433824, 0.00669384, 0.01547529, 0.01725038, 0.11046181,
       0.13234806, 0.02989218, 0.01213193, 0.25709894, 0.00848632,
       0.04208256, 0.01683051, 0.06404236, 0.07606154, 0.14031993,
       0.06415975, 0.06469961, 0.03656554, 0.00664803, 0.01207516,
       0.09254242, 0.01426531, 0.0066179 , 0.01576879, 0.00849256,
       0.01764582, 0.02789723, 0.01992998, 0.00975312, 0.00747295,
       0.13285365, 0.14759837, 0.00622466, 0.01378898, 0.02996963,
       0.13451563, 0.15042448, 0.09892177, 0.03382575, 0.02590695,
       0.02478941, 0.15760751, 0.06468345, 0.1307011 , 0.04568035,
       0.23233249, 0.09696953, 0.01224141, 0.05767405, 0.01930667,
       0.04467998, 0.01228979, 0.00798959],[0.        , 0.03571072, 0.00462663, 0.00924378, 0.0166629 ,
       0.03891136, 0.02453443, 0.00732554, 0.02137006, 0.05405517,
       0.03774528, 0.00741996, 0.01538227, 0.03998992, 0.00578434,
       0.00758505, 0.00896416, 0.00531894, 0.04353197, 0.15984351,
       0.05492638, 0.0268588 , 0.02482542, 0.04449346, 0.03077517,
       0.00420611, 0.00633139, 0.16640232, 0.03692877, 0.0262144 ,
       0.13907125, 0.12475101, 0.03534555, 0.05581713, 0.03176815,
       0.13214634, 0.10403842, 0.03081725, 0.01389587, 0.10814191,
       0.08107002, 0.1442887 , 0.02211138, 0.01541497, 0.01120795,
       0.0118373 , 0.01760377, 0.02168672, 0.02649808, 0.23706208,
       0.0616561 , 0.01819069, 0.01124043, 0.1180932 , 0.01585877,
       0.02856366, 0.02046607, 0.08067119, 0.00505515, 0.02117236,
       0.02171709, 0.01654456, 0.12885927, 0.03459101, 0.03368766,
       0.09713797, 0.00727064, 0.01341654, 0.01794371, 0.13450349,
       0.16984995, 0.03073426, 0.0156291 , 0.27568097, 0.00825469,
       0.04764134, 0.01663453, 0.07283884, 0.07994959, 0.13259113,
       0.05249759, 0.07456872, 0.03485957, 0.00490082, 0.01323853,
       0.11690605, 0.01231449, 0.00686796, 0.01642608, 0.00878197,
       0.01565933, 0.02884371, 0.02303395, 0.00910821, 0.00866563,
       0.14620651, 0.15180492, 0.00632953, 0.01060682, 0.02871795,
       0.13728554, 0.13552661, 0.11470551, 0.03629415, 0.02803246,
       0.02724328, 0.1212256 , 0.0646083 , 0.14634512, 0.05217243,
       0.25506415, 0.08601263, 0.01649029, 0.05885412, 0.02036856,
       0.04281409, 0.01325881, 0.00902007],[0.        , 0.03555554, 0.0046951 , 0.0090827 , 0.01727388,
       0.03852861, 0.02755152, 0.00831905, 0.02033213, 0.04953037,
       0.04391446, 0.00756076, 0.01335071, 0.03876803, 0.00502464,
       0.01063133, 0.00919174, 0.00579007, 0.03809955, 0.15671425,
       0.04726164, 0.02661618, 0.01938719, 0.04421191, 0.03133757,
       0.00422146, 0.00679675, 0.13795099, 0.03249022, 0.02652342,
       0.15014577, 0.13281272, 0.04405947, 0.05209508, 0.03300254,
       0.10179094, 0.10177871, 0.0320364 , 0.01413364, 0.10671804,
       0.09308248, 0.12900475, 0.02323439, 0.01430881, 0.01120551,
       0.00949572, 0.01437207, 0.01687544, 0.02921964, 0.27356166,
       0.05564578, 0.01656884, 0.01159954, 0.11413087, 0.01640396,
       0.02178443, 0.02345632, 0.08826084, 0.00578944, 0.02601543,
       0.02907882, 0.01832598, 0.13568567, 0.03361508, 0.0364863 ,
       0.1171142 , 0.0070109 , 0.01442091, 0.01562996, 0.14623607,
       0.15365289, 0.03199465, 0.01461547, 0.23562658, 0.00807024,
       0.05476193, 0.01934432, 0.06742755, 0.09173566, 0.1354845 ,
       0.07114993, 0.06502944, 0.03534967, 0.00664705, 0.0135173 ,
       0.09940304, 0.0143279 , 0.00562274, 0.01496864, 0.00908215,
       0.01626597, 0.03095068, 0.02255332, 0.00853452, 0.00691863,
       0.14317865, 0.13709117, 0.00758917, 0.01219447, 0.02573964,
       0.12682223, 0.16619842, 0.11155936, 0.04133004, 0.03154384,
       0.02540861, 0.14757866, 0.06040584, 0.14755547, 0.05803954,
       0.24273717, 0.07191701, 0.01513993, 0.05772609, 0.01629806,
       0.04236497, 0.01177029, 0.00865305],[0.        , 0.04046196, 0.00413342, 0.00795637, 0.01678955,
       0.03601407, 0.02941362, 0.00875292, 0.02029751, 0.04933935,
       0.04580126, 0.00696626, 0.01453369, 0.03673052, 0.00604973,
       0.01047843, 0.0079685 , 0.00543179, 0.04005336, 0.16752892,
       0.04460486, 0.02324467, 0.02164337, 0.04310889, 0.03321585,
       0.00441413, 0.00620488, 0.16819528, 0.03014053, 0.02478698,
       0.14247734, 0.11451672, 0.03343806, 0.06208746, 0.03396139,
       0.14482453, 0.10660293, 0.03360953, 0.01147707, 0.1081374 ,
       0.0819003 , 0.12949576, 0.02106399, 0.01328388, 0.012632  ,
       0.012122  , 0.01725912, 0.01815724, 0.03014932, 0.25944303,
       0.05551633, 0.02142639, 0.0123486 , 0.11115305, 0.01833324,
       0.02448256, 0.02084239, 0.08141407, 0.00585477, 0.02154011,
       0.02491827, 0.01668221, 0.13058945, 0.02989482, 0.03337014,
       0.09881188, 0.00818032, 0.0137004 , 0.01828208, 0.13074975,
       0.15258436, 0.03995358, 0.01228806, 0.22277008, 0.00798708,
       0.04998388, 0.01776501, 0.07155219, 0.07734605, 0.11312826,
       0.06791651, 0.07423383, 0.036245  , 0.00596864, 0.01496741,
       0.0948874 , 0.01476978, 0.00592993, 0.01920949, 0.0076612 ,
       0.01587543, 0.03344922, 0.02275317, 0.00787964, 0.0098659 ,
       0.16876922, 0.11140075, 0.00590233, 0.01067961, 0.02633385,
       0.11499629, 0.14815479, 0.1097389 , 0.03459243, 0.02736818,
       0.02365101, 0.15229131, 0.06532157, 0.138572  , 0.05327948,
       0.26890928, 0.07469749, 0.01559137, 0.04955026, 0.01665629,
       0.04090802, 0.01291871, 0.00724419],[0.        , 0.03353951, 0.00438269, 0.00919437, 0.01842165,
       0.04333257, 0.02760069, 0.00706571, 0.02359678, 0.05414293,
       0.03634338, 0.00766647, 0.01342881, 0.04165392, 0.00558465,
       0.00970789, 0.00922572, 0.00530104, 0.04769022, 0.131863  ,
       0.04827315, 0.01972915, 0.02150789, 0.04656718, 0.03249391,
       0.00467985, 0.00690358, 0.1660589 , 0.03442753, 0.02889321,
       0.14232196, 0.12620977, 0.04337949, 0.05556838, 0.03403436,
       0.11978912, 0.12239595, 0.02557511, 0.01486108, 0.11147002,
       0.09417818, 0.15004085, 0.01992904, 0.01475463, 0.00931087,
       0.0093025 , 0.01852112, 0.02206087, 0.03118176, 0.21218925,
       0.05341953, 0.01831857, 0.01010056, 0.121822  , 0.01700092,
       0.02371069, 0.02484212, 0.08477986, 0.00574829, 0.02053557,
       0.02447998, 0.01622204, 0.11838002, 0.03094048, 0.0386778 ,
       0.09799405, 0.00762238, 0.01310658, 0.02004955, 0.10777226,
       0.15098224, 0.03588116, 0.01442854, 0.23780156, 0.00704023,
       0.04428078, 0.01804541, 0.06165348, 0.07792264, 0.1349584 ,
       0.06691497, 0.06088483, 0.03577756, 0.00530732, 0.01372091,
       0.10179396, 0.01194442, 0.00622089, 0.01596395, 0.00810862,
       0.01676895, 0.03344533, 0.02001042, 0.00885017, 0.00772732,
       0.1330693 , 0.1357859 , 0.00721363, 0.01359616, 0.03048583,
       0.14021589, 0.14511627, 0.11753724, 0.035427  , 0.02461525,
       0.02567241, 0.11864474, 0.06357701, 0.13668932, 0.0430124 ,
       0.27139106, 0.08724121, 0.01779406, 0.05558461, 0.01661861,
       0.04131296, 0.01302446, 0.00955807],[0.        , 0.04461958, 0.00527259, 0.01185061, 0.02490666,
       0.04979339, 0.03773983, 0.00956223, 0.03165394, 0.06891125,
       0.04736104, 0.00880659, 0.01704964, 0.04662131, 0.00771343,
       0.01025606, 0.01110417, 0.00665391, 0.05876971, 0.1924338 ,
       0.058914  , 0.03005325, 0.02706484, 0.06051019, 0.04305341,
       0.00547863, 0.00823962, 0.19485109, 0.04093505, 0.03721239,
       0.16651512, 0.16053294, 0.0493756 , 0.06739161, 0.04446422,
       0.14806282, 0.1327931 , 0.04061724, 0.01793485, 0.11955593,
       0.1167085 , 0.1556122 , 0.029159  , 0.02039657, 0.01372289,
       0.01383716, 0.02291413, 0.02412565, 0.03716677, 0.3023301 ,
       0.07277263, 0.02065534, 0.01513994, 0.14596579, 0.01931195,
       0.03015086, 0.02658745, 0.10430516, 0.00663454, 0.02798011,
       0.03314241, 0.01970701, 0.1450317 , 0.03661003, 0.04436285,
       0.12487599, 0.00924548, 0.01699446, 0.02216922, 0.15838809,
       0.21710471, 0.0416417 , 0.01800219, 0.30192507, 0.00929841,
       0.06151957, 0.02313291, 0.07679408, 0.10621081, 0.1481055 ,
       0.08187861, 0.08384461, 0.05136775, 0.0083863 , 0.01634018,
       0.12106431, 0.01672714, 0.00832171, 0.01999625, 0.00920883,
       0.02124158, 0.04089577, 0.02493148, 0.01087456, 0.0104791 ,
       0.18609211, 0.16112001, 0.00959648, 0.01548861, 0.03379273,
       0.14763293, 0.18730378, 0.14309573, 0.04913   , 0.03636405,
       0.02979061, 0.16892916, 0.07855603, 0.16499607, 0.07084461,
       0.29594594, 0.1096352 , 0.01752505, 0.06714429, 0.02021374,
       0.06071038, 0.01726886, 0.0113363 ],[0.        , 0.04020464, 0.0055856 , 0.01149953, 0.02523937,
       0.04792815, 0.03516888, 0.00862215, 0.0285765 , 0.06707036,
       0.05052665, 0.00902121, 0.01803015, 0.04542376, 0.00714417,
       0.01057196, 0.01067539, 0.00669915, 0.05207638, 0.1767666 ,
       0.06197613, 0.02943793, 0.03055526, 0.05636289, 0.04033569,
       0.00521379, 0.00916563, 0.214252  , 0.04324164, 0.03548253,
       0.18797885, 0.17123964, 0.04852555, 0.06576616, 0.038642  ,
       0.14100708, 0.13841531, 0.03958589, 0.01630913, 0.14541924,
       0.10969223, 0.17696526, 0.02599171, 0.0181664 , 0.0142245 ,
       0.01190786, 0.02143981, 0.02267389, 0.03516255, 0.29732871,
       0.0747052 , 0.02257402, 0.01355608, 0.15338503, 0.01965855,
       0.03125423, 0.02885537, 0.12286782, 0.00775457, 0.02556603,
       0.03168868, 0.02044981, 0.13979959, 0.04124147, 0.04622694,
       0.13977487, 0.00922757, 0.01740372, 0.02082992, 0.17753313,
       0.17883804, 0.0438256 , 0.01649114, 0.30292417, 0.00955845,
       0.06769811, 0.02016153, 0.07701358, 0.10268932, 0.17867631,
       0.08101038, 0.07730853, 0.0465418 , 0.00691493, 0.01470536,
       0.13574872, 0.01594369, 0.00792526, 0.01944606, 0.00920519,
       0.01887437, 0.03930381, 0.02769671, 0.00998924, 0.01218254,
       0.18620528, 0.1774477 , 0.00833645, 0.01566467, 0.03445257,
       0.1545026 , 0.15832291, 0.13055286, 0.04822185, 0.03309891,
       0.03214439, 0.17192319, 0.07507527, 0.15731547, 0.05897307,
       0.30369571, 0.1125355 , 0.01638677, 0.07387537, 0.0226657 ,
       0.06186352, 0.01634897, 0.01079712],[0.        , 0.05168102, 0.00563334, 0.01030408, 0.02350414,
       0.04735533, 0.03638097, 0.00993171, 0.02851999, 0.07259226,
       0.04507243, 0.00866987, 0.01724092, 0.05010755, 0.00679091,
       0.01171215, 0.01048439, 0.00679213, 0.04629084, 0.16532701,
       0.05754541, 0.03058206, 0.03117882, 0.05788486, 0.04113838,
       0.00518284, 0.00874767, 0.18304744, 0.04089728, 0.03189133,
       0.18377091, 0.15696244, 0.05323162, 0.07257155, 0.04162559,
       0.13592648, 0.16254046, 0.03404558, 0.0178106 , 0.14322762,
       0.11632417, 0.17657634, 0.02590524, 0.0180873 , 0.01177383,
       0.0145705 , 0.02003472, 0.02324174, 0.03836797, 0.29266504,
       0.05908998, 0.02006193, 0.01422451, 0.13925847, 0.02085549,
       0.03007024, 0.02800264, 0.10791014, 0.00735163, 0.02668581,
       0.03221188, 0.0213448 , 0.1583796 , 0.03407223, 0.04915963,
       0.14149751, 0.00797213, 0.01619716, 0.02095809, 0.16370475,
       0.19103222, 0.04876642, 0.01600938, 0.30046563, 0.01100313,
       0.06141942, 0.02126695, 0.07869538, 0.10712327, 0.1638826 ,
       0.0868816 , 0.08168813, 0.04643837, 0.00735024, 0.0174573 ,
       0.1246421 , 0.01715083, 0.00752217, 0.02149373, 0.0109366 ,
       0.01858485, 0.03814088, 0.02534804, 0.01064873, 0.01002184,
       0.17632149, 0.17611963, 0.0087108 , 0.01622341, 0.03527879,
       0.14349777, 0.18121732, 0.14592204, 0.04889736, 0.03986536,
       0.0306617 , 0.18301437, 0.08065786, 0.15933691, 0.05908287,
       0.33638591, 0.0980878 , 0.01813458, 0.06937576, 0.0212231 ,
       0.04881737, 0.01739933, 0.01240803],[0.        , 0.04713177, 0.00524261, 0.01124317, 0.02598341,
       0.04994376, 0.03078096, 0.01026657, 0.0300874 , 0.06591928,
       0.05135905, 0.00959888, 0.01726093, 0.05092305, 0.00783772,
       0.01052137, 0.01074759, 0.00616871, 0.05129841, 0.1937079 ,
       0.06733321, 0.02773818, 0.02718423, 0.05671686, 0.04278512,
       0.00509042, 0.00877491, 0.1894337 , 0.0404066 , 0.03564905,
       0.16514681, 0.14238467, 0.0484062 , 0.06924714, 0.04431707,
       0.1581722 , 0.14822054, 0.03452531, 0.01881512, 0.14185279,
       0.11778427, 0.18845619, 0.02398049, 0.01748491, 0.01349339,
       0.01378   , 0.02248393, 0.02535143, 0.03746865, 0.31372832,
       0.07245928, 0.02070592, 0.0140731 , 0.13419637, 0.01836575,
       0.03284289, 0.02975437, 0.10780765, 0.00757817, 0.02954081,
       0.02959822, 0.02149695, 0.13948996, 0.0410514 , 0.04797775,
       0.13055473, 0.00992825, 0.01753758, 0.0229922 , 0.15731762,
       0.18250972, 0.04343536, 0.01570653, 0.2810742 , 0.01174113,
       0.05578962, 0.02249619, 0.07137817, 0.09766041, 0.16297889,
       0.0869775 , 0.07658921, 0.04125   , 0.00855344, 0.01547129,
       0.1291175 , 0.01583995, 0.00856353, 0.01962236, 0.01037524,
       0.02077205, 0.0400852 , 0.02735847, 0.01203343, 0.01198705,
       0.1791225 , 0.17882825, 0.00852486, 0.01502833, 0.03750755,
       0.14327569, 0.18078739, 0.12786253, 0.04738411, 0.03488449,
       0.03356648, 0.1665114 , 0.07604453, 0.16823692, 0.05838483,
       0.28121826, 0.10903245, 0.01866273, 0.07199273, 0.023733  ,
       0.05276896, 0.01673512, 0.01160133],[0.        , 0.05135467, 0.00528496, 0.01272903, 0.02546261,
       0.04345459, 0.0347557 , 0.00971514, 0.02585146, 0.06509138,
       0.0441608 , 0.00832178, 0.01545118, 0.04687211, 0.00715973,
       0.01043576, 0.00982739, 0.00701059, 0.05499467, 0.18106187,
       0.06222544, 0.02942769, 0.02622915, 0.05589006, 0.0431471 ,
       0.00608879, 0.00850745, 0.19330054, 0.04096188, 0.03272733,
       0.18061858, 0.16615219, 0.04787209, 0.06655562, 0.04320239,
       0.16486924, 0.14454783, 0.03716569, 0.01821655, 0.12957457,
       0.11539138, 0.20297147, 0.02421359, 0.01726635, 0.01394322,
       0.01242524, 0.02364082, 0.02628365, 0.03956849, 0.30964328,
       0.07375389, 0.02306411, 0.01296202, 0.13639832, 0.02183657,
       0.03103152, 0.0300009 , 0.10419355, 0.00836036, 0.0262905 ,
       0.03371911, 0.02062664, 0.15325295, 0.04424893, 0.05206059,
       0.13655518, 0.00981178, 0.01785736, 0.02194467, 0.15665871,
       0.2078767 , 0.04476336, 0.0168834 , 0.29318046, 0.01189485,
       0.06498479, 0.02176048, 0.0832478 , 0.09358128, 0.16108716,
       0.08087183, 0.07085554, 0.03938825, 0.008119  , 0.01619514,
       0.13935892, 0.01451124, 0.00708504, 0.02405119, 0.01001359,
       0.02170419, 0.03736896, 0.02776131, 0.01121314, 0.01192278,
       0.18387386, 0.16815911, 0.00765232, 0.01481096, 0.03472409,
       0.14783063, 0.19258206, 0.13982224, 0.04756532, 0.03490312,
       0.03354586, 0.17709612, 0.07332745, 0.16466811, 0.05803312,
       0.33187753, 0.10788495, 0.01689225, 0.07428934, 0.02112621,
       0.05571363, 0.01642559, 0.0103161 ],[0.        , 0.04207348, 0.00598199, 0.01131055, 0.02479309,
       0.04576738, 0.03335735, 0.00986468, 0.02974655, 0.06752341,
       0.04744091, 0.00911032, 0.01610973, 0.04607232, 0.00709357,
       0.01257723, 0.01107643, 0.00683581, 0.0442395 , 0.18238915,
       0.06517081, 0.02750339, 0.03106134, 0.04760983, 0.04408481,
       0.00493888, 0.00805365, 0.20295983, 0.04247807, 0.03719486,
       0.16726239, 0.15133999, 0.05597651, 0.07047873, 0.04656507,
       0.13420869, 0.14484273, 0.03511817, 0.0182021 , 0.13792649,
       0.10462454, 0.17039435, 0.02774445, 0.01794827, 0.0148269 ,
       0.01228176, 0.02300479, 0.02390407, 0.03699185, 0.29695482,
       0.06824794, 0.0200815 , 0.01524278, 0.15433573, 0.01999294,
       0.03337434, 0.02772381, 0.1231638 , 0.00762374, 0.02939858,
       0.03076642, 0.01967779, 0.16451894, 0.04094902, 0.04458861,
       0.1142602 , 0.00979441, 0.01865571, 0.01934524, 0.14823384,
       0.20806609, 0.04605813, 0.01847254, 0.26221603, 0.00942301,
       0.06393752, 0.02198034, 0.08604396, 0.10531107, 0.15303805,
       0.08154856, 0.08371944, 0.04455428, 0.00686266, 0.01717885,
       0.13113921, 0.01833238, 0.00705735, 0.01954808, 0.01047624,
       0.02237341, 0.03930118, 0.02772548, 0.01142121, 0.01215004,
       0.1549874 , 0.15271553, 0.00874014, 0.01432626, 0.03316577,
       0.15936658, 0.16957497, 0.12695272, 0.04712573, 0.03494787,
       0.03429007, 0.15168708, 0.08436946, 0.1732087 , 0.06583534,
       0.341006  , 0.10693081, 0.0168116 , 0.07094021, 0.02065388,
       0.04828246, 0.01661885, 0.01216807],[0.        , 0.06073298, 0.00620588, 0.01467498, 0.02877084,
       0.05085484, 0.04249212, 0.01161718, 0.03832481, 0.08067425,
       0.05577084, 0.00962463, 0.02088579, 0.05683882, 0.00893897,
       0.01397063, 0.01290094, 0.00817136, 0.06550721, 0.21131449,
       0.07192043, 0.03740356, 0.03184486, 0.06416831, 0.04624447,
       0.00724016, 0.00968221, 0.23144396, 0.05024291, 0.04093314,
       0.19623624, 0.19820763, 0.05933238, 0.0798965 , 0.05178152,
       0.19470573, 0.18305884, 0.04867617, 0.02189597, 0.16719771,
       0.13600772, 0.22680037, 0.02989233, 0.02133563, 0.01657841,
       0.01676169, 0.02757359, 0.03006458, 0.04618793, 0.37745504,
       0.0825414 , 0.02511131, 0.01628181, 0.17452338, 0.02535774,
       0.03554162, 0.03621902, 0.1382767 , 0.00905872, 0.03265786,
       0.03456116, 0.02386126, 0.19442185, 0.05029133, 0.06119077,
       0.17048775, 0.01216077, 0.01869701, 0.028138  , 0.18245586,
       0.23985418, 0.05251043, 0.02175522, 0.35213924, 0.01288034,
       0.0770398 , 0.0285579 , 0.0959597 , 0.10626717, 0.2000104 ,
       0.10493138, 0.10116447, 0.05453734, 0.00821284, 0.01810625,
       0.15290807, 0.02066738, 0.00889519, 0.02413129, 0.01344151,
       0.02249903, 0.04299503, 0.03068245, 0.01166827, 0.01418815,
       0.18216024, 0.22611943, 0.01182437, 0.01886293, 0.04223184,
       0.18527468, 0.22474656, 0.17697191, 0.05352519, 0.04288705,
       0.04057765, 0.19739808, 0.09657112, 0.20891388, 0.06906633,
       0.35848048, 0.12266164, 0.02278657, 0.09134906, 0.02992454,
       0.06023449, 0.02034194, 0.012997  ],[0.        , 0.05502609, 0.00675773, 0.0129125 , 0.02873902,
       0.05740612, 0.04265681, 0.01078523, 0.03804935, 0.08076413,
       0.06070105, 0.01103215, 0.02179014, 0.053691  , 0.00875781,
       0.01333733, 0.01102858, 0.00781927, 0.06519445, 0.21719037,
       0.07054484, 0.03769106, 0.03471585, 0.06933527, 0.05148968,
       0.00584216, 0.01010815, 0.2558224 , 0.05036884, 0.0404577 ,
       0.22103504, 0.18848556, 0.06236313, 0.08113454, 0.05429302,
       0.17703888, 0.17248928, 0.04536263, 0.02019675, 0.15281205,
       0.1426758 , 0.20509351, 0.03184894, 0.02009767, 0.01593024,
       0.0147884 , 0.02500553, 0.02854681, 0.04406002, 0.36082845,
       0.08817347, 0.02688425, 0.01600122, 0.16907902, 0.02730324,
       0.0381578 , 0.03142142, 0.12844587, 0.00832316, 0.03151037,
       0.03988526, 0.02512094, 0.19545511, 0.04666404, 0.05928031,
       0.15138151, 0.01121066, 0.01900273, 0.02700415, 0.18893075,
       0.2382008 , 0.05095327, 0.02284579, 0.35591339, 0.01289424,
       0.07410218, 0.02445055, 0.09764179, 0.11755028, 0.18475598,
       0.10570306, 0.09675581, 0.04961734, 0.00877848, 0.02221941,
       0.15564961, 0.02056295, 0.0089927 , 0.02438865, 0.01340813,
       0.02420879, 0.04107408, 0.03605502, 0.0115555 , 0.01197677,
       0.21021087, 0.19413538, 0.00989717, 0.01761496, 0.04216805,
       0.17705336, 0.2100793 , 0.16891956, 0.05856394, 0.04148077,
       0.039098  , 0.19090619, 0.09541285, 0.20275781, 0.07662918,
       0.39342148, 0.11801152, 0.02128227, 0.0824959 , 0.0278954 ,
       0.05971122, 0.01979778, 0.014338  ],[0.        , 0.05554203, 0.00643128, 0.01337627, 0.02624249,
       0.05344913, 0.04495167, 0.01157159, 0.03647763, 0.0849774 ,
       0.05851252, 0.00943753, 0.02110378, 0.05995495, 0.00863896,
       0.01254371, 0.01150743, 0.00749213, 0.06433922, 0.25132006,
       0.07212109, 0.0358821 , 0.0303965 , 0.06760569, 0.0498087 ,
       0.00697376, 0.01058457, 0.24853686, 0.04474439, 0.04088453,
       0.19964941, 0.18518921, 0.06115982, 0.07883332, 0.05476461,
       0.19230421, 0.19293929, 0.04559947, 0.02179207, 0.15322918,
       0.14195576, 0.21097542, 0.03380795, 0.0211762 , 0.0153925 ,
       0.01455323, 0.02646924, 0.02971062, 0.04858781, 0.3759198 ,
       0.07998707, 0.02642005, 0.01505238, 0.16421675, 0.02320676,
       0.03958409, 0.03317064, 0.12998379, 0.00798252, 0.03416378,
       0.03834193, 0.02637478, 0.20122056, 0.05313115, 0.06180537,
       0.15817554, 0.01095455, 0.02014497, 0.02349209, 0.16940499,
       0.23056704, 0.05529796, 0.02030191, 0.35765466, 0.01173751,
       0.07990675, 0.02729778, 0.09324448, 0.11405083, 0.19849527,
       0.10002577, 0.10162034, 0.05814319, 0.00962118, 0.01937897,
       0.14734579, 0.01902467, 0.008946  , 0.02814526, 0.01327476,
       0.02680022, 0.04706612, 0.03201729, 0.01130258, 0.01222918,
       0.20757519, 0.18833526, 0.01034177, 0.01975852, 0.04242735,
       0.18153485, 0.21372235, 0.16572888, 0.05906082, 0.04414014,
       0.03828446, 0.19232383, 0.0884869 , 0.20624961, 0.06871223,
       0.35800291, 0.12091319, 0.02044913, 0.08614513, 0.02856419,
       0.06262437, 0.02046608, 0.01387762],[0.        , 0.05354009, 0.0064448 , 0.01446691, 0.03238882,
       0.0590363 , 0.04710566, 0.01198839, 0.03559422, 0.0885771 ,
       0.05707432, 0.01017915, 0.02161525, 0.05546017, 0.00779169,
       0.0141228 , 0.0124864 , 0.00774654, 0.06559761, 0.1973777 ,
       0.07119295, 0.0393998 , 0.03453328, 0.0654147 , 0.0487323 ,
       0.00643787, 0.01054502, 0.21462255, 0.04845113, 0.04228358,
       0.21261836, 0.18165461, 0.05512374, 0.08795673, 0.0493525 ,
       0.19587701, 0.16888069, 0.04363178, 0.02082074, 0.13437967,
       0.13636066, 0.2094351 , 0.02861953, 0.02227999, 0.01643482,
       0.01547174, 0.02309115, 0.03184498, 0.0476818 , 0.38088107,
       0.08898264, 0.02759606, 0.0178721 , 0.151637  , 0.02544827,
       0.03885926, 0.03494336, 0.1460755 , 0.00899042, 0.03082787,
       0.03566151, 0.02453426, 0.18312688, 0.04602609, 0.05362589,
       0.15237237, 0.01194747, 0.02009887, 0.02711854, 0.19259996,
       0.25568289, 0.05673154, 0.01997223, 0.32367359, 0.01327611,
       0.08125086, 0.02589108, 0.10292644, 0.11610952, 0.19380271,
       0.10758521, 0.09219343, 0.05510669, 0.00846634, 0.02082657,
       0.15205473, 0.0194685 , 0.00990686, 0.02384352, 0.01081665,
       0.0274283 , 0.04633295, 0.03360068, 0.01196494, 0.01303884,
       0.2221268 , 0.22031559, 0.00945646, 0.01921185, 0.03623705,
       0.17107931, 0.22519424, 0.1758065 , 0.05781092, 0.0434896 ,
       0.04160102, 0.18116999, 0.08314167, 0.19467316, 0.08264535,
       0.35213496, 0.11705647, 0.02063495, 0.08042358, 0.0235843 ,
       0.06159052, 0.01696723, 0.01428992],[0.        , 0.05454027, 0.00563251, 0.01324698, 0.02852894,
       0.06001884, 0.04306014, 0.01109723, 0.03654974, 0.07592312,
       0.06611171, 0.01100593, 0.02066336, 0.06027147, 0.0080378 ,
       0.01415652, 0.01345005, 0.00822829, 0.05875987, 0.21896825,
       0.07188172, 0.03656317, 0.03445698, 0.06702762, 0.04958416,
       0.00609869, 0.00987303, 0.21179669, 0.04789566, 0.04071222,
       0.1987371 , 0.19300786, 0.05890015, 0.08394788, 0.05225244,
       0.17401989, 0.18611585, 0.04222388, 0.02280961, 0.15506523,
       0.13874722, 0.21206258, 0.03296   , 0.02078386, 0.01818315,
       0.01723306, 0.02793104, 0.03228712, 0.04898491, 0.39692005,
       0.08064831, 0.02692926, 0.01724376, 0.17136073, 0.02475289,
       0.0408201 , 0.03238494, 0.14821277, 0.00881953, 0.03581873,
       0.04205294, 0.02413435, 0.17281043, 0.04844508, 0.05437592,
       0.1524038 , 0.01082302, 0.02076692, 0.02859207, 0.18278619,
       0.22515985, 0.04761664, 0.02115575, 0.37291403, 0.01306126,
       0.0698084 , 0.02489703, 0.09679492, 0.11896604, 0.18264057,
       0.09786052, 0.09986682, 0.05926401, 0.00914806, 0.01985873,
       0.14441287, 0.01993257, 0.0094523 , 0.02412692, 0.01254169,
       0.02569627, 0.0447044 , 0.03555526, 0.01153749, 0.01321182,
       0.20300748, 0.19882726, 0.01011746, 0.0158919 , 0.03963299,
       0.17043195, 0.21993   , 0.16013049, 0.05402463, 0.0411201 ,
       0.04330834, 0.20244947, 0.09489564, 0.21042161, 0.07943803,
       0.38149042, 0.12141461, 0.02221459, 0.08651521, 0.02730239,
       0.06473447, 0.02102344, 0.01246093],[0.        , 0.05515291, 0.00644297, 0.01455123, 0.03009845,
       0.05857929, 0.04117564, 0.01272378, 0.03521104, 0.08065433,
       0.05400263, 0.01065178, 0.0227123 , 0.05122059, 0.0082479 ,
       0.01268986, 0.01276023, 0.00795463, 0.05836006, 0.20458995,
       0.07463791, 0.03915171, 0.03492898, 0.06248964, 0.04521343,
       0.0061735 , 0.00978937, 0.22348044, 0.04534507, 0.0443019 ,
       0.19899905, 0.19808293, 0.0520088 , 0.0898946 , 0.0508108 ,
       0.18369694, 0.18002136, 0.04325366, 0.02231804, 0.14083976,
       0.13223064, 0.22750397, 0.03093238, 0.02072789, 0.01638982,
       0.01658856, 0.02777297, 0.02760122, 0.04550191, 0.37201231,
       0.08092152, 0.02584393, 0.01919704, 0.17561429, 0.02600391,
       0.03822478, 0.0339894 , 0.1406588 , 0.00882384, 0.0333913 ,
       0.04012829, 0.02470418, 0.20086012, 0.04356173, 0.05200884,
       0.13684715, 0.0108436 , 0.01909881, 0.02746175, 0.19646808,
       0.22151775, 0.05542783, 0.02223252, 0.38969365, 0.01196101,
       0.06922985, 0.02665105, 0.08897369, 0.12373086, 0.19824613,
       0.10515616, 0.09758358, 0.05690751, 0.00890228, 0.01972934,
       0.16329513, 0.02043277, 0.00863995, 0.02340543, 0.01256059,
       0.025075  , 0.04896711, 0.03251571, 0.01315529, 0.01290305,
       0.21972668, 0.21194846, 0.00998871, 0.01925533, 0.04126284,
       0.19664957, 0.20996059, 0.16695583, 0.05697527, 0.03994027,
       0.03957077, 0.19646685, 0.09308089, 0.19762387, 0.074531  ,
       0.38506377, 0.11505899, 0.01993571, 0.08829336, 0.02635554,
       0.06054441, 0.02054148, 0.01337886],[0.        , 0.04103865, 0.00543965, 0.01172538, 0.02474281,
       0.05415892, 0.03095633, 0.00855408, 0.02624067, 0.06430124,
       0.048301  , 0.00854233, 0.01855557, 0.0474498 , 0.0062757 ,
       0.01021864, 0.01116804, 0.00623666, 0.05149388, 0.17214417,
       0.05516428, 0.03419851, 0.02899548, 0.05021728, 0.04504854,
       0.00550059, 0.00887067, 0.19091529, 0.03732209, 0.03378459,
       0.17576991, 0.15455397, 0.04285791, 0.07319117, 0.04174725,
       0.15642916, 0.14281229, 0.0394633 , 0.01711361, 0.12507472,
       0.10551537, 0.17095444, 0.02509799, 0.01794138, 0.01331714,
       0.01278753, 0.02062022, 0.02360967, 0.04278603, 0.33543656,
       0.07398519, 0.02333791, 0.01537943, 0.12398745, 0.01870443,
       0.03030754, 0.0236472 , 0.11568884, 0.00732989, 0.02805853,
       0.03044468, 0.02009614, 0.1679407 , 0.03841324, 0.0486451 ,
       0.12750948, 0.00763411, 0.01794548, 0.02380592, 0.14909444,
       0.16900173, 0.03952751, 0.01577194, 0.30821234, 0.00912217,
       0.05878062, 0.02182505, 0.08291432, 0.09225994, 0.18037616,
       0.07631603, 0.09017358, 0.04525265, 0.00700398, 0.0154509 ,
       0.12355461, 0.01681052, 0.00806617, 0.02043729, 0.01045463,
       0.02010806, 0.03803202, 0.02826161, 0.01027163, 0.01195889,
       0.16877984, 0.16675564, 0.00826983, 0.01597573, 0.03210686,
       0.15343657, 0.16695273, 0.13683538, 0.04744183, 0.03869477,
       0.02988414, 0.16547356, 0.06648399, 0.1727244 , 0.06290981,
       0.32027571, 0.10255858, 0.02058247, 0.06518357, 0.02043925,
       0.0544496 , 0.01613041, 0.01136624],[0.        , 0.04649314, 0.00525075, 0.01121609, 0.02612485,
       0.05141176, 0.03108161, 0.0091436 , 0.02990863, 0.06481283,
       0.04978309, 0.00918767, 0.01664983, 0.05242159, 0.00744109,
       0.01120213, 0.01100552, 0.00743373, 0.04857746, 0.1672315 ,
       0.06804787, 0.03119921, 0.02596836, 0.05734458, 0.04361301,
       0.0047647 , 0.00825405, 0.19623929, 0.03637602, 0.03942451,
       0.16489723, 0.16403536, 0.05237116, 0.06759819, 0.04385813,
       0.15226359, 0.154178  , 0.03543707, 0.01658758, 0.13840457,
       0.11304562, 0.16203076, 0.02362462, 0.01700486, 0.01349625,
       0.01267486, 0.0235343 , 0.02510908, 0.04022065, 0.32756852,
       0.07157867, 0.02141239, 0.01482419, 0.14475555, 0.02095958,
       0.03103833, 0.02832536, 0.10543805, 0.00716664, 0.02709318,
       0.02837488, 0.02093821, 0.14942   , 0.03729979, 0.04856962,
       0.14152655, 0.00997605, 0.01818551, 0.02558147, 0.14989498,
       0.19951248, 0.04937885, 0.01853267, 0.27517628, 0.00990299,
       0.06370548, 0.02155016, 0.07460748, 0.102595  , 0.1432408 ,
       0.08426502, 0.07988206, 0.03799879, 0.00828695, 0.01644311,
       0.14737821, 0.01660019, 0.00734014, 0.01951751, 0.01032074,
       0.02088119, 0.03719762, 0.02930051, 0.01063849, 0.01191017,
       0.16728503, 0.18335451, 0.00977936, 0.01427099, 0.03464264,
       0.14814927, 0.16754994, 0.1344207 , 0.04392193, 0.03385057,
       0.03180322, 0.15991524, 0.07393349, 0.16440364, 0.06624668,
       0.31388578, 0.09612442, 0.0190518 , 0.06658379, 0.02040625,
       0.05339889, 0.01654155, 0.0112716 ],[0.        , 0.04584133, 0.00571369, 0.01197847, 0.02748294,
       0.0484638 , 0.03435091, 0.00942808, 0.0298754 , 0.06591196,
       0.0541782 , 0.00973526, 0.01543962, 0.04617301, 0.00830767,
       0.01060244, 0.0103198 , 0.00626358, 0.04630516, 0.15992368,
       0.06348354, 0.03216854, 0.02763177, 0.05457328, 0.04776268,
       0.0054946 , 0.00955319, 0.20720224, 0.0367942 , 0.0290397 ,
       0.16152799, 0.17071979, 0.05243023, 0.06860897, 0.04697907,
       0.15411544, 0.13041548, 0.03810842, 0.02010622, 0.11708261,
       0.1002073 , 0.18721639, 0.02661718, 0.01875444, 0.01380602,
       0.01294504, 0.02361819, 0.02578467, 0.03354175, 0.31569668,
       0.06813111, 0.02315122, 0.01332187, 0.13865407, 0.01993615,
       0.03563236, 0.02696953, 0.10813806, 0.00670099, 0.02799801,
       0.03094173, 0.02321081, 0.15017602, 0.03676433, 0.04518692,
       0.11465848, 0.00884561, 0.01779315, 0.02053502, 0.15683643,
       0.20370977, 0.04737453, 0.0183265 , 0.32942471, 0.01056063,
       0.0602472 , 0.02118757, 0.0761093 , 0.11172093, 0.14296824,
       0.0715956 , 0.08284753, 0.04708111, 0.00776215, 0.01529667,
       0.11318583, 0.01551843, 0.00767679, 0.02285044, 0.01030014,
       0.02060922, 0.03673763, 0.02578387, 0.01052262, 0.01007247,
       0.20776501, 0.16400799, 0.00844992, 0.01540274, 0.03402574,
       0.14723577, 0.17717314, 0.14228382, 0.0493134 , 0.03449425,
       0.0310566 , 0.15968291, 0.0798195 , 0.15523179, 0.06320551,
       0.303208  , 0.09526271, 0.01843174, 0.0709292 , 0.02071983,
       0.04935222, 0.01639961, 0.01200909],[0.        , 0.04449891, 0.00505033, 0.01124386, 0.02541699,
       0.0498548 , 0.03257655, 0.00807439, 0.0256689 , 0.06157176,
       0.04894104, 0.00931636, 0.01682461, 0.05396766, 0.00683655,
       0.01273498, 0.01127517, 0.00698174, 0.05024935, 0.17232284,
       0.06099121, 0.03235204, 0.03135199, 0.0561449 , 0.04336943,
       0.0058677 , 0.00810482, 0.21266944, 0.03876282, 0.03609619,
       0.15760457, 0.1490464 , 0.04701271, 0.07651245, 0.03947464,
       0.15176564, 0.15182647, 0.03240774, 0.01886047, 0.12335696,
       0.10026367, 0.18933632, 0.02968668, 0.01733581, 0.01367003,
       0.01454297, 0.02225348, 0.02568218, 0.03677808, 0.30008623,
       0.07861439, 0.02000681, 0.01431943, 0.13709915, 0.01928936,
       0.0336234 , 0.02875003, 0.10447184, 0.00762883, 0.02504045,
       0.03326994, 0.01947662, 0.14487962, 0.04487278, 0.05349879,
       0.14042909, 0.00877817, 0.01670465, 0.02344585, 0.17491171,
       0.20086674, 0.03909563, 0.01676108, 0.26125184, 0.0107789 ,
       0.05909075, 0.01895334, 0.08068763, 0.08628878, 0.16267296,
       0.07514868, 0.07986167, 0.03806201, 0.00730941, 0.01753978,
       0.12866615, 0.01538057, 0.00860782, 0.02152328, 0.01146227,
       0.021389  , 0.03932208, 0.02651124, 0.00943572, 0.01035707,
       0.1897383 , 0.16087154, 0.00788553, 0.01592895, 0.02990109,
       0.1566111 , 0.17575008, 0.12835479, 0.04447738, 0.03649891,
       0.03651106, 0.16985007, 0.07314334, 0.18352781, 0.05619571,
       0.31423909, 0.10627628, 0.01679217, 0.0654459 , 0.02127999,
       0.05388832, 0.01521377, 0.01112111],[0.        , 0.04398471, 0.00558908, 0.01222858, 0.02278763,
       0.05030331, 0.03564658, 0.0087665 , 0.02727751, 0.06413409,
       0.04937305, 0.00847217, 0.01969524, 0.04864537, 0.00668959,
       0.01091163, 0.0097521 , 0.00688476, 0.05051963, 0.16826088,
       0.06090357, 0.03284312, 0.02766314, 0.05744669, 0.04261838,
       0.00524901, 0.00832528, 0.18252673, 0.04359987, 0.03790743,
       0.1570245 , 0.17535745, 0.04914445, 0.07090535, 0.04521679,
       0.15235428, 0.14819057, 0.04157228, 0.01860693, 0.13555026,
       0.10561431, 0.19471422, 0.02445449, 0.02077618, 0.01310884,
       0.0119305 , 0.02064939, 0.02467736, 0.03844998, 0.30466668,
       0.06090842, 0.02139023, 0.0140396 , 0.14233539, 0.01906889,
       0.03176922, 0.0307387 , 0.11633987, 0.00796959, 0.02389492,
       0.03564953, 0.02106627, 0.17014683, 0.04462325, 0.04535523,
       0.14152304, 0.00942607, 0.0184965 , 0.02404563, 0.14054462,
       0.2174168 , 0.04792868, 0.01785643, 0.30996154, 0.01056237,
       0.07102519, 0.02041869, 0.08460604, 0.11104395, 0.16458758,
       0.09205633, 0.07408363, 0.04239526, 0.00668864, 0.01698978,
       0.122519  , 0.01745488, 0.00748324, 0.02028452, 0.01089157,
       0.02068153, 0.03981801, 0.02795837, 0.0106637 , 0.01042307,
       0.17551162, 0.18823271, 0.00858103, 0.01435529, 0.03887409,
       0.14380806, 0.17351857, 0.14885873, 0.04362991, 0.03684762,
       0.03225614, 0.149568  , 0.07171189, 0.1670567 , 0.05612274,
       0.29550605, 0.1078061 , 0.0194355 , 0.07126311, 0.02236217,
       0.04957843, 0.01631402, 0.01264304],[0.        , 0.04388944, 0.00556386, 0.01130952, 0.02416036,
       0.0557361 , 0.03339029, 0.00812293, 0.02756052, 0.0670139 ,
       0.04752797, 0.009045  , 0.01758533, 0.04434075, 0.00691713,
       0.01150361, 0.00969308, 0.00607491, 0.05140639, 0.17295634,
       0.05983025, 0.03113699, 0.02536039, 0.05169588, 0.04074811,
       0.00556304, 0.00834583, 0.1625228 , 0.03831031, 0.03167075,
       0.1800935 , 0.15019278, 0.04731679, 0.07565679, 0.04504847,
       0.14758202, 0.12676564, 0.03957803, 0.01808863, 0.1391671 ,
       0.11516724, 0.19284102, 0.02526704, 0.02015419, 0.01224747,
       0.0134759 , 0.02155989, 0.02763025, 0.0380108 , 0.33383907,
       0.07341631, 0.02124329, 0.01329248, 0.14086055, 0.02156742,
       0.02791515, 0.03144561, 0.09884593, 0.00767109, 0.02753026,
       0.03171631, 0.02182227, 0.14210879, 0.04507868, 0.04512058,
       0.1275035 , 0.01005717, 0.01749427, 0.02287689, 0.14812854,
       0.19491821, 0.04402922, 0.01745346, 0.31069846, 0.00973339,
       0.05829932, 0.02046891, 0.07603186, 0.09149504, 0.17366128,
       0.08632955, 0.07988169, 0.04962335, 0.00806063, 0.01696804,
       0.12381555, 0.01465449, 0.00791065, 0.01851237, 0.01042462,
       0.02175214, 0.0439079 , 0.03091115, 0.00936411, 0.01058232,
       0.18503259, 0.17266008, 0.00790612, 0.01571181, 0.02966273,
       0.13344741, 0.17224533, 0.13546613, 0.04148521, 0.03497317,
       0.03587061, 0.15254604, 0.06911669, 0.16953505, 0.05725428,
       0.30908519, 0.09193934, 0.01778748, 0.07703547, 0.02153957,
       0.05183763, 0.01793188, 0.01138994]])
    list_fu_he_Q = np.array([
        [0.        , 0.02722658, 0.00314177, 0.0059637 , 0.02078784,
       0.02082494, 0.01885091, 0.00383237, 0.01501693, 0.03389924,
       0.0238096 , 0.00605229, 0.00722447, 0.03413582, 0.0095354 ,
       0.0081278 , 0.00756144, 0.00359568, 0.02446013, 0.09666617,
       0.0439416 , 0.01475299, 0.01287643, 0.0294892 , 0.04654624,
       0.00659542, 0.00819694, 0.17246887, 0.01932187, 0.02689193,
       0.09718151, 0.13117277, 0.04270431, 0.02510055, 0.02785935,
       0.1146578 , 0.10025297, 0.01517104, 0.0126374 , 0.10926094,
       0.07579174, 0.07161834, 0.01849022, 0.01143608, 0.00956375,
       0.00598632, 0.01305251, 0.01426259, 0.01779229, 0.33764399,
       0.04693775, 0.01881842, 0.01236981, 0.09209398, 0.00827042,
       0.0291582 , 0.01744614, 0.10589097, 0.00485832, 0.01408618,
       0.03004469, 0.01545582, 0.14818066, 0.01442058, 0.03054372,
       0.06275398, 0.00788125, 0.00720161, 0.01130333, 0.11376985,
       0.07587652, 0.02783775, 0.00734028, 0.20309513, 0.00940528,
       0.03476151, 0.01386655, 0.07119237, 0.04892174, 0.12602621,
       0.05886666, 0.0597743 , 0.03403744, 0.00773961, 0.00871868,
       0.0685806 , 0.00609966, 0.00319227, 0.0125449 , 0.01031127,
       0.01970437, 0.02661032, 0.02120631, 0.00461148, 0.01788333,
       0.06953729, 0.10338229, 0.0035761 , 0.01007712, 0.01527752,
       0.0955179 , 0.13212049, 0.04686542, 0.03719189, 0.02167152,
       0.02381554, 0.12702378, 0.04141358, 0.1093818 , 0.0544191 ,
       0.25456205, 0.05920594, 0.01289506, 0.05163581, 0.01457049,
       0.02773815, 0.00783574, 0.00530358],[0.        , 0.0315336 , 0.0034176 , 0.00616829, 0.01886825,
       0.01994358, 0.0166237 , 0.00338834, 0.01527175, 0.03093883,
       0.02428864, 0.00594624, 0.0073305 , 0.03606127, 0.0087783 ,
       0.00743381, 0.00775099, 0.00351451, 0.02470578, 0.10200223,
       0.04810075, 0.01545009, 0.01186233, 0.02682948, 0.04643973,
       0.00776658, 0.00770603, 0.16360616, 0.0185691 , 0.02951119,
       0.10004334, 0.13796156, 0.04061818, 0.02294246, 0.02753702,
       0.11026667, 0.10223154, 0.01580322, 0.01194551, 0.09897205,
       0.08697752, 0.07479519, 0.02157407, 0.01146513, 0.00992409,
       0.00588081, 0.01144238, 0.01561395, 0.01623514, 0.39123572,
       0.04361172, 0.01689416, 0.01162764, 0.08682249, 0.007963  ,
       0.02687426, 0.01770879, 0.10763408, 0.00527469, 0.01344528,
       0.02512437, 0.01276657, 0.14625613, 0.01667597, 0.02870282,
       0.06034269, 0.00809597, 0.00827387, 0.01195903, 0.11639973,
       0.06993285, 0.02431941, 0.00723788, 0.19094844, 0.00836592,
       0.03507926, 0.0134431 , 0.06089037, 0.04835601, 0.12736643,
       0.05631816, 0.05167673, 0.03468955, 0.0079381 , 0.00803991,
       0.08155891, 0.00657295, 0.00378784, 0.01255516, 0.01039527,
       0.02055739, 0.02398445, 0.01876585, 0.00522687, 0.01738502,
       0.06899471, 0.10159427, 0.00331818, 0.00977537, 0.01335921,
       0.0994076 , 0.12317717, 0.05435983, 0.03998133, 0.02066079,
       0.02629101, 0.12891496, 0.03857512, 0.11639418, 0.05105932,
       0.27662513, 0.06512397, 0.01338333, 0.05843224, 0.01451712,
       0.02859327, 0.00889107, 0.00591207],[0.        , 0.03089769, 0.00365303, 0.00665502, 0.01996026,
       0.02016911, 0.01811601, 0.00347489, 0.013741  , 0.03003241,
       0.02354754, 0.00540401, 0.00670499, 0.03439497, 0.00877868,
       0.00772581, 0.00789465, 0.00344705, 0.02468457, 0.10753864,
       0.05395063, 0.01566558, 0.01192384, 0.02657994, 0.0454846 ,
       0.00754056, 0.00773028, 0.16046637, 0.01675339, 0.03174478,
       0.09425609, 0.143553  , 0.04070653, 0.02510902, 0.0278007 ,
       0.1161385 , 0.09805929, 0.01669032, 0.01242947, 0.09604349,
       0.08458342, 0.07221384, 0.02073415, 0.01195921, 0.00945382,
       0.00597813, 0.01228627, 0.01535231, 0.0170891 , 0.37660889,
       0.04011468, 0.01685981, 0.01180619, 0.08493869, 0.00823637,
       0.02899647, 0.0159206 , 0.10456893, 0.00508777, 0.01448881,
       0.02893562, 0.01497057, 0.13463336, 0.01472293, 0.02982804,
       0.05781231, 0.00786739, 0.00750979, 0.01214245, 0.10968545,
       0.07194931, 0.02449548, 0.00683615, 0.17855531, 0.00860558,
       0.03804401, 0.01450159, 0.06912365, 0.05055364, 0.12464511,
       0.05847809, 0.0535073 , 0.03502579, 0.0085709 , 0.00842558,
       0.07759472, 0.00611898, 0.00351225, 0.01321553, 0.01055731,
       0.01950325, 0.02601036, 0.01994322, 0.00450149, 0.01870878,
       0.06553705, 0.10913444, 0.00338608, 0.00873575, 0.01438187,
       0.10799854, 0.14239575, 0.05155268, 0.04142307, 0.01875577,
       0.0236609 , 0.13306521, 0.04031082, 0.11968081, 0.05133672,
       0.27951624, 0.06758011, 0.01241826, 0.05898298, 0.01530531,
       0.02842325, 0.0090123 , 0.00556328],[0.        , 0.03018238, 0.00345122, 0.00635629, 0.0186042 ,
       0.02066798, 0.01872172, 0.00360097, 0.01578263, 0.0305575 ,
       0.02251132, 0.00540659, 0.00682885, 0.03259186, 0.00799006,
       0.00795942, 0.00736401, 0.00362229, 0.02335495, 0.11219033,
       0.04926644, 0.01690585, 0.01164078, 0.02770285, 0.04487832,
       0.00743086, 0.00733186, 0.15182591, 0.01818074, 0.02925193,
       0.09837348, 0.13155851, 0.0385282 , 0.02578171, 0.0263114 ,
       0.11085354, 0.09175026, 0.01639215, 0.01163711, 0.10701659,
       0.08950768, 0.07760315, 0.02121519, 0.01245673, 0.00925013,
       0.0063459 , 0.01259108, 0.01460346, 0.01713121, 0.37349455,
       0.04483512, 0.01584417, 0.01249632, 0.08854801, 0.00827593,
       0.02708354, 0.01641499, 0.09462068, 0.00489967, 0.01461737,
       0.02851289, 0.01341057, 0.13146078, 0.01654414, 0.03142784,
       0.05311054, 0.0082333 , 0.00739232, 0.01199573, 0.11812933,
       0.07287885, 0.02427672, 0.00654889, 0.17827947, 0.00922311,
       0.03571937, 0.0133273 , 0.06770878, 0.04950931, 0.1280538 ,
       0.05597052, 0.05359976, 0.03575435, 0.00835515, 0.00792626,
       0.07882307, 0.00624079, 0.00360002, 0.01271448, 0.01036277,
       0.01852755, 0.02425518, 0.01979444, 0.00481321, 0.01959804,
       0.06494055, 0.11235803, 0.00337659, 0.00888705, 0.01414802,
       0.11007769, 0.13643501, 0.05011722, 0.04125221, 0.01987607,
       0.02634535, 0.12865616, 0.04142217, 0.1131752 , 0.05477504,
       0.26813894, 0.0622306 , 0.01192802, 0.05526996, 0.01580594,
       0.02841136, 0.00833942, 0.00581384],[0.        , 0.02955229, 0.0034442 , 0.00697374, 0.01976191,
       0.020716  , 0.01821771, 0.00343703, 0.01620926, 0.03270685,
       0.02237542, 0.00578677, 0.00687559, 0.03751494, 0.00876061,
       0.00799321, 0.00730561, 0.00347765, 0.02461928, 0.1112071 ,
       0.04551225, 0.01578999, 0.01166202, 0.02966984, 0.04607425,
       0.00766383, 0.00702107, 0.15990876, 0.01828338, 0.02944213,
       0.09323805, 0.13971057, 0.03750504, 0.0253567 , 0.02935782,
       0.10803943, 0.09373379, 0.0161565 , 0.01151474, 0.10217164,
       0.07969392, 0.07593463, 0.02042578, 0.01220605, 0.00949375,
       0.00583371, 0.01276379, 0.01529361, 0.01751266, 0.36921126,
       0.04018349, 0.01705731, 0.01202892, 0.08948226, 0.0077643 ,
       0.02575705, 0.01601104, 0.10079576, 0.00523203, 0.01518284,
       0.02820296, 0.01405808, 0.13450428, 0.01574778, 0.02922834,
       0.05714374, 0.00859327, 0.00733601, 0.01231876, 0.11741173,
       0.07003404, 0.02589276, 0.0064093 , 0.18594186, 0.00911218,
       0.03665833, 0.01341562, 0.0682978 , 0.05012294, 0.13277193,
       0.05607475, 0.05357637, 0.03631748, 0.00835757, 0.00765814,
       0.07416662, 0.00660705, 0.00335875, 0.01292151, 0.01080343,
       0.02081946, 0.02398508, 0.02068872, 0.00448194, 0.01936094,
       0.06694911, 0.11207487, 0.00350996, 0.00917136, 0.01401326,
       0.11108454, 0.12936674, 0.05017611, 0.03966576, 0.02035745,
       0.02514685, 0.12459312, 0.03959443, 0.1111502 , 0.05184139,
       0.26412482, 0.06438644, 0.01208322, 0.06096158, 0.01622629,
       0.02676707, 0.00935306, 0.00528759],[0.        , 0.02969202, 0.00318665, 0.00664744, 0.01908861,
       0.01975398, 0.0175488 , 0.00369701, 0.01576247, 0.03369082,
       0.02408283, 0.00562691, 0.00715095, 0.03399488, 0.00894304,
       0.00780766, 0.00770317, 0.00382011, 0.02293103, 0.11400879,
       0.04877979, 0.01542885, 0.01143054, 0.02746911, 0.04730004,
       0.0076681 , 0.0074716 , 0.15772865, 0.01731488, 0.03140522,
       0.0945733 , 0.13482215, 0.04361504, 0.02533804, 0.02948251,
       0.1047018 , 0.09585379, 0.01736748, 0.01169502, 0.09446791,
       0.08125468, 0.06911909, 0.01814936, 0.01214715, 0.00944972,
       0.00570951, 0.01267331, 0.01619889, 0.01684842, 0.34564879,
       0.0436276 , 0.01677702, 0.01206337, 0.08545594, 0.00821276,
       0.02650964, 0.01690719, 0.09955257, 0.00488994, 0.01471481,
       0.02658697, 0.01376987, 0.14327187, 0.01496999, 0.02993786,
       0.05343027, 0.00808262, 0.00745036, 0.01183272, 0.11458352,
       0.07267699, 0.02697509, 0.00660481, 0.1791853 , 0.00899026,
       0.03565873, 0.01303949, 0.06720997, 0.04950167, 0.12843413,
       0.05409196, 0.05520154, 0.03462531, 0.00849329, 0.00795976,
       0.07400047, 0.00609627, 0.00362994, 0.01307007, 0.01092102,
       0.01855053, 0.02470395, 0.01973444, 0.00469419, 0.01898338,
       0.06373019, 0.11127395, 0.0035237 , 0.00886754, 0.01384333,
       0.09578891, 0.14198429, 0.04893259, 0.04198   , 0.02018504,
       0.02596817, 0.12795101, 0.04262972, 0.11059166, 0.0545608 ,
       0.26506186, 0.06639115, 0.01256889, 0.05662071, 0.01490145,
       0.02637679, 0.00930858, 0.00581935],[0.        , 0.03458221, 0.00358855, 0.00766932, 0.02098476,
       0.02326767, 0.02066803, 0.00361558, 0.01626317, 0.03440661,
       0.02716411, 0.0067297 , 0.00778938, 0.03832488, 0.009984  ,
       0.00864175, 0.00803905, 0.00403066, 0.02763039, 0.11541463,
       0.05278492, 0.01783607, 0.01361431, 0.03296545, 0.05135247,
       0.00867106, 0.0085878 , 0.1758222 , 0.02042297, 0.03309368,
       0.10748544, 0.15331247, 0.04413421, 0.02883312, 0.0292787 ,
       0.13024753, 0.106104  , 0.01857515, 0.01267142, 0.10854222,
       0.09167912, 0.07339213, 0.02096205, 0.0139733 , 0.01094296,
       0.00678982, 0.01511727, 0.01825461, 0.01861986, 0.40195991,
       0.05006234, 0.01744319, 0.01354809, 0.09529864, 0.00879654,
       0.02940776, 0.0191343 , 0.11077201, 0.00565747, 0.01581646,
       0.02994567, 0.01520124, 0.15963678, 0.01754455, 0.03402697,
       0.06601844, 0.00914041, 0.00816584, 0.01221536, 0.14071946,
       0.08437111, 0.02828277, 0.00755491, 0.19668504, 0.010151  ,
       0.04195565, 0.0148874 , 0.07312856, 0.05374963, 0.14796808,
       0.05925705, 0.06298763, 0.0425272 , 0.00952949, 0.00843277,
       0.08700119, 0.00685988, 0.00385895, 0.0149316 , 0.01112548,
       0.02263443, 0.02643962, 0.02220046, 0.00552636, 0.0201787 ,
       0.07917343, 0.13020854, 0.00392821, 0.0102996 , 0.01599796,
       0.1212832 , 0.14790277, 0.05826257, 0.04392977, 0.02151297,
       0.02810332, 0.14505618, 0.04364995, 0.1279001 , 0.05690125,
       0.30121413, 0.07093379, 0.01418743, 0.06383499, 0.01724932,
       0.03005718, 0.00987019, 0.0063411 ],[0.        , 0.0344704 , 0.00356362, 0.00721337, 0.0212211 ,
       0.02294205, 0.02199564, 0.00376267, 0.01596135, 0.03643648,
       0.02402592, 0.00637386, 0.00735493, 0.03947996, 0.00989521,
       0.00878008, 0.00824468, 0.00365991, 0.02524311, 0.11836269,
       0.05001095, 0.01746073, 0.01353965, 0.03113992, 0.04894023,
       0.00802536, 0.00805212, 0.16766471, 0.01952515, 0.03480943,
       0.10509014, 0.14693722, 0.04945893, 0.02697287, 0.03080725,
       0.12538706, 0.103904  , 0.01865874, 0.01243921, 0.10962695,
       0.09034324, 0.07737012, 0.0217422 , 0.01347768, 0.01041144,
       0.006737  , 0.01318184, 0.01728813, 0.01902836, 0.38781387,
       0.04922497, 0.01872364, 0.01382204, 0.09673828, 0.00855264,
       0.02900525, 0.01870704, 0.11308658, 0.00570167, 0.01571078,
       0.03041219, 0.01586264, 0.1482165 , 0.01725407, 0.03210262,
       0.06449441, 0.00865328, 0.00794915, 0.01326331, 0.13797869,
       0.0768083 , 0.02745826, 0.00724151, 0.2032633 , 0.01019245,
       0.03871952, 0.01496695, 0.0713973 , 0.05509721, 0.13478163,
       0.06097567, 0.063117  , 0.03894841, 0.0096563 , 0.00908613,
       0.08477359, 0.00707608, 0.00378617, 0.01312604, 0.01140564,
       0.02318575, 0.02792861, 0.02225986, 0.00524772, 0.02033845,
       0.07365792, 0.11357946, 0.00395825, 0.0097932 , 0.01554947,
       0.11680443, 0.15400875, 0.05332293, 0.04390349, 0.02117782,
       0.02894843, 0.13242022, 0.04572914, 0.12593035, 0.0582969 ,
       0.29145857, 0.0711716 , 0.01409447, 0.06468768, 0.01765933,
       0.02898232, 0.00924625, 0.00621356],[0.        , 0.03197268, 0.00361985, 0.00765603, 0.02178559,
       0.023386  , 0.0204834 , 0.00373908, 0.01788681, 0.03537038,
       0.02571942, 0.00663891, 0.00797004, 0.03880259, 0.00976312,
       0.00847678, 0.00844499, 0.00404357, 0.0258603 , 0.11458607,
       0.05528289, 0.01723474, 0.01407584, 0.03000566, 0.05054306,
       0.00803957, 0.00781979, 0.17786286, 0.0202518 , 0.03269587,
       0.10702918, 0.15070103, 0.04420571, 0.02767548, 0.02982852,
       0.11892279, 0.10692491, 0.01839926, 0.01390266, 0.11405362,
       0.09733624, 0.07859267, 0.02158136, 0.01323457, 0.01090023,
       0.00713994, 0.0132128 , 0.01855223, 0.02023852, 0.39304074,
       0.04917971, 0.01924302, 0.01317103, 0.09358934, 0.00925177,
       0.02867161, 0.01859629, 0.1145665 , 0.00588633, 0.01737277,
       0.03080967, 0.0157954 , 0.15157681, 0.01696222, 0.03553224,
       0.0643856 , 0.00851638, 0.00850002, 0.01361311, 0.12114999,
       0.08561213, 0.02898469, 0.00730573, 0.20974551, 0.00963287,
       0.04024856, 0.01460854, 0.07425722, 0.05428966, 0.13527453,
       0.05905981, 0.0579635 , 0.0415683 , 0.00937748, 0.00886016,
       0.08781401, 0.00663721, 0.00389327, 0.01337295, 0.01163371,
       0.02165971, 0.02530828, 0.02283378, 0.00505996, 0.01995619,
       0.07054379, 0.12364437, 0.00372405, 0.00997803, 0.01468091,
       0.1205014 , 0.14264132, 0.05626798, 0.04676377, 0.02165887,
       0.02839219, 0.14460706, 0.04844065, 0.13587205, 0.05495394,
       0.30199501, 0.07306828, 0.01399065, 0.06793808, 0.01768797,
       0.03142032, 0.01032243, 0.00622609],[0.        , 0.03201543, 0.0039508 , 0.00742277, 0.02133834,
       0.02283559, 0.02047062, 0.00386053, 0.01679666, 0.03496983,
       0.02614021, 0.00603654, 0.00809728, 0.03893209, 0.0094119 ,
       0.0093954 , 0.00851349, 0.00393387, 0.02744862, 0.11779483,
       0.05704798, 0.01861469, 0.01326048, 0.03167457, 0.04974726,
       0.00773459, 0.00822116, 0.17257957, 0.0192004 , 0.03320347,
       0.11079216, 0.14865827, 0.04737123, 0.02721059, 0.03271076,
       0.12352568, 0.10805981, 0.01814321, 0.0131142 , 0.1110922 ,
       0.09096143, 0.08148316, 0.02159092, 0.01362834, 0.01079707,
       0.00700187, 0.01367297, 0.01743583, 0.01953324, 0.41662916,
       0.05116047, 0.01929197, 0.013836  , 0.09904707, 0.00906986,
       0.03020883, 0.01797236, 0.10697366, 0.00571901, 0.01597036,
       0.03129132, 0.01586204, 0.16204265, 0.01749551, 0.03350459,
       0.0621247 , 0.00894298, 0.00803726, 0.01312521, 0.13904437,
       0.08085098, 0.02668778, 0.00733531, 0.20411424, 0.00989986,
       0.0405472 , 0.01509132, 0.07491756, 0.05246117, 0.14194249,
       0.06253195, 0.06224785, 0.03835146, 0.00894214, 0.00914437,
       0.08262182, 0.00685521, 0.00381654, 0.01421576, 0.01177796,
       0.02192742, 0.02578953, 0.02352321, 0.00525846, 0.02058437,
       0.06804749, 0.12295559, 0.00367448, 0.01032457, 0.01710944,
       0.11779187, 0.14940435, 0.05668105, 0.04471894, 0.02299115,
       0.02647467, 0.14229076, 0.04449289, 0.1353529 , 0.05936329,
       0.30508196, 0.07704671, 0.01340504, 0.06198144, 0.01899861,
       0.02973309, 0.0100915 , 0.00658456],[0.        , 0.03404493, 0.00385445, 0.00710142, 0.02052202,
       0.02208776, 0.02077648, 0.00384942, 0.01631762, 0.03551256,
       0.02442698, 0.00623797, 0.00777092, 0.04020953, 0.009514  ,
       0.00892014, 0.00817336, 0.00395949, 0.02462998, 0.11533063,
       0.05490925, 0.01705309, 0.01400284, 0.03261465, 0.04884333,
       0.00850885, 0.00827531, 0.17387949, 0.01988864, 0.03411381,
       0.10089898, 0.14690188, 0.04566238, 0.02867552, 0.03019503,
       0.1201786 , 0.10455367, 0.0193444 , 0.01334556, 0.10918758,
       0.09410211, 0.08427903, 0.02193572, 0.01275659, 0.01089817,
       0.00675551, 0.01505668, 0.01792341, 0.01929886, 0.38606918,
       0.05132159, 0.01886872, 0.01329766, 0.09228197, 0.00896276,
       0.02912503, 0.01812041, 0.10434727, 0.00590888, 0.01656482,
       0.02895836, 0.01651401, 0.15938896, 0.01757234, 0.03425893,
       0.06744224, 0.00880541, 0.00852011, 0.01212908, 0.13846843,
       0.08341186, 0.02774816, 0.00767343, 0.21371558, 0.01009367,
       0.04101329, 0.01426752, 0.07597943, 0.05405387, 0.14850582,
       0.06024255, 0.05874004, 0.03947391, 0.00967375, 0.00813166,
       0.08521422, 0.00667815, 0.00382787, 0.01404193, 0.0122339 ,
       0.02161204, 0.0277951 , 0.02221884, 0.00504094, 0.02039468,
       0.07497413, 0.11755825, 0.00406286, 0.01042751, 0.01554158,
       0.11589851, 0.15295866, 0.05525411, 0.04406023, 0.02352758,
       0.02829487, 0.14847768, 0.04413838, 0.13728702, 0.05752223,
       0.29728335, 0.07332686, 0.01292605, 0.06438257, 0.01770544,
       0.03082964, 0.00972943, 0.00632946],[0.        , 0.03442897, 0.00348528, 0.0072168 , 0.02142738,
       0.02300824, 0.01999082, 0.00369655, 0.01665914, 0.03513044,
       0.024795  , 0.00668375, 0.00733737, 0.03724665, 0.00918693,
       0.00892798, 0.00882871, 0.00393628, 0.02488874, 0.11372198,
       0.05836011, 0.01860452, 0.01356646, 0.02996953, 0.05009866,
       0.00826431, 0.00834291, 0.17267613, 0.01907945, 0.03544292,
       0.10987508, 0.15278624, 0.04438773, 0.02876945, 0.03267318,
       0.11872907, 0.10318294, 0.01832777, 0.01280096, 0.1127382 ,
       0.08696504, 0.08085794, 0.02168563, 0.01285878, 0.01103487,
       0.00672736, 0.01449184, 0.01622128, 0.01908967, 0.43608048,
       0.04840562, 0.01915638, 0.01277513, 0.09202291, 0.00872431,
       0.02857179, 0.0185158 , 0.11580963, 0.0055779 , 0.01673001,
       0.02869222, 0.01674602, 0.16441189, 0.01775815, 0.03413676,
       0.0631082 , 0.00916917, 0.00879374, 0.01271162, 0.12734278,
       0.07955378, 0.02743945, 0.0074891 , 0.20193984, 0.00987508,
       0.04054089, 0.01514195, 0.07394217, 0.05504691, 0.14960899,
       0.06432564, 0.06160014, 0.04220644, 0.00942871, 0.00902924,
       0.08228997, 0.00665883, 0.00375391, 0.01412774, 0.01097662,
       0.02220559, 0.02699294, 0.02140463, 0.0056269 , 0.0199105 ,
       0.07750358, 0.11846359, 0.00386966, 0.01063551, 0.01658634,
       0.11535277, 0.14999172, 0.05448502, 0.04586518, 0.02188539,
       0.0263866 , 0.14100833, 0.04665117, 0.12931733, 0.05713331,
       0.29547189, 0.06800279, 0.01339114, 0.06137482, 0.0171775 ,
       0.02997971, 0.00923972, 0.00618995],[0.        , 0.03679088, 0.00423883, 0.00804878, 0.02395785,
       0.02529092, 0.02179554, 0.00419099, 0.01802128, 0.03802221,
       0.02772253, 0.00713581, 0.00830079, 0.04331391, 0.01007739,
       0.00990697, 0.00929802, 0.00410158, 0.02918018, 0.12707278,
       0.06065873, 0.01961895, 0.01451873, 0.03517735, 0.05430523,
       0.0090675 , 0.00957908, 0.19373316, 0.02205398, 0.03642063,
       0.11190675, 0.17906963, 0.04968885, 0.03295921, 0.0356643 ,
       0.13131921, 0.12004091, 0.01981   , 0.01434923, 0.12304092,
       0.10364319, 0.08617658, 0.02385225, 0.01483503, 0.0116492 ,
       0.00766166, 0.01605247, 0.01915471, 0.02080272, 0.44475141,
       0.05501501, 0.02074405, 0.0143416 , 0.09910705, 0.00995327,
       0.03142151, 0.02060126, 0.12391559, 0.00579168, 0.01730645,
       0.03284637, 0.01633099, 0.16473831, 0.01938603, 0.03531449,
       0.07128516, 0.01024526, 0.00969446, 0.01396557, 0.14149148,
       0.09148656, 0.02988059, 0.00804426, 0.22674611, 0.01081416,
       0.04359953, 0.01685171, 0.08005523, 0.05960045, 0.15372468,
       0.06973412, 0.06681613, 0.04293229, 0.01051634, 0.00955953,
       0.0986126 , 0.0077603 , 0.00426876, 0.01546915, 0.01318246,
       0.02406052, 0.02871728, 0.02400991, 0.00579714, 0.0216898 ,
       0.08039199, 0.13076036, 0.00425537, 0.0112763 , 0.01761494,
       0.12283231, 0.1764049 , 0.0630611 , 0.04771703, 0.02395096,
       0.03135823, 0.15351189, 0.04821454, 0.13795389, 0.06331291,
       0.34437157, 0.07778845, 0.01520368, 0.06712425, 0.01956221,
       0.0330284 , 0.01012958, 0.00661718],[0.        , 0.03582312, 0.00420324, 0.00749491, 0.02398885,
       0.0253768 , 0.0224539 , 0.00425938, 0.01877622, 0.03885849,
       0.02760619, 0.00671329, 0.0084079 , 0.04419251, 0.01013931,
       0.00982301, 0.0093316 , 0.00437241, 0.02863218, 0.12685432,
       0.05789059, 0.01963266, 0.01525923, 0.03605363, 0.05537215,
       0.00898568, 0.00872517, 0.19064257, 0.02243985, 0.03689626,
       0.11643317, 0.17351826, 0.04707756, 0.03061346, 0.03417909,
       0.13881874, 0.11315244, 0.02000109, 0.01438582, 0.12242357,
       0.1056074 , 0.08279685, 0.02424022, 0.0141551 , 0.01213005,
       0.00734835, 0.01489701, 0.01910759, 0.02281515, 0.44139452,
       0.05058716, 0.01975277, 0.0148146 , 0.09956677, 0.0096833 ,
       0.03250924, 0.02127376, 0.12076377, 0.00621277, 0.01749654,
       0.03362617, 0.0170765 , 0.16136961, 0.01956464, 0.03641835,
       0.07111056, 0.00945225, 0.00980907, 0.01378198, 0.14273836,
       0.08853185, 0.03192037, 0.00791583, 0.21796331, 0.01144332,
       0.04322235, 0.01606451, 0.08485718, 0.06043217, 0.15148136,
       0.06498102, 0.06998699, 0.04449551, 0.01027353, 0.00981352,
       0.09148528, 0.00777698, 0.00434925, 0.01520102, 0.01205781,
       0.02483594, 0.02982388, 0.02505304, 0.00573112, 0.02224948,
       0.07814233, 0.13880071, 0.00412547, 0.01123112, 0.0179471 ,
       0.13461229, 0.17155573, 0.05720637, 0.05051673, 0.02364677,
       0.02878566, 0.15524234, 0.05200793, 0.13391902, 0.06459941,
       0.33075065, 0.08036529, 0.01416326, 0.06961334, 0.01950784,
       0.03283237, 0.01073101, 0.0068481 ],[0.        , 0.03617085, 0.00425282, 0.00845943, 0.02403292,
       0.02586223, 0.02315751, 0.00424958, 0.01927379, 0.03899874,
       0.02873038, 0.00690486, 0.00838123, 0.04348121, 0.01070636,
       0.00998445, 0.00937202, 0.0044982 , 0.02862742, 0.13646508,
       0.05891687, 0.01883541, 0.01445052, 0.03547941, 0.05644485,
       0.00894742, 0.00877654, 0.18549818, 0.02149855, 0.03704579,
       0.12117004, 0.17430593, 0.05060567, 0.02952236, 0.03309862,
       0.14162401, 0.11780663, 0.02039963, 0.01425095, 0.12381072,
       0.10302869, 0.08894661, 0.02451549, 0.01493632, 0.01226455,
       0.00792395, 0.01538793, 0.01902988, 0.02059571, 0.43471798,
       0.05385762, 0.02055573, 0.01501207, 0.10219145, 0.01021525,
       0.03322923, 0.02032806, 0.12142089, 0.00599794, 0.01698716,
       0.03285853, 0.01779996, 0.17275488, 0.01914119, 0.03679686,
       0.07320177, 0.01037983, 0.00927335, 0.01389342, 0.14156514,
       0.08339243, 0.03022748, 0.00853485, 0.21308902, 0.01085066,
       0.04356133, 0.0166412 , 0.08373475, 0.05987393, 0.15005596,
       0.06860771, 0.06540315, 0.04529299, 0.01012616, 0.01017229,
       0.09309   , 0.0071498 , 0.00429524, 0.01565536, 0.0130246 ,
       0.02548636, 0.02949481, 0.02486163, 0.00607733, 0.02199795,
       0.08513337, 0.1275693 , 0.00412078, 0.01088581, 0.01756959,
       0.12773225, 0.15660476, 0.06208287, 0.05204695, 0.02427541,
       0.02909586, 0.1548722 , 0.049679  , 0.14012253, 0.06259514,
       0.3155409 , 0.08048319, 0.01555793, 0.06988844, 0.02020846,
       0.0311774 , 0.01084331, 0.00694632],[0.        , 0.03824032, 0.00424102, 0.00832511, 0.02300985,
       0.02433466, 0.02208654, 0.00400341, 0.01836187, 0.03860692,
       0.0279429 , 0.00697598, 0.00817978, 0.04462484, 0.01077298,
       0.00976473, 0.00935555, 0.00414314, 0.02801341, 0.13289421,
       0.05861248, 0.02018902, 0.01437165, 0.03375123, 0.05528122,
       0.00880887, 0.00905353, 0.20118844, 0.02260478, 0.03639512,
       0.11809909, 0.17037541, 0.04947255, 0.03017518, 0.03372549,
       0.13445878, 0.11762531, 0.02056618, 0.01445342, 0.12863314,
       0.10339076, 0.09156148, 0.02344932, 0.0144258 , 0.01130323,
       0.00774343, 0.01631607, 0.01946638, 0.02190724, 0.43362801,
       0.05448719, 0.02066761, 0.015557  , 0.1027029 , 0.00928509,
       0.03190927, 0.01956114, 0.12222426, 0.00630481, 0.01808752,
       0.03384356, 0.01618483, 0.17070751, 0.01887727, 0.03685331,
       0.0711029 , 0.00936953, 0.00961764, 0.014362  , 0.14160991,
       0.08863438, 0.03024576, 0.00854733, 0.23580513, 0.01094443,
       0.04423515, 0.01632255, 0.08237053, 0.06188093, 0.1559692 ,
       0.06367442, 0.07059176, 0.04226764, 0.00977303, 0.01022942,
       0.09106027, 0.00725282, 0.0043086 , 0.01621127, 0.01244801,
       0.02567769, 0.02942775, 0.02478785, 0.006053  , 0.02153012,
       0.08732206, 0.13514989, 0.00421303, 0.01137461, 0.01704613,
       0.12295846, 0.16416329, 0.06162418, 0.0519544 , 0.02456132,
       0.03037885, 0.15340167, 0.04639795, 0.14211537, 0.06179191,
       0.32665917, 0.08325016, 0.01559489, 0.07219761, 0.02039798,
       0.03465353, 0.01069418, 0.00683015],[0.        , 0.03723041, 0.00435779, 0.00799019, 0.02282418,
       0.02570245, 0.02160024, 0.00438148, 0.01897532, 0.0409191 ,
       0.02915257, 0.00658821, 0.00859168, 0.0439928 , 0.01072271,
       0.00938298, 0.00910551, 0.00438747, 0.02877987, 0.12976778,
       0.05759459, 0.02112099, 0.0147377 , 0.03391667, 0.05809796,
       0.00922265, 0.00874034, 0.19744026, 0.02241694, 0.03707208,
       0.11973769, 0.16412661, 0.05080394, 0.03046367, 0.03444621,
       0.13359648, 0.11651276, 0.02058825, 0.01422637, 0.12674587,
       0.10339958, 0.08561679, 0.02404496, 0.0149079 , 0.01099665,
       0.00771167, 0.01614776, 0.01855894, 0.02040743, 0.46566793,
       0.05615895, 0.02083795, 0.0152149 , 0.10715763, 0.00993291,
       0.03211273, 0.02007741, 0.11882355, 0.00596028, 0.01736318,
       0.03437666, 0.01832029, 0.16485958, 0.01878381, 0.03492001,
       0.0724622 , 0.00952356, 0.00910408, 0.01417778, 0.14060868,
       0.08738289, 0.03010756, 0.008147  , 0.23469721, 0.0109279 ,
       0.04538173, 0.01720348, 0.08230195, 0.06292605, 0.15254591,
       0.06484671, 0.06440387, 0.0456099 , 0.01029338, 0.0098786 ,
       0.09631033, 0.00794885, 0.00417653, 0.01581672, 0.01325447,
       0.02438145, 0.03116146, 0.02396383, 0.00591289, 0.02271242,
       0.08282605, 0.14056079, 0.00451944, 0.0107026 , 0.01703304,
       0.13014967, 0.16100368, 0.06194058, 0.05032468, 0.02469907,
       0.03022558, 0.14931877, 0.04780632, 0.14417303, 0.06224936,
       0.32617231, 0.07961612, 0.01505212, 0.06923596, 0.02012996,
       0.03356777, 0.01054646, 0.00689342],[0.        , 0.03710714, 0.0040412 , 0.00778066, 0.02343514,
       0.02471886, 0.02278459, 0.0041984 , 0.01823753, 0.04043875,
       0.02844521, 0.0067945 , 0.00857622, 0.04315669, 0.01041316,
       0.01027426, 0.0091662 , 0.00445365, 0.02826846, 0.12029157,
       0.06076219, 0.01955392, 0.01465245, 0.03562601, 0.05445216,
       0.00937645, 0.00897056, 0.19244831, 0.02161055, 0.03851697,
       0.1212277 , 0.15800563, 0.0467046 , 0.03001216, 0.03397693,
       0.13857012, 0.12384853, 0.02010099, 0.01470914, 0.12210522,
       0.1028588 , 0.08993912, 0.02356074, 0.01419669, 0.0117094 ,
       0.00751362, 0.01530263, 0.01970977, 0.02179282, 0.44629476,
       0.05329584, 0.02129214, 0.01426179, 0.10733999, 0.00991317,
       0.03047384, 0.02136014, 0.12099824, 0.00610475, 0.01797927,
       0.0323018 , 0.0178805 , 0.17653915, 0.0186993 , 0.03842767,
       0.07157156, 0.00975028, 0.00952759, 0.01437697, 0.14505821,
       0.08634326, 0.03222153, 0.00828526, 0.23552287, 0.01080333,
       0.04275129, 0.01641786, 0.08498325, 0.05992401, 0.16351873,
       0.06425328, 0.06780367, 0.04288577, 0.00986072, 0.00996631,
       0.09903387, 0.00763625, 0.00418849, 0.01577167, 0.01229544,
       0.02351561, 0.02808418, 0.02390839, 0.0058475 , 0.02331795,
       0.08528865, 0.1316581 , 0.00433701, 0.01103088, 0.01738781,
       0.12698651, 0.15893311, 0.06118405, 0.0488616 , 0.02388179,
       0.02876527, 0.15482685, 0.05134363, 0.13904421, 0.06356378,
       0.32559402, 0.0753364 , 0.01479816, 0.07520849, 0.0196723 ,
       0.03437031, 0.01130658, 0.00691151],[0.        , 0.03264348, 0.00384019, 0.00702928, 0.02084366,
       0.02245722, 0.01982464, 0.00385258, 0.01715608, 0.03679353,
       0.02532323, 0.00615172, 0.00768203, 0.03964516, 0.00906064,
       0.00858515, 0.00832962, 0.00388418, 0.02511513, 0.11781865,
       0.05673528, 0.01943467, 0.01312291, 0.03354759, 0.05035019,
       0.00790304, 0.00832213, 0.16164508, 0.02059555, 0.03439939,
       0.10374072, 0.15007929, 0.04701198, 0.02852211, 0.03197817,
       0.12526376, 0.10557571, 0.01915343, 0.01315125, 0.1189793 ,
       0.0874824 , 0.07622514, 0.02096664, 0.01246574, 0.01040829,
       0.0073871 , 0.01399928, 0.01686765, 0.01979189, 0.42035364,
       0.0499488 , 0.01897584, 0.01376462, 0.08935614, 0.00878141,
       0.02853868, 0.01826098, 0.10649179, 0.0059695 , 0.01612812,
       0.03046792, 0.01554109, 0.15535125, 0.01698227, 0.0325758 ,
       0.06858783, 0.00880327, 0.00884198, 0.01257623, 0.13031608,
       0.08013685, 0.02769645, 0.00715257, 0.21597674, 0.00970663,
       0.03945444, 0.0153289 , 0.07306064, 0.05326452, 0.14555352,
       0.06130762, 0.0641487 , 0.03955109, 0.00903308, 0.00855571,
       0.08827117, 0.00682351, 0.00411279, 0.01451028, 0.01238218,
       0.02209856, 0.02658962, 0.02361385, 0.00538541, 0.01903681,
       0.07777217, 0.11554441, 0.00395709, 0.01044696, 0.01643594,
       0.12318847, 0.1509559 , 0.05471512, 0.04398325, 0.02142887,
       0.02784079, 0.13172305, 0.04621563, 0.12378718, 0.05990468,
       0.30581512, 0.06953446, 0.01372817, 0.06257927, 0.01789878,
       0.02982728, 0.00922915, 0.00628704],[0.        , 0.03427011, 0.00376513, 0.00725004, 0.02087342,
       0.02386502, 0.02072183, 0.00400404, 0.01776563, 0.03561921,
       0.02376888, 0.00623064, 0.00752593, 0.04051957, 0.00949667,
       0.0086318 , 0.00816791, 0.00395663, 0.02458045, 0.11835552,
       0.05052405, 0.01737263, 0.0131786 , 0.03188562, 0.05003747,
       0.0080886 , 0.00807755, 0.17195965, 0.0200599 , 0.03299311,
       0.1020048 , 0.14864026, 0.04630834, 0.02812259, 0.0311273 ,
       0.115716  , 0.10222587, 0.01812345, 0.01343824, 0.11588273,
       0.09106161, 0.08381057, 0.02191289, 0.01265767, 0.01096434,
       0.00734236, 0.01308923, 0.01796073, 0.01979821, 0.38536037,
       0.04996921, 0.01848417, 0.01274439, 0.09743755, 0.00864005,
       0.02828325, 0.01817362, 0.1063576 , 0.00548053, 0.01696886,
       0.02934252, 0.01524312, 0.15872768, 0.01745409, 0.03180887,
       0.06322573, 0.00871618, 0.00855797, 0.01310628, 0.13126098,
       0.07874417, 0.02778883, 0.00723673, 0.21182718, 0.00997426,
       0.04028517, 0.0159246 , 0.0791181 , 0.05572464, 0.14534982,
       0.06216643, 0.05989807, 0.0420313 , 0.0089844 , 0.00898036,
       0.08738845, 0.00681895, 0.00386548, 0.01472776, 0.01253647,
       0.02221894, 0.02716648, 0.02281603, 0.00504644, 0.01961845,
       0.0724875 , 0.12459998, 0.00394812, 0.01035478, 0.01550551,
       0.11633433, 0.15142206, 0.05122886, 0.0445188 , 0.02268762,
       0.02781312, 0.13580215, 0.04295439, 0.13415723, 0.05822449,
       0.29267979, 0.06951433, 0.01333665, 0.06466259, 0.01819367,
       0.03065981, 0.01004592, 0.00633601],[0.        , 0.03475454, 0.003836  , 0.00751682, 0.02184647,
       0.02282863, 0.02151748, 0.0040488 , 0.01705254, 0.03696674,
       0.02484428, 0.00609093, 0.00777254, 0.03788627, 0.00917251,
       0.00884018, 0.00823487, 0.00403084, 0.02648055, 0.11046056,
       0.05283902, 0.01889565, 0.01338686, 0.03275897, 0.04843024,
       0.00805818, 0.00797019, 0.17228209, 0.02065786, 0.03232121,
       0.10164301, 0.15150155, 0.04433764, 0.0282285 , 0.03059159,
       0.12655329, 0.10683405, 0.01808447, 0.01286092, 0.11001237,
       0.0923061 , 0.07794849, 0.02235474, 0.01329306, 0.01114662,
       0.00665609, 0.01304662, 0.01820541, 0.0189099 , 0.42597322,
       0.04951329, 0.019289  , 0.01298103, 0.09187991, 0.00866568,
       0.02889524, 0.0188822 , 0.10662448, 0.00541543, 0.01588796,
       0.02902378, 0.01572387, 0.14920274, 0.01595051, 0.03432013,
       0.0661926 , 0.00885777, 0.00858574, 0.01343305, 0.13002476,
       0.08106195, 0.02612868, 0.00730076, 0.21056905, 0.01016244,
       0.03999105, 0.01508179, 0.07297738, 0.0552093 , 0.14682991,
       0.06437788, 0.0592014 , 0.04041795, 0.00993828, 0.00928025,
       0.08415807, 0.00663825, 0.00385172, 0.01321032, 0.01125388,
       0.02202174, 0.02604488, 0.02285283, 0.00532483, 0.02064379,
       0.07300235, 0.12386207, 0.00380015, 0.00989757, 0.01641988,
       0.11806046, 0.14345575, 0.05921008, 0.04534381, 0.02101802,
       0.02729964, 0.13344591, 0.04653198, 0.13075175, 0.05647109,
       0.30186018, 0.07478023, 0.01379031, 0.06161573, 0.01831198,
       0.03205031, 0.0095428 , 0.00624475],[0.        , 0.03373391, 0.0038003 , 0.00702626, 0.0221806 ,
       0.02289773, 0.01963256, 0.00385313, 0.01778965, 0.0347941 ,
       0.02630581, 0.00625431, 0.00772119, 0.03930121, 0.00982531,
       0.00848453, 0.00863158, 0.0038376 , 0.02460904, 0.11658191,
       0.05507701, 0.01772952, 0.01294779, 0.03139585, 0.04974934,
       0.00850234, 0.00844116, 0.18214098, 0.02052557, 0.03325671,
       0.10375653, 0.15258688, 0.04456321, 0.02785358, 0.03085652,
       0.12741137, 0.10468372, 0.0188497 , 0.01320942, 0.1124687 ,
       0.09186778, 0.07742487, 0.02184125, 0.01267938, 0.01030009,
       0.00675564, 0.01314052, 0.01715312, 0.02050912, 0.43347349,
       0.04940723, 0.01928208, 0.01388848, 0.0989527 , 0.00895638,
       0.02955827, 0.0187926 , 0.11591629, 0.00568636, 0.01744389,
       0.03031229, 0.01547256, 0.15175937, 0.0176674 , 0.03434663,
       0.06501882, 0.00862514, 0.00871221, 0.01291037, 0.13682706,
       0.07801439, 0.02849418, 0.0072742 , 0.20837105, 0.0096174 ,
       0.04297   , 0.01487633, 0.07551273, 0.05656119, 0.14192182,
       0.0634212 , 0.05830243, 0.03939759, 0.00915954, 0.0086535 ,
       0.08898466, 0.00685736, 0.00421039, 0.01448843, 0.01136279,
       0.02216227, 0.02863898, 0.02213741, 0.00532   , 0.02021476,
       0.07429452, 0.12523065, 0.00381617, 0.01010373, 0.01652549,
       0.11906709, 0.15683599, 0.0583625 , 0.041381  , 0.0226319 ,
       0.02887806, 0.13702036, 0.0426346 , 0.12964089, 0.05522866,
       0.28699868, 0.07357822, 0.01361641, 0.06484538, 0.01723184,
       0.02885086, 0.01042509, 0.00623813],[0.        , 0.03520306, 0.00360109, 0.00768859, 0.0201027 ,
       0.02248878, 0.02173439, 0.00399548, 0.01664115, 0.03469009,
       0.02689336, 0.00594799, 0.00796895, 0.03918583, 0.00954565,
       0.00887455, 0.00841114, 0.00416003, 0.02512075, 0.11950339,
       0.0562668 , 0.01803323, 0.01363725, 0.0337507 , 0.05375706,
       0.00788011, 0.00747747, 0.16963689, 0.02021715, 0.03439206,
       0.10442847, 0.15313371, 0.0460941 , 0.02915313, 0.03186627,
       0.12686845, 0.10623156, 0.01765916, 0.01311386, 0.12339569,
       0.0953212 , 0.07939666, 0.0221705 , 0.01367115, 0.01065579,
       0.00674313, 0.01402417, 0.01765053, 0.01830663, 0.37568507,
       0.04684606, 0.01868341, 0.01302917, 0.0960195 , 0.00891573,
       0.02850143, 0.01784994, 0.10943078, 0.00599877, 0.01673879,
       0.02836063, 0.01542272, 0.15741132, 0.01774092, 0.03375242,
       0.06309073, 0.00859761, 0.00803551, 0.0130615 , 0.13122219,
       0.08080622, 0.02651529, 0.00792242, 0.20153722, 0.00926503,
       0.04023016, 0.01521928, 0.07240607, 0.05519447, 0.14039037,
       0.05732937, 0.05924909, 0.03767056, 0.00913839, 0.00868017,
       0.08705923, 0.00670161, 0.00408104, 0.0133618 , 0.01156227,
       0.02195327, 0.0270712 , 0.02213594, 0.00536394, 0.02040052,
       0.0769063 , 0.11418897, 0.0038551 , 0.0102124 , 0.01696486,
       0.11817677, 0.15213189, 0.0540964 , 0.04803421, 0.02209619,
       0.02817167, 0.13840365, 0.04442009, 0.13199749, 0.06063757,
       0.31611968, 0.07311918, 0.01375054, 0.06447126, 0.01760659,
       0.02979885, 0.00926044, 0.00631245],[0.        , 0.03262149, 0.00370085, 0.00723635, 0.02186403,
       0.02280996, 0.0208691 , 0.00373427, 0.01786195, 0.03319911,
       0.02654975, 0.0059713 , 0.0076251 , 0.04093491, 0.00972674,
       0.00915172, 0.00838175, 0.00413062, 0.0255913 , 0.11912208,
       0.04976153, 0.01787121, 0.01247133, 0.03238293, 0.04704575,
       0.00837386, 0.00794571, 0.1654995 , 0.0192987 , 0.03292522,
       0.1020051 , 0.15521923, 0.04273698, 0.02755053, 0.03061507,
       0.12744536, 0.10651033, 0.01871579, 0.01305392, 0.11743968,
       0.09502424, 0.07297667, 0.02284649, 0.01288846, 0.01102578,
       0.00684981, 0.01392928, 0.01797621, 0.01983036, 0.39041959,
       0.04996721, 0.01967276, 0.01286855, 0.09301603, 0.00917561,
       0.02768483, 0.01837037, 0.11158648, 0.00566366, 0.0169632 ,
       0.02968195, 0.01614781, 0.1524397 , 0.01693272, 0.03076929,
       0.06941279, 0.00904938, 0.00829139, 0.01300356, 0.12889162,
       0.07661227, 0.02986298, 0.00756949, 0.20827705, 0.01023583,
       0.03914007, 0.01467842, 0.0727455 , 0.05172667, 0.13761316,
       0.05731343, 0.06246901, 0.04140105, 0.00943518, 0.00889079,
       0.08921694, 0.00670012, 0.00383012, 0.01467891, 0.01185381,
       0.02194115, 0.026878  , 0.0214101 , 0.00499517, 0.0203901 ,
       0.07146695, 0.12698517, 0.0039693 , 0.00988704, 0.01618004,
       0.12103497, 0.1491018 , 0.05883152, 0.04201671, 0.0212176 ,
       0.02940368, 0.13765524, 0.04529447, 0.13065794, 0.05841272,
       0.30772523, 0.07406342, 0.01370947, 0.068572  , 0.01836226,
       0.03077105, 0.00982399, 0.00652354]])

    _test_agent_1(env, agent, list_feng_11, list_feng_101, list_feng_32, list_feng_66,
                list_fu_he, list_fu_he_Q)

