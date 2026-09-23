from gurobipy import *
import gurobipy as grb
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from SAC_largesystem import PowerSystemEnv, SACContinuous
import rl_utils

import time


class Solve_MPC:
    def __init__(self, Q_1_mat, Q_2_mat, load_min, load_max, list_r_x_pu, list_p_q_pu, load_min_Q, load_max_Q):
        self.Q_1_mat = Q_1_mat
        self.Q_2_mat = Q_2_mat
        self.load_min = load_min
        self.load_max = load_max

        self.load_min_Q = load_min_Q
        self.load_max_Q = load_max_Q

        self.list_r_x_pu = list_r_x_pu
        self.list_p_q_pu = list_p_q_pu

        self.gen_min = 0
        self.gen_max = 1
        self.gen_ramp = 0.4

        self.SOC_min = 0.125
        self.SOC_max = 2.375
        self.SOC_scale = 0.5

        self.pv_min = 0
        self.pv_max = 3

        self.wind_min = 0
        self.wind_max = 2

        self.svc_min = -0.3
        self.svc_max = 0.3
        self.svc_scale = 0.3

        self.scale_Q = 0

        self.a_29_2 = 0.00240
        self.a_29_1 = 12.3299
        self.a_29_0 = 0
        self.a_64_2 = 0.00240
        self.a_64_1 = 12.3299
        self.a_64_0 = 0

        Price_high = 185
        Price_median = 123
        Price_low = 64
        self.Prices = np.array([Price_low, Price_low, Price_low, Price_low, Price_low, Price_low,
                                Price_low, Price_low, Price_low, Price_low, Price_low, Price_low,
                                Price_median, Price_median, Price_median, Price_median, Price_median, Price_median,
                                Price_median, Price_median, Price_median, Price_median, Price_median, Price_median,
                                Price_high, Price_high, Price_high, Price_high, Price_high, Price_high,
                                Price_high, Price_high, Price_high, Price_high, Price_high, Price_high,
                                Price_median, Price_median, Price_median, Price_median, Price_median, Price_median,
                                Price_median, Price_median, Price_median, Price_median, Price_median, Price_median])

    def _get_pq_t(self):
        list_p_q_pu_t = []
        for t in range(24 * 2):
            list_p_q_pu_t.append(np.array(self.list_p_q_pu).copy())
        list_p_q_pu_t = np.array(list_p_q_pu_t)
        for t in range(24 * 2):
            for i in range(118):
                list_p_q_pu_t[t][i][0] = np.array(self.load_list)[t][i].copy()
                list_p_q_pu_t[t][i][1] = np.array(self.load_list_Q)[t][i].copy()

        return list_p_q_pu_t

    def sol_pro(self, current_time, window_time,
                load_list, load_list_Q,
                P_DG_29_init, P_DG_64_init,
                SOC_BSS_12_init, SOC_BSS_34_init, SOC_BSS_68_init, SOC_BSS_103_init,
                wind_11_avail_list, wind_32_avail_list, wind_66_avail_list, wind_101_avail_list,
                pv_22_avail_list, pv_36_avail_list, pv_70_avail_list, pv_107_avail_list):
        self.load_list = load_list
        self.load_list_Q = load_list_Q
        self.wind_11_avail_list = wind_11_avail_list
        self.wind_32_avail_list = wind_32_avail_list
        self.wind_66_avail_list = wind_66_avail_list
        self.wind_101_avail_list = wind_101_avail_list
        self.pv_22_avail_list = pv_22_avail_list
        self.pv_36_avail_list = pv_36_avail_list
        self.pv_70_avail_list = pv_70_avail_list
        self.pv_107_avail_list = pv_107_avail_list
        self.list_p_q_pu_t = self._get_pq_t()

        obj = "error"
        try:
            model = Model('MPC_ICNN')

            model.setParam('MIPFocus',0)

            P_ij = model.addMVar(shape=(window_time - 1, 1, 118), lb=-100, ub=100, vtype=GRB.CONTINUOUS, name='P_ij')
            Q_ij = model.addMVar(shape=(window_time - 1, 1, 118), lb=-100, ub=100, vtype=GRB.CONTINUOUS, name='Q_ij')
            l_ij = model.addMVar(shape=(window_time - 1, 1, 118), lb=0, vtype=GRB.CONTINUOUS, name='l_ij')
            v_i = model.addMVar(shape=(window_time - 1, 1, 119), lb=0.9 * 0.9, ub=1.1 * 1.1, vtype=GRB.CONTINUOUS, name='v_i')
            lv_ij_i = model.addMVar(shape=(window_time - 1, 1, 118), vtype=GRB.CONTINUOUS, name='lv_ij_i')
            PP_ij = model.addMVar(shape=(window_time - 1, 1, 118), vtype=GRB.CONTINUOUS, name='PP_ij')
            QQ_ij = model.addMVar(shape=(window_time - 1, 1, 118), vtype=GRB.CONTINUOUS, name='QQ_ij')

            Loss = model.addMVar(shape=(window_time - 1,), lb=0, vtype=GRB.CONTINUOUS, name='Loss')
            Loss_i = model.addMVar(shape=(window_time - 1, 1, 118), lb=0, vtype=GRB.CONTINUOUS, name='Loss_i')

            Q_svc_20 = model.addMVar(shape=(window_time - 1,), lb=-0.3, ub=0.3, vtype=GRB.CONTINUOUS, name='Q_svc_20')
            Q_svc_56 = model.addMVar(shape=(window_time - 1,), lb=-0.3, ub=0.3, vtype=GRB.CONTINUOUS, name='Q_svc_56')
            Q_svc_79 = model.addMVar(shape=(window_time - 1,), lb=-0.3, ub=0.3, vtype=GRB.CONTINUOUS, name='Q_svc_79')
            Q_svc_105 = model.addMVar(shape=(window_time - 1,), lb=-0.3, ub=0.3, vtype=GRB.CONTINUOUS, name='Q_svc_105')

            Q_wt_11 = model.addMVar(shape=(window_time - 1,), vtype=GRB.CONTINUOUS, name='Q_wt_11')
            Q_wt_32 = model.addMVar(shape=(window_time - 1,), vtype=GRB.CONTINUOUS, name='Q_wt_32')
            Q_wt_66 = model.addMVar(shape=(window_time - 1,), vtype=GRB.CONTINUOUS, name='Q_wt_66')
            Q_wt_101 = model.addMVar(shape=(window_time - 1,), vtype=GRB.CONTINUOUS, name='Q_wt_101')

            Q_pv_22 = model.addMVar(shape=(window_time - 1,), vtype=GRB.CONTINUOUS, name='Q_pv_22')
            Q_pv_36 = model.addMVar(shape=(window_time - 1,), vtype=GRB.CONTINUOUS, name='Q_pv_36')
            Q_pv_70 = model.addMVar(shape=(window_time - 1,), vtype=GRB.CONTINUOUS, name='Q_pv_70')
            Q_pv_107 = model.addMVar(shape=(window_time - 1,), vtype=GRB.CONTINUOUS, name='Q_pv_107')

            P_DG_29 = model.addMVar(shape=(window_time - 1,), lb=self.gen_min, ub=self.gen_max, vtype=GRB.CONTINUOUS, name='P_DG_29')
            P_DG_64 = model.addMVar(shape=(window_time - 1,), lb=self.gen_min, ub=self.gen_max, vtype=GRB.CONTINUOUS, name='P_DG_64')

            SOC_BSS_12 = model.addMVar(shape=(window_time - 1,), lb=self.SOC_min, ub=self.SOC_max, vtype=GRB.CONTINUOUS, name='SOC_BSS_12')
            SOC_BSS_34 = model.addMVar(shape=(window_time - 1,), lb=self.SOC_min, ub=self.SOC_max, vtype=GRB.CONTINUOUS, name='SOC_BSS_34')
            SOC_BSS_68 = model.addMVar(shape=(window_time - 1,), lb=self.SOC_min, ub=self.SOC_max, vtype=GRB.CONTINUOUS, name='SOC_BSS_68')
            SOC_BSS_103 = model.addMVar(shape=(window_time - 1,), lb=self.SOC_min, ub=self.SOC_max, vtype=GRB.CONTINUOUS, name='SOC_BSS_103')

            P_BSS_ch_12 = model.addMVar(shape=(window_time - 1,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS, name='P_BSS_ch_12')
            P_BSS_ch_34 = model.addMVar(shape=(window_time - 1,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS, name='P_BSS_ch_34')
            P_BSS_ch_68 = model.addMVar(shape=(window_time - 1,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS, name='P_BSS_ch_68')
            P_BSS_ch_103 = model.addMVar(shape=(window_time - 1,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS, name='P_BSS_ch_103')
            P_BSS_dch_12 = model.addMVar(shape=(window_time - 1,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS, name='P_BSS_dch_12')
            P_BSS_dch_34 = model.addMVar(shape=(window_time - 1,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS, name='P_BSS_dch_34')
            P_BSS_dch_68 = model.addMVar(shape=(window_time - 1,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS, name='P_BSS_dch_68')
            P_BSS_dch_103 = model.addMVar(shape=(window_time - 1,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS, name='P_BSS_dch_103')

            cost_balance = model.addMVar(shape=(window_time - 1,), vtype=GRB.CONTINUOUS, name='cost_balance')
            P_balance = model.addMVar(shape=(window_time - 1,), vtype=GRB.CONTINUOUS, name='P_balance')
            cost_window = model.addVar(vtype=GRB.CONTINUOUS, name='cost_window')
            cost_day = model.addVar(vtype=GRB.CONTINUOUS, name='cost_day')
            cost_hour = model.addMVar(shape=(window_time - 1,), vtype=GRB.CONTINUOUS, name='cost_hour')
            cost_genera = model.addMVar(shape=(window_time - 1,), vtype=GRB.CONTINUOUS, name='cost_genera')

            model.setObjective(cost_day, GRB.MINIMIZE)

            model.addConstr(cost_day == cost_window)

            model.addConstr(cost_window == cost_hour.sum())
            for t in range(window_time - 1):
                model.addConstr(cost_hour[t] == cost_genera[t] + cost_balance[t])
                model.addConstr(cost_balance[t] == (P_balance[t] * self.Prices[t + current_time]) / 2)
                model.addConstr(P_balance[t] == grb.max_(P_ij[t, 0, 0], 0))
                model.addConstr(cost_genera[t] == (self.a_29_2 * P_DG_29[t] * P_DG_29[t] + self.a_29_1 * P_DG_29[t] + self.a_29_0 +
                                self.a_64_2 * P_DG_64[t] * P_DG_64[t] + self.a_64_1 * P_DG_64[t] + self.a_64_0) / 2)

            for t in range(window_time - 1):
                model.addConstr(self.wind_11_avail_list[t + current_time] * self.wind_11_avail_list[t + current_time]
                                + Q_wt_11[t] * Q_wt_11[t] <= self.wind_max * self.wind_max)
                model.addConstr(self.wind_32_avail_list[t + current_time] * self.wind_32_avail_list[t + current_time]
                                + Q_wt_32[t] * Q_wt_32[t] <= self.wind_max * self.wind_max)
                model.addConstr(self.wind_66_avail_list[t + current_time] * self.wind_66_avail_list[t + current_time]
                                + Q_wt_66[t] * Q_wt_66[t] <= self.wind_max * self.wind_max)
                model.addConstr(self.wind_101_avail_list[t + current_time] * self.wind_101_avail_list[t + current_time]
                                + Q_wt_101[t] * Q_wt_101[t] <= self.wind_max * self.wind_max)
                model.addConstr(self.pv_22_avail_list[t + current_time] * self.pv_22_avail_list[t + current_time]
                                + Q_pv_22[t] * Q_pv_22[t] <= self.pv_max * self.pv_max)
                model.addConstr(self.pv_36_avail_list[t + current_time] * self.pv_36_avail_list[t + current_time]
                                + Q_pv_36[t] * Q_pv_36[t] <= self.pv_max * self.pv_max)
                model.addConstr(self.pv_70_avail_list[t + current_time] * self.pv_70_avail_list[t + current_time]
                                + Q_pv_70[t] * Q_pv_70[t] <= self.pv_max * self.pv_max)
                model.addConstr(self.pv_107_avail_list[t + current_time] * self.pv_107_avail_list[t + current_time]
                                + Q_pv_107[t] * Q_pv_107[t] <= self.pv_max * self.pv_max)

            model.addConstr(P_DG_29[0] <= P_DG_29_init + self.gen_ramp / 2)
            model.addConstr(P_DG_64[0] <= P_DG_64_init + self.gen_ramp / 2)
            model.addConstr(P_DG_29[0] >= P_DG_29_init - self.gen_ramp / 2)
            model.addConstr(P_DG_64[0] >= P_DG_64_init - self.gen_ramp / 2)
            for t in range(window_time - 2):
                model.addConstr(P_DG_29[t + 1] <= P_DG_29[t] + self.gen_ramp / 2)
                model.addConstr(P_DG_64[t + 1] <= P_DG_64[t] + self.gen_ramp / 2)
                model.addConstr(P_DG_29[t + 1] >= P_DG_29[t] - self.gen_ramp / 2)
                model.addConstr(P_DG_64[t + 1] >= P_DG_64[t] - self.gen_ramp / 2)

            model.addConstr(SOC_BSS_12[0] == SOC_BSS_12_init + (P_BSS_ch_12[0] - P_BSS_dch_12[0]) / 2)
            for t in range(window_time - 2):
                model.addConstr(SOC_BSS_12[t + 1] == SOC_BSS_12[t] + (P_BSS_ch_12[t + 1] - P_BSS_dch_12[t + 1]) / 2)
            model.addConstr(SOC_BSS_34[0] == SOC_BSS_34_init + (P_BSS_ch_34[0] - P_BSS_dch_34[0]) / 2)
            for t in range(window_time - 2):
                model.addConstr(SOC_BSS_34[t + 1] == SOC_BSS_34[t] + (P_BSS_ch_34[t + 1] - P_BSS_dch_34[t + 1]) / 2)
            model.addConstr(SOC_BSS_68[0] == SOC_BSS_68_init + (P_BSS_ch_68[0] - P_BSS_dch_68[0]) / 2)
            for t in range(window_time - 2):
                model.addConstr(SOC_BSS_68[t + 1] == SOC_BSS_68[t] + (P_BSS_ch_68[t + 1] - P_BSS_dch_68[t + 1]) / 2)
            model.addConstr(SOC_BSS_103[0] == SOC_BSS_103_init + (P_BSS_ch_103[0] - P_BSS_dch_103[0]) / 2)
            for t in range(window_time - 2):
                model.addConstr(SOC_BSS_103[t + 1] == SOC_BSS_103[t] + (P_BSS_ch_103[t + 1] - P_BSS_dch_103[t + 1]) / 2)

            for t in range(window_time - 1):
                pass
                pass
            for t in range(window_time - 1):
                pass
                pass
            for t in range(window_time - 1):
                pass
                pass
            for t in range(window_time - 1):
                pass
                pass

            for t in range(window_time - 1):
                model.addConstr(v_i[t, 0, 0] == 1)

                for i in range(118):
                    model.addConstr(Loss_i[t, 0, i] == self.list_r_x_pu[i][0] * l_ij[t, 0, i])
                model.addConstr(Loss[t] == Loss_i[t].sum())

                for i in range(118):
                    model.addConstr(PP_ij[t, 0, i] == P_ij[t, 0, i] * P_ij[t, 0, i])
                    model.addConstr(QQ_ij[t, 0, i] == Q_ij[t, 0, i] * Q_ij[t, 0, i])

                model.addConstr(-self.list_p_q_pu_t[t + current_time][0][0]==
                                (P_ij[t,0,1]+P_ij[t,0,62]+P_ij[t,0,99])
                                -(P_ij[t,0,0]-self.list_r_x_pu[0][0]*l_ij[t,0,0]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][0][1]==
                                (Q_ij[t,0,1]+Q_ij[t,0,62]+Q_ij[t,0,99])
                                -(Q_ij[t,0,0]-self.list_r_x_pu[0][1]*l_ij[t,0,0]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][1][0]==
                                (P_ij[t,0,2]+P_ij[t,0,3]+P_ij[t,0,9])
                                -(P_ij[t,0,1]-self.list_r_x_pu[1][0]*l_ij[t,0,1]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][1][1]==
                                (Q_ij[t,0,2]+Q_ij[t,0,3]+Q_ij[t,0,9])
                                -(Q_ij[t,0,1]-self.list_r_x_pu[1][1]*l_ij[t,0,1]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][2][0]==
                                -(P_ij[t,0,2]-self.list_r_x_pu[2][0]*l_ij[t,0,2]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][2][1]==
                                -(Q_ij[t,0,2]-self.list_r_x_pu[2][1]*l_ij[t,0,2]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][3][0]==
                                (P_ij[t,0,4]+P_ij[t,0,27])
                                -(P_ij[t,0,3]-self.list_r_x_pu[3][0]*l_ij[t,0,3]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][3][1]==
                                (Q_ij[t,0,4]+Q_ij[t,0,27])
                                -(Q_ij[t,0,3]-self.list_r_x_pu[3][1]*l_ij[t,0,3]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][4][0] ==
                                (P_ij[t,0,5])
                                -(P_ij[t,0,4]-self.list_r_x_pu[4][0]*l_ij[t,0,4]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][4][1] ==
                                (Q_ij[t,0,5])
                                -(Q_ij[t,0,4]-self.list_r_x_pu[4][1]*l_ij[t,0,4]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][5][0]==
                                (P_ij[t,0,6])
                                -(P_ij[t,0,5]-self.list_r_x_pu[5][0]*l_ij[t,0,5]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][5][1]==
                                (Q_ij[t,0,6])
                                -(Q_ij[t,0,5]-self.list_r_x_pu[5][1]*l_ij[t,0,5]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][6][0]==
                                (P_ij[t,0,7])
                                -(P_ij[t,0,6]-self.list_r_x_pu[6][0]*l_ij[t,0,6]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][6][1]==
                                (Q_ij[t,0,7])
                                -(Q_ij[t,0,6]-self.list_r_x_pu[6][1]*l_ij[t,0,6]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][7][0]==
                                (P_ij[t,0,8])
                                -(P_ij[t,0,7]-self.list_r_x_pu[7][0]*l_ij[t,0,7]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][7][1]==
                                (Q_ij[t,0,8])
                                -(Q_ij[t,0,7]-self.list_r_x_pu[7][1]*l_ij[t,0,7]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][8][0]==
                                -(P_ij[t,0,8]-self.list_r_x_pu[8][0]*l_ij[t,0,8]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][8][1]==
                                -(Q_ij[t,0,8]-self.list_r_x_pu[8][1]*l_ij[t,0,8]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][9][0] + self.wind_11_avail_list[t + current_time] ==
                                (P_ij[t,0,10])
                                -(P_ij[t,0,9]-self.list_r_x_pu[9][0]*l_ij[t,0,9]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][9][1] + Q_wt_11[t] ==
                                (Q_ij[t,0,10])
                                -(Q_ij[t,0,9]-self.list_r_x_pu[9][1]*l_ij[t,0,9]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][10][0]- P_BSS_ch_12[t] * 1.02 + P_BSS_dch_12[t] * 0.98==
                                (P_ij[t,0,11]+P_ij[t,0,17])
                                -(P_ij[t,0,10]-self.list_r_x_pu[10][0]*l_ij[t,0,10]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][10][1]==
                                (Q_ij[t,0,11]+Q_ij[t,0,17])
                                -(Q_ij[t,0,10]-self.list_r_x_pu[10][1]*l_ij[t,0,10]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][11][0] ==
                                (P_ij[t,0,12])
                                -(P_ij[t,0,11]-self.list_r_x_pu[11][0]*l_ij[t,0,11]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][11][1] ==
                                (Q_ij[t,0,12])
                                -(Q_ij[t,0,11]-self.list_r_x_pu[11][1]*l_ij[t,0,11]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][12][0]==
                                (P_ij[t,0,13])
                                -(P_ij[t,0,12]-self.list_r_x_pu[12][0]*l_ij[t,0,12]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][12][1]==
                                (Q_ij[t,0,13])
                                -(Q_ij[t,0,12]-self.list_r_x_pu[12][1]*l_ij[t,0,12]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][13][0]==
                                (P_ij[t,0,14])
                                -(P_ij[t,0,13]-self.list_r_x_pu[13][0]*l_ij[t,0,13]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][13][1]==
                                (Q_ij[t,0,14])
                                -(Q_ij[t,0,13]-self.list_r_x_pu[13][1]*l_ij[t,0,13]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][14][0]==
                                (P_ij[t,0,15])
                                -(P_ij[t,0,14]-self.list_r_x_pu[14][0]*l_ij[t,0,14]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][14][1]==
                                (Q_ij[t,0,15])
                                -(Q_ij[t,0,14]-self.list_r_x_pu[14][1]*l_ij[t,0,14]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][15][0]==
                                (P_ij[t,0,16])
                                -(P_ij[t,0,15]-self.list_r_x_pu[15][0]*l_ij[t,0,15]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][15][1]==
                                (Q_ij[t,0,16])
                                -(Q_ij[t,0,15]-self.list_r_x_pu[15][1]*l_ij[t,0,15]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][16][0]==
                                -(P_ij[t,0,16]-self.list_r_x_pu[16][0]*l_ij[t,0,16]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][16][1]==
                                -(Q_ij[t,0,16]-self.list_r_x_pu[16][1]*l_ij[t,0,16]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][17][0]==
                                (P_ij[t,0,18])
                                -(P_ij[t,0,17]-self.list_r_x_pu[17][0]*l_ij[t,0,17]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][17][1]==
                                (Q_ij[t,0,18])
                                -(Q_ij[t,0,17]-self.list_r_x_pu[17][1]*l_ij[t,0,17]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][18][0]==
                                (P_ij[t,0,19])
                                -(P_ij[t,0,18]-self.list_r_x_pu[18][0]*l_ij[t,0,18]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][18][1] + Q_svc_20[t] ==
                                (Q_ij[t,0,19])
                                -(Q_ij[t,0,18]-self.list_r_x_pu[18][1]*l_ij[t,0,18]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][19][0]==
                                (P_ij[t,0,20])
                                -(P_ij[t,0,19]-self.list_r_x_pu[19][0]*l_ij[t,0,19]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][19][1]==
                                (Q_ij[t,0,20])
                                -(Q_ij[t,0,19]-self.list_r_x_pu[19][1]*l_ij[t,0,19]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][20][0] + self.pv_22_avail_list[t + current_time] ==
                                (P_ij[t,0,21])
                                -(P_ij[t,0,20]-self.list_r_x_pu[20][0]*l_ij[t,0,20]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][20][1] + Q_pv_22[t] ==
                                (Q_ij[t,0,21])
                                *(Q_ij[t,0,20]-self.list_r_x_pu[20][1]*l_ij[t,0,20]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][21][0]==
                                (P_ij[t,0,22])
                                -(P_ij[t,0,21]-self.list_r_x_pu[21][0]*l_ij[t,0,21]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][21][1]==
                                (Q_ij[t,0,22])
                                -(Q_ij[t,0,21]-self.list_r_x_pu[21][1]*l_ij[t,0,21]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][22][0]==
                                (P_ij[t,0,23])
                                -(P_ij[t,0,22]-self.list_r_x_pu[22][0]*l_ij[t,0,22]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][22][1]==
                                (Q_ij[t,0,23])
                                -(Q_ij[t,0,22]-self.list_r_x_pu[22][1]*l_ij[t,0,22]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][23][0]==
                                (P_ij[t,0,24])
                                -(P_ij[t,0,23]-self.list_r_x_pu[23][0]*l_ij[t,0,23]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][23][1]==
                                (Q_ij[t,0,24])
                                -(Q_ij[t,0,23]-self.list_r_x_pu[23][1]*l_ij[t,0,23]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][24][0]==
                                (P_ij[t,0,25])
                                -(P_ij[t,0,24]-self.list_r_x_pu[24][0]*l_ij[t,0,24]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][24][1]==
                                (Q_ij[t,0,25])
                                -(Q_ij[t,0,24]-self.list_r_x_pu[24][1]*l_ij[t,0,24]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][25][0]==
                                (P_ij[t,0,26])
                                -(P_ij[t,0,25]-self.list_r_x_pu[25][0]*l_ij[t,0,25]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][25][1]==
                                (Q_ij[t,0,26])
                                -(Q_ij[t,0,25]-self.list_r_x_pu[25][1]*l_ij[t,0,25]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][26][0]==
                                -(P_ij[t,0,26]-self.list_r_x_pu[26][0]*l_ij[t,0,26]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][26][1]==
                                -(Q_ij[t,0,26]-self.list_r_x_pu[26][1]*l_ij[t,0,26]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][27][0] + P_DG_29[t] ==
                                (P_ij[t,0,28])
                                -(P_ij[t,0,27]-self.list_r_x_pu[27][0]*l_ij[t,0,27]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][27][1]==
                                (Q_ij[t,0,28])
                                -(Q_ij[t,0,27]-self.list_r_x_pu[27][1]*l_ij[t,0,27]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][28][0]==
                                (P_ij[t,0,29]+P_ij[t,0,37]+P_ij[t,0,54])
                                -(P_ij[t,0,28]-self.list_r_x_pu[28][0]*l_ij[t,0,28]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][28][1]==
                                (Q_ij[t,0,29]+Q_ij[t,0,37]+Q_ij[t,0,54])
                                -(Q_ij[t,0,28]-self.list_r_x_pu[28][1]*l_ij[t,0,28]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][29][0]==
                                (P_ij[t,0,30]+P_ij[t,0,35])
                                -(P_ij[t,0,29]-self.list_r_x_pu[29][0]*l_ij[t,0,29]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][29][1]==
                                (Q_ij[t,0,30]+Q_ij[t,0,35])
                                -(Q_ij[t,0,29]-self.list_r_x_pu[29][1]*l_ij[t,0,29]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][30][0] + self.wind_32_avail_list[t + current_time] ==
                                (P_ij[t,0,31])
                                -(P_ij[t,0,30]-self.list_r_x_pu[30][0]*l_ij[t,0,30]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][30][1] + Q_wt_32[t] ==
                                (Q_ij[t,0,31])
                                -(Q_ij[t,0,30]-self.list_r_x_pu[30][1]*l_ij[t,0,30]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][31][0]==
                                (P_ij[t,0,32])
                                -(P_ij[t,0,31]-self.list_r_x_pu[31][0]*l_ij[t,0,31]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][31][1]==
                                (Q_ij[t,0,32])
                                -(Q_ij[t,0,31]-self.list_r_x_pu[31][1]*l_ij[t,0,31]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][32][0]- P_BSS_ch_34[t] * 1.02 + P_BSS_dch_34[t] * 0.98==
                                (P_ij[t,0,33])
                                -(P_ij[t,0,32]-self.list_r_x_pu[32][0]*l_ij[t,0,32]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][32][1]==
                                (Q_ij[t,0,33])
                                -(Q_ij[t,0,32]-self.list_r_x_pu[32][1]*l_ij[t,0,32]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][33][0] ==
                                (P_ij[t,0,34])
                                -(P_ij[t,0,33]-self.list_r_x_pu[33][0]*l_ij[t,0,33]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][33][1]==
                                (Q_ij[t,0,34])
                                -(Q_ij[t,0,33]-self.list_r_x_pu[33][1]*l_ij[t,0,33]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][34][0] + self.pv_36_avail_list[t + current_time]==
                                (P_ij[t,0,46])
                                -(P_ij[t,0,34]-self.list_r_x_pu[34][0]*l_ij[t,0,34]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][34][1] + Q_pv_36[t]==
                                (Q_ij[t,0,46])
                                -(Q_ij[t,0,34]-self.list_r_x_pu[34][1]*l_ij[t,0,34]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][35][0]==
                                (P_ij[t,0,36])
                                -(P_ij[t,0,35]-self.list_r_x_pu[35][0]*l_ij[t,0,35]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][35][1]==
                                (Q_ij[t,0,36])
                                -(Q_ij[t,0,35]-self.list_r_x_pu[35][1]*l_ij[t,0,35]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][36][0]==
                                -(P_ij[t,0,36]-self.list_r_x_pu[36][0]*l_ij[t,0,36]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][36][1]==
                                -(Q_ij[t,0,36]-self.list_r_x_pu[36][1]*l_ij[t,0,36]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][37][0]==
                                (P_ij[t,0,38])
                                -(P_ij[t,0,37]-self.list_r_x_pu[37][0]*l_ij[t,0,37]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][37][1]==
                                (Q_ij[t,0,38])
                                -(Q_ij[t,0,37]-self.list_r_x_pu[37][1]*l_ij[t,0,37]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][38][0]==
                                (P_ij[t,0,39])
                                -(P_ij[t,0,38]-self.list_r_x_pu[38][0]*l_ij[t,0,38]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][38][1]==
                                (Q_ij[t,0,39])
                                -(Q_ij[t,0,38]-self.list_r_x_pu[38][1]*l_ij[t,0,38]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][39][0] ==
                                (P_ij[t,0,40])
                                -(P_ij[t,0,39]-self.list_r_x_pu[39][0]*l_ij[t,0,39]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][39][1] ==
                                (Q_ij[t,0,40])
                                -(Q_ij[t,0,39]-self.list_r_x_pu[39][1]*l_ij[t,0,39]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][40][0]==
                                (P_ij[t,0,41])
                                -(P_ij[t,0,40]-self.list_r_x_pu[40][0]*l_ij[t,0,40]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][40][1]==
                                (Q_ij[t,0,41])
                                -(Q_ij[t,0,40]-self.list_r_x_pu[40][1]*l_ij[t,0,40]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][41][0]==
                                (P_ij[t,0,42])
                                -(P_ij[t,0,41]-self.list_r_x_pu[41][0]*l_ij[t,0,41]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][41][1]==
                                (Q_ij[t,0,42])
                                -(Q_ij[t,0,41]-self.list_r_x_pu[41][1]*l_ij[t,0,41]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][42][0]==
                                (P_ij[t,0,43])
                                -(P_ij[t,0,42]-self.list_r_x_pu[42][0]*l_ij[t,0,42]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][42][1]==
                                (Q_ij[t,0,43])
                                -(Q_ij[t,0,42]-self.list_r_x_pu[42][1]*l_ij[t,0,42]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][43][0]==
                                (P_ij[t,0,44])
                                -(P_ij[t,0,43]-self.list_r_x_pu[43][0]*l_ij[t,0,43]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][43][1]==
                                (Q_ij[t,0,44])
                                -(Q_ij[t,0,43]-self.list_r_x_pu[43][1]*l_ij[t,0,43]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][44][0]==
                                (P_ij[t,0,45])
                                -(P_ij[t,0,44]-self.list_r_x_pu[44][0]*l_ij[t,0,44]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][44][1]==
                                (Q_ij[t,0,45])
                                -(Q_ij[t,0,44]-self.list_r_x_pu[44][1]*l_ij[t,0,44]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][45][0]==
                                -(P_ij[t,0,45]-self.list_r_x_pu[45][0]*l_ij[t,0,45]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][45][1]==
                                -(Q_ij[t,0,45]-self.list_r_x_pu[45][1]*l_ij[t,0,45]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][46][0] ==
                                (P_ij[t,0,47])
                                -(P_ij[t,0,46]-self.list_r_x_pu[46][0]*l_ij[t,0,46]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][46][1] ==
                                (Q_ij[t,0,47])
                                -(Q_ij[t,0,46]-self.list_r_x_pu[46][1]*l_ij[t,0,46]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][47][0]==
                                (P_ij[t,0,48])
                                -(P_ij[t,0,47]-self.list_r_x_pu[47][0]*l_ij[t,0,47]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][47][1]==
                                (Q_ij[t,0,48])
                                -(Q_ij[t,0,47]-self.list_r_x_pu[47][1]*l_ij[t,0,47]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][48][0]==
                                (P_ij[t,0,49])
                                -(P_ij[t,0,48]-self.list_r_x_pu[48][0]*l_ij[t,0,48]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][48][1]==
                                (Q_ij[t,0,49])
                                -(Q_ij[t,0,48]-self.list_r_x_pu[48][1]*l_ij[t,0,48]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][49][0]==
                                (P_ij[t,0,50])
                                -(P_ij[t,0,49]-self.list_r_x_pu[49][0]*l_ij[t,0,49]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][49][1]==
                                (Q_ij[t,0,50])
                                -(Q_ij[t,0,49]-self.list_r_x_pu[49][1]*l_ij[t,0,49]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][50][0]==
                                (P_ij[t,0,51])
                                -(P_ij[t,0,50]-self.list_r_x_pu[50][0]*l_ij[t,0,50]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][50][1]==
                                (Q_ij[t,0,51])
                                -(Q_ij[t,0,50]-self.list_r_x_pu[50][1]*l_ij[t,0,50]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][51][0]==
                                (P_ij[t,0,52])
                                -(P_ij[t,0,51]-self.list_r_x_pu[51][0]*l_ij[t,0,51]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][51][1]==
                                (Q_ij[t,0,52])
                                -(Q_ij[t,0,51]-self.list_r_x_pu[51][1]*l_ij[t,0,51]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][52][0]==
                                (P_ij[t,0,53])
                                -(P_ij[t,0,52]-self.list_r_x_pu[52][0]*l_ij[t,0,52]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][52][1]==
                                (Q_ij[t,0,53])
                                -(Q_ij[t,0,52]-self.list_r_x_pu[52][1]*l_ij[t,0,52]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][53][0]==
                                -(P_ij[t,0,53]-self.list_r_x_pu[53][0]*l_ij[t,0,53]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][53][1]==
                                -(Q_ij[t,0,53]-self.list_r_x_pu[53][1]*l_ij[t,0,53]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][54][0] ==
                                (P_ij[t,0,55])
                                -(P_ij[t,0,54]-self.list_r_x_pu[54][0]*l_ij[t,0,54]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][54][1]+ Q_svc_56[t] ==
                                (Q_ij[t,0,55])
                                -(Q_ij[t,0,54]-self.list_r_x_pu[54][1]*l_ij[t,0,54]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][55][0]==
                                (P_ij[t,0,56])
                                -(P_ij[t,0,55]-self.list_r_x_pu[55][0]*l_ij[t,0,55]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][55][1]==
                                (Q_ij[t,0,56])
                                -(Q_ij[t,0,55]-self.list_r_x_pu[55][1]*l_ij[t,0,55]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][56][0]==
                                (P_ij[t,0,57])
                                -(P_ij[t,0,56]-self.list_r_x_pu[56][0]*l_ij[t,0,56]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][56][1]==
                                (Q_ij[t,0,57])
                                -(Q_ij[t,0,56]-self.list_r_x_pu[56][1]*l_ij[t,0,56]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][57][0]==
                                (P_ij[t,0,58])
                                -(P_ij[t,0,57]-self.list_r_x_pu[57][0]*l_ij[t,0,57]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][57][1]==
                                (Q_ij[t,0,58])
                                -(Q_ij[t,0,57]-self.list_r_x_pu[57][1]*l_ij[t,0,57]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][58][0]==
                                (P_ij[t,0,59])
                                -(P_ij[t,0,58]-self.list_r_x_pu[58][0]*l_ij[t,0,58]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][58][1]==
                                (Q_ij[t,0,59])
                                -(Q_ij[t,0,58]-self.list_r_x_pu[58][1]*l_ij[t,0,58]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][59][0]==
                                (P_ij[t,0,60])
                                -(P_ij[t,0,59]-self.list_r_x_pu[59][0]*l_ij[t,0,59]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][59][1]==
                                (Q_ij[t,0,60])
                                -(Q_ij[t,0,59]-self.list_r_x_pu[59][1]*l_ij[t,0,59]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][60][0]==
                                (P_ij[t,0,61])
                                -(P_ij[t,0,60]-self.list_r_x_pu[60][0]*l_ij[t,0,60]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][60][1]==
                                (Q_ij[t,0,61])
                                -(Q_ij[t,0,60]-self.list_r_x_pu[60][1]*l_ij[t,0,60]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][61][0]==
                                -(P_ij[t,0,61]-self.list_r_x_pu[61][0]*l_ij[t,0,61]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][61][1]==
                                -(Q_ij[t,0,61]-self.list_r_x_pu[61][1]*l_ij[t,0,61]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][62][0] + P_DG_64[t] ==
                                (P_ij[t,0,63])
                                -(P_ij[t,0,62]-self.list_r_x_pu[62][0]*l_ij[t,0,62]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][62][1]==
                                (Q_ij[t,0,63])
                                -(Q_ij[t,0,62]-self.list_r_x_pu[62][1]*l_ij[t,0,62]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][63][0]==
                                (P_ij[t,0,64]+P_ij[t,0,77])
                                -(P_ij[t,0,63]-self.list_r_x_pu[63][0]*l_ij[t,0,63]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][63][1]==
                                (Q_ij[t,0,64]+Q_ij[t,0,77])
                                -(Q_ij[t,0,63]-self.list_r_x_pu[63][1]*l_ij[t,0,63]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][64][0] + self.wind_66_avail_list[t + current_time] ==
                                (P_ij[t,0,65]+P_ij[t,0,88])
                                -(P_ij[t,0,64]-self.list_r_x_pu[64][0]*l_ij[t,0,64]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][64][1] + Q_wt_66[t] ==
                                (Q_ij[t,0,65]+Q_ij[t,0,88])
                                -(Q_ij[t,0,64]-self.list_r_x_pu[64][1]*l_ij[t,0,64]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][65][0] ==
                                (P_ij[t,0,66])
                                -(P_ij[t,0,65]-self.list_r_x_pu[65][0]*l_ij[t,0,65]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][65][1]==
                                (Q_ij[t,0,66])
                                -(Q_ij[t,0,65]-self.list_r_x_pu[65][1]*l_ij[t,0,65]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][66][0]- P_BSS_ch_68[t] * 1.02 + P_BSS_dch_68[t] * 0.98==
                                (P_ij[t,0,67])
                                -(P_ij[t,0,66]-self.list_r_x_pu[66][0]*l_ij[t,0,66]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][66][1]==
                                (Q_ij[t,0,67])
                                -(Q_ij[t,0,66]-self.list_r_x_pu[66][1]*l_ij[t,0,66]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][67][0] ==
                                (P_ij[t,0,68])
                                -(P_ij[t,0,67]-self.list_r_x_pu[67][0]*l_ij[t,0,67]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][67][1] ==
                                (Q_ij[t,0,68])
                                -(Q_ij[t,0,67]-self.list_r_x_pu[67][1]*l_ij[t,0,67]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][68][0]+ self.pv_70_avail_list[t + current_time]==
                                (P_ij[t,0,69])
                                -(P_ij[t,0,68]-self.list_r_x_pu[68][0]*l_ij[t,0,68]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][68][1] + Q_pv_70[t]==
                                (Q_ij[t,0,69])
                                -(Q_ij[t,0,68]-self.list_r_x_pu[68][1]*l_ij[t,0,68]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][69][0]==
                                (P_ij[t,0,70])
                                -(P_ij[t,0,69]-self.list_r_x_pu[69][0]*l_ij[t,0,69]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][69][1]==
                                (Q_ij[t,0,70])
                                -(Q_ij[t,0,69]-self.list_r_x_pu[69][1]*l_ij[t,0,69]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][70][0]==
                                (P_ij[t,0,71])
                                -(P_ij[t,0,70]-self.list_r_x_pu[70][0]*l_ij[t,0,70]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][70][1]==
                                (Q_ij[t,0,71])
                                -(Q_ij[t,0,70]-self.list_r_x_pu[70][1]*l_ij[t,0,70]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][71][0]==
                                (P_ij[t,0,72])
                                -(P_ij[t,0,71]-self.list_r_x_pu[71][0]*l_ij[t,0,71]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][71][1]==
                                (Q_ij[t,0,72])
                                -(Q_ij[t,0,71]-self.list_r_x_pu[71][1]*l_ij[t,0,71]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][72][0]==
                                (P_ij[t,0,73])
                                -(P_ij[t,0,72]-self.list_r_x_pu[72][0]*l_ij[t,0,72]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][72][1]==
                                (Q_ij[t,0,73])
                                -(Q_ij[t,0,72]-self.list_r_x_pu[72][1]*l_ij[t,0,72]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][73][0]==
                                (P_ij[t,0,74])
                                -(P_ij[t,0,73]-self.list_r_x_pu[73][0]*l_ij[t,0,73]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][73][1]==
                                (Q_ij[t,0,74])
                                -(Q_ij[t,0,73]-self.list_r_x_pu[73][1]*l_ij[t,0,73]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][74][0]==
                                (P_ij[t,0,75])
                                -(P_ij[t,0,74]-self.list_r_x_pu[74][0]*l_ij[t,0,74]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][74][1]==
                                (Q_ij[t,0,75])
                                -(Q_ij[t,0,74]-self.list_r_x_pu[74][1]*l_ij[t,0,74]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][75][0]==
                                (P_ij[t,0,76])
                                -(P_ij[t,0,75]-self.list_r_x_pu[75][0]*l_ij[t,0,75]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][75][1]==
                                (Q_ij[t,0,76])
                                -(Q_ij[t,0,75]-self.list_r_x_pu[75][1]*l_ij[t,0,75]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][76][0]==
                                -(P_ij[t,0,76]-self.list_r_x_pu[76][0]*l_ij[t,0,76]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][76][1]==
                                -(Q_ij[t,0,76]-self.list_r_x_pu[76][1]*l_ij[t,0,76]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][77][0]==
                                (P_ij[t,0,78])
                                -(P_ij[t,0,77]-self.list_r_x_pu[77][0]*l_ij[t,0,77]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][77][1] + Q_svc_79[t] ==
                                (Q_ij[t,0,78])
                                -(Q_ij[t,0,77]-self.list_r_x_pu[77][1]*l_ij[t,0,77]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][78][0] ==
                                (P_ij[t,0,79]+P_ij[t,0,85])
                                -(P_ij[t,0,78]-self.list_r_x_pu[78][0]*l_ij[t,0,78]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][78][1]==
                                (Q_ij[t,0,79]+Q_ij[t,0,85])
                                -(Q_ij[t,0,78]-self.list_r_x_pu[78][1]*l_ij[t,0,78]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][79][0]==
                                (P_ij[t,0,80])
                                -(P_ij[t,0,79]-self.list_r_x_pu[79][0]*l_ij[t,0,79]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][79][1]==
                                (Q_ij[t,0,80])
                                -(Q_ij[t,0,79]-self.list_r_x_pu[79][1]*l_ij[t,0,79]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][80][0]==
                                (P_ij[t,0,81])
                                -(P_ij[t,0,80]-self.list_r_x_pu[80][0]*l_ij[t,0,80]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][80][1]==
                                (Q_ij[t,0,81])
                                -(Q_ij[t,0,80]-self.list_r_x_pu[80][1]*l_ij[t,0,80]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][81][0]==
                                (P_ij[t,0,82])
                                -(P_ij[t,0,81]-self.list_r_x_pu[81][0]*l_ij[t,0,81]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][81][1]==
                                (Q_ij[t,0,82])
                                -(Q_ij[t,0,81]-self.list_r_x_pu[81][1]*l_ij[t,0,81]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][82][0]==
                                (P_ij[t,0,83])
                                -(P_ij[t,0,82]-self.list_r_x_pu[82][0]*l_ij[t,0,82]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][82][1]==
                                (Q_ij[t,0,83])
                                -(Q_ij[t,0,82]-self.list_r_x_pu[82][1]*l_ij[t,0,82]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][83][0]==
                                (P_ij[t,0,84])
                                -(P_ij[t,0,83]-self.list_r_x_pu[83][0]*l_ij[t,0,83]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][83][1]==
                                (Q_ij[t,0,84])
                                -(Q_ij[t,0,83]-self.list_r_x_pu[83][1]*l_ij[t,0,83]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][84][0]==
                                -(P_ij[t,0,84]-self.list_r_x_pu[84][0]*l_ij[t,0,84]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][84][1]==
                                -(Q_ij[t,0,84]-self.list_r_x_pu[84][1]*l_ij[t,0,84]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][85][0]==
                                (P_ij[t,0,86])
                                -(P_ij[t,0,85]-self.list_r_x_pu[85][0]*l_ij[t,0,85]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][85][1]==
                                (Q_ij[t,0,86])
                                -(Q_ij[t,0,85]-self.list_r_x_pu[85][1]*l_ij[t,0,85]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][86][0]==
                                (P_ij[t,0,87])
                                -(P_ij[t,0,86]-self.list_r_x_pu[86][0]*l_ij[t,0,86]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][86][1]==
                                (Q_ij[t,0,87])
                                -(Q_ij[t,0,86]-self.list_r_x_pu[86][1]*l_ij[t,0,86]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][87][0]==
                                -(P_ij[t,0,87]-self.list_r_x_pu[87][0]*l_ij[t,0,87]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][87][1]==
                                -(Q_ij[t,0,87]-self.list_r_x_pu[87][1]*l_ij[t,0,87]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][88][0] ==
                                (P_ij[t,0,89])
                                -(P_ij[t,0,88]-self.list_r_x_pu[88][0]*l_ij[t,0,88]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][88][1] ==
                                (Q_ij[t,0,89])
                                -(Q_ij[t,0,88]-self.list_r_x_pu[88][1]*l_ij[t,0,88]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][89][0] ==
                                (P_ij[t,0,90])
                                -(P_ij[t,0,89]-self.list_r_x_pu[89][0]*l_ij[t,0,89]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][89][1]==
                                (Q_ij[t,0,90])
                                -(Q_ij[t,0,89]-self.list_r_x_pu[89][1]*l_ij[t,0,89]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][90][0]==
                                (P_ij[t,0,91]+P_ij[t,0,95])
                                -(P_ij[t,0,90]-self.list_r_x_pu[90][0]*l_ij[t,0,90]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][90][1]==
                                (Q_ij[t,0,91]+Q_ij[t,0,95])
                                -(Q_ij[t,0,90]-self.list_r_x_pu[90][1]*l_ij[t,0,90]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][91][0]==
                                (P_ij[t,0,92])
                                -(P_ij[t,0,91]-self.list_r_x_pu[91][0]*l_ij[t,0,91]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][91][1]==
                                (Q_ij[t,0,92])
                                -(Q_ij[t,0,91]-self.list_r_x_pu[91][1]*l_ij[t,0,91]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][92][0]==
                                (P_ij[t,0,93])
                                -(P_ij[t,0,92]-self.list_r_x_pu[92][0]*l_ij[t,0,92]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][92][1]==
                                (Q_ij[t,0,93])
                                -(Q_ij[t,0,92]-self.list_r_x_pu[92][1]*l_ij[t,0,92]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][93][0]==(1*P_ij[t,0,94])
                                -(1*(P_ij[t,0,93]-self.list_r_x_pu[93][0]*l_ij[t,0,93])))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][93][1]==(1*Q_ij[t,0,94])
                                -(1*(Q_ij[t,0,93]-self.list_r_x_pu[93][1]*l_ij[t,0,93])))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][94][0]==
                                -(1*(P_ij[t,0,94]-self.list_r_x_pu[94][0]*l_ij[t,0,94])))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][94][1]==
                                -(1*(Q_ij[t,0,94]-self.list_r_x_pu[94][1]*l_ij[t,0,94])))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][95][0]==
                                (P_ij[t,0,96])
                                -(P_ij[t,0,95]-self.list_r_x_pu[95][0]*l_ij[t,0,95]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][95][1]==
                                (Q_ij[t,0,96])
                                -(Q_ij[t,0,95]-self.list_r_x_pu[95][1]*l_ij[t,0,95]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][96][0]==
                                (P_ij[t,0,97])
                                -(P_ij[t,0,96]-self.list_r_x_pu[96][0]*l_ij[t,0,96]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][96][1]==
                                (Q_ij[t,0,97])
                                -(Q_ij[t,0,96]-self.list_r_x_pu[96][1]*l_ij[t,0,96]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][97][0]==
                                (P_ij[t,0,98])
                                -(P_ij[t,0,97]-self.list_r_x_pu[97][0]*l_ij[t,0,97]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][97][1]==
                                (Q_ij[t,0,98])
                                -(Q_ij[t,0,97]-self.list_r_x_pu[97][1]*l_ij[t,0,97]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][98][0]==
                                -(P_ij[t,0,98]-self.list_r_x_pu[98][0]*l_ij[t,0,98]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][98][1]==
                                -(Q_ij[t,0,98]-self.list_r_x_pu[98][1]*l_ij[t,0,98]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][99][0] + self.wind_101_avail_list[t + current_time] ==
                                (P_ij[t,0,100]+P_ij[t,0,113])
                                -(P_ij[t,0,99]-self.list_r_x_pu[99][0]*l_ij[t,0,99]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][99][1] + Q_wt_101[t] ==
                                (Q_ij[t,0,100]+Q_ij[t,0,113])
                                -(Q_ij[t,0,99]-self.list_r_x_pu[99][1]*l_ij[t,0,99]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][100][0] ==
                                (P_ij[t,0,101])
                                -(P_ij[t,0,100]-self.list_r_x_pu[100][0]*l_ij[t,0,100]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][100][1]==
                                (Q_ij[t,0,101])
                                -(Q_ij[t,0,100]-self.list_r_x_pu[100][1]*l_ij[t,0,100]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][101][0]- P_BSS_ch_103[t] * 1.02 + P_BSS_dch_103[t] * 0.98==
                                (P_ij[t,0,102])
                                -(P_ij[t,0,101]-self.list_r_x_pu[101][0]*l_ij[t,0,101]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][101][1]==
                                (Q_ij[t,0,102])
                                -(Q_ij[t,0,101]-self.list_r_x_pu[101][1]*l_ij[t,0,101]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][102][0]==
                                (P_ij[t,0,103])
                                -(P_ij[t,0,102]-self.list_r_x_pu[102][0]*l_ij[t,0,102]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][102][1]==
                                (Q_ij[t,0,103])
                                -(Q_ij[t,0,102]-self.list_r_x_pu[102][1]*l_ij[t,0,102]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][103][0]==
                                (P_ij[t,0,104])
                                -(P_ij[t,0,103]-self.list_r_x_pu[103][0]*l_ij[t,0,103]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][103][1]+ Q_svc_105[t]==
                                (Q_ij[t,0,104])
                                -(Q_ij[t,0,103]-self.list_r_x_pu[103][1]*l_ij[t,0,103]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][104][0]==
                                (P_ij[t,0,105])
                                -(P_ij[t,0,104]-self.list_r_x_pu[104][0]*l_ij[t,0,104]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][104][1]==
                                (Q_ij[t,0,105])
                                -(Q_ij[t,0,104]-self.list_r_x_pu[104][1]*l_ij[t,0,104]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][105][0]+ self.pv_107_avail_list[t + current_time] ==
                                (P_ij[t,0,106])
                                -(P_ij[t,0,105]-self.list_r_x_pu[105][0]*l_ij[t,0,105]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][105][1]+Q_pv_107[t] ==
                                (Q_ij[t,0,106])
                                -(Q_ij[t,0,105]-self.list_r_x_pu[105][1]*l_ij[t,0,105]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][106][0]==
                                (P_ij[t,0,107])
                                -(P_ij[t,0,106]-self.list_r_x_pu[106][0]*l_ij[t,0,106]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][106][1]==
                                (Q_ij[t,0,107])
                                -(Q_ij[t,0,106]-self.list_r_x_pu[106][1]*l_ij[t,0,106]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][107][0]==
                                (P_ij[t,0,108])
                                -(P_ij[t,0,107]-self.list_r_x_pu[107][0]*l_ij[t,0,107]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][107][1]==
                                (Q_ij[t,0,108])
                                -(Q_ij[t,0,107]-self.list_r_x_pu[107][1]*l_ij[t,0,107]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][108][0]==
                                (P_ij[t,0,109])
                                -(P_ij[t,0,108]-self.list_r_x_pu[108][0]*l_ij[t,0,108]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][108][1]==
                                (Q_ij[t,0,109])
                                -(Q_ij[t,0,108]-self.list_r_x_pu[108][1]*l_ij[t,0,108]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][109][0]==
                                (P_ij[t,0,110]+P_ij[t,0,111])
                                -(P_ij[t,0,109]-self.list_r_x_pu[109][0]*l_ij[t,0,109]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][109][1]==
                                (Q_ij[t,0,110]+Q_ij[t,0,111])
                                -(Q_ij[t,0,109]-self.list_r_x_pu[109][1]*l_ij[t,0,109]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][110][0]==
                                -(P_ij[t,0,110]-self.list_r_x_pu[110][0]*l_ij[t,0,110]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][110][1]==
                                -(Q_ij[t,0,110]-self.list_r_x_pu[110][1]*l_ij[t,0,110]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][111][0]==
                                (P_ij[t,0,112])
                                -(P_ij[t,0,111]-self.list_r_x_pu[111][0]*l_ij[t,0,111]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][111][1]==
                                (Q_ij[t,0,112])
                                -(Q_ij[t,0,111]-self.list_r_x_pu[111][1]*l_ij[t,0,111]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][112][0]==
                                -(P_ij[t,0,112]-self.list_r_x_pu[112][0]*l_ij[t,0,112]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][112][1]==
                                -(Q_ij[t,0,112]-self.list_r_x_pu[112][1]*l_ij[t,0,112]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][113][0] ==
                                (P_ij[t,0,114])
                                -(P_ij[t,0,113]-self.list_r_x_pu[113][0]*l_ij[t,0,113]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][113][1] ==
                                (Q_ij[t,0,114])
                                -(Q_ij[t,0,113]-self.list_r_x_pu[113][1]*l_ij[t,0,113]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][114][0]==
                                (P_ij[t,0,115])
                                -(P_ij[t,0,114]-self.list_r_x_pu[114][0]*l_ij[t,0,114]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][114][1]==
                                (Q_ij[t,0,115])
                                -(Q_ij[t,0,114]-self.list_r_x_pu[114][1]*l_ij[t,0,114]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][115][0]==
                                (P_ij[t,0,116])
                                -(P_ij[t,0,115]-self.list_r_x_pu[115][0]*l_ij[t,0,115]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][115][1]==
                                (Q_ij[t,0,116])
                                -(Q_ij[t,0,115]-self.list_r_x_pu[115][1]*l_ij[t,0,115]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][116][0]==
                                (P_ij[t,0,117])
                                -(P_ij[t,0,116]-self.list_r_x_pu[116][0]*l_ij[t,0,116]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][116][1]==
                                (Q_ij[t,0,117])
                                -(Q_ij[t,0,116]-self.list_r_x_pu[116][1]*l_ij[t,0,116]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][117][0]==
                                -(P_ij[t,0,117]-self.list_r_x_pu[117][0]*l_ij[t,0,117]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][117][1]==
                                -(Q_ij[t,0,117]-self.list_r_x_pu[117][1]*l_ij[t,0,117]))

                model.addConstr((v_i[t,0,0]-v_i[t,0,1]-2*(self.list_r_x_pu[0][0]*P_ij[t,0,0]+self.list_r_x_pu[0][1]*Q_ij[t,0,0])+(self.list_r_x_pu[0][0]*self.list_r_x_pu[0][0]+self.list_r_x_pu[0][1]*self.list_r_x_pu[0][1])*l_ij[t,0,0])==0)

                model.addConstr((v_i[t,0,1]-v_i[t,0,2]-2*(self.list_r_x_pu[1][0]*P_ij[t,0,1]+self.list_r_x_pu[1][1]*Q_ij[t,0,1])+(self.list_r_x_pu[1][0]*self.list_r_x_pu[1][0]+self.list_r_x_pu[1][1]*self.list_r_x_pu[1][1])*l_ij[t,0,1])==0)

                model.addConstr((v_i[t,0,2]-v_i[t,0,3]-2*(self.list_r_x_pu[2][0]*P_ij[t,0,2]+self.list_r_x_pu[2][1]*Q_ij[t,0,2])+(self.list_r_x_pu[2][0]*self.list_r_x_pu[2][0]+self.list_r_x_pu[2][1]*self.list_r_x_pu[2][1])*l_ij[t,0,2])==0)

                model.addConstr((v_i[t,0,2]-v_i[t,0,4]-2*(self.list_r_x_pu[3][0]*P_ij[t,0,3]+self.list_r_x_pu[3][1]*Q_ij[t,0,3])+(self.list_r_x_pu[3][0]*self.list_r_x_pu[3][0]+self.list_r_x_pu[3][1]*self.list_r_x_pu[3][1])*l_ij[t,0,3])==0)

                model.addConstr((v_i[t,0,4]-v_i[t,0,5]-2*(self.list_r_x_pu[4][0]*P_ij[t,0,4]+self.list_r_x_pu[4][1]*Q_ij[t,0,4])+(self.list_r_x_pu[4][0]*self.list_r_x_pu[4][0]+self.list_r_x_pu[4][1]*self.list_r_x_pu[4][1])*l_ij[t,0,4])==0)

                model.addConstr((v_i[t,0,5]-v_i[t,0,6]-2*(self.list_r_x_pu[5][0]*P_ij[t,0,5]+self.list_r_x_pu[5][1]*Q_ij[t,0,5])+(self.list_r_x_pu[5][0]*self.list_r_x_pu[5][0]+self.list_r_x_pu[5][1]*self.list_r_x_pu[5][1])*l_ij[t,0,5])==0)

                model.addConstr((v_i[t,0,6]-v_i[t,0,7]-2*(self.list_r_x_pu[6][0]*P_ij[t,0,6]+self.list_r_x_pu[6][1]*Q_ij[t,0,6])+(self.list_r_x_pu[6][0]*self.list_r_x_pu[6][0]+self.list_r_x_pu[6][1]*self.list_r_x_pu[6][1])*l_ij[t,0,6])==0)

                model.addConstr((v_i[t,0,7]-v_i[t,0,8]-2*(self.list_r_x_pu[7][0]*P_ij[t,0,7]+self.list_r_x_pu[7][1]*Q_ij[t,0,7])+(self.list_r_x_pu[7][0]*self.list_r_x_pu[7][0]+self.list_r_x_pu[7][1]*self.list_r_x_pu[7][1])*l_ij[t,0,7])==0)

                model.addConstr((v_i[t,0,8]-v_i[t,0,9]-2*(self.list_r_x_pu[8][0]*P_ij[t,0,8]+self.list_r_x_pu[8][1]*Q_ij[t,0,8])+(self.list_r_x_pu[8][0]*self.list_r_x_pu[8][0]+self.list_r_x_pu[8][1]*self.list_r_x_pu[8][1])*l_ij[t,0,8])==0)

                model.addConstr((v_i[t,0,2]-v_i[t,0,10]-2*(self.list_r_x_pu[9][0]*P_ij[t,0,9]+self.list_r_x_pu[9][1]*Q_ij[t,0,9])+(self.list_r_x_pu[9][0]*self.list_r_x_pu[9][0]+self.list_r_x_pu[9][1]*self.list_r_x_pu[9][1])*l_ij[t,0,9])==0)

                model.addConstr((v_i[t,0,10]-v_i[t,0,11]-2*(self.list_r_x_pu[10][0]*P_ij[t,0,10]+self.list_r_x_pu[10][1]*Q_ij[t,0,10])+(self.list_r_x_pu[10][0]*self.list_r_x_pu[10][0]+self.list_r_x_pu[10][1]*self.list_r_x_pu[10][1])*l_ij[t,0,10])==0)

                model.addConstr((v_i[t,0,11]-v_i[t,0,12]-2*(self.list_r_x_pu[11][0]*P_ij[t,0,11]+self.list_r_x_pu[11][1]*Q_ij[t,0,11])+(self.list_r_x_pu[11][0]*self.list_r_x_pu[11][0]+self.list_r_x_pu[11][1]*self.list_r_x_pu[11][1])*l_ij[t,0,11])==0)

                model.addConstr((v_i[t,0,12]-v_i[t,0,13]-2*(self.list_r_x_pu[12][0]*P_ij[t,0,12]+self.list_r_x_pu[12][1]*Q_ij[t,0,12])+(self.list_r_x_pu[12][0]*self.list_r_x_pu[12][0]+self.list_r_x_pu[12][1]*self.list_r_x_pu[12][1])*l_ij[t,0,12])==0)

                model.addConstr((v_i[t,0,13]-v_i[t,0,14]-2*(self.list_r_x_pu[13][0]*P_ij[t,0,13]+self.list_r_x_pu[13][1]*Q_ij[t,0,13])+(self.list_r_x_pu[13][0]*self.list_r_x_pu[13][0]+self.list_r_x_pu[13][1]*self.list_r_x_pu[13][1])*l_ij[t,0,13])==0)

                model.addConstr((v_i[t,0,14]-v_i[t,0,15]-2*(self.list_r_x_pu[14][0]*P_ij[t,0,14]+self.list_r_x_pu[14][1]*Q_ij[t,0,14])+(self.list_r_x_pu[14][0]*self.list_r_x_pu[14][0]+self.list_r_x_pu[14][1]*self.list_r_x_pu[14][1])*l_ij[t,0,14])==0)

                model.addConstr((v_i[t,0,15]-v_i[t,0,16]-2*(self.list_r_x_pu[15][0]*P_ij[t,0,15]+self.list_r_x_pu[15][1]*Q_ij[t,0,15])+(self.list_r_x_pu[15][0]*self.list_r_x_pu[15][0]+self.list_r_x_pu[15][1]*self.list_r_x_pu[15][1])*l_ij[t,0,15])==0)

                model.addConstr((v_i[t,0,16]-v_i[t,0,17]-2*(self.list_r_x_pu[16][0]*P_ij[t,0,16]+self.list_r_x_pu[16][1]*Q_ij[t,0,16])+(self.list_r_x_pu[16][0]*self.list_r_x_pu[16][0]+self.list_r_x_pu[16][1]*self.list_r_x_pu[16][1])*l_ij[t,0,16])==0)

                model.addConstr((v_i[t,0,11]-v_i[t,0,18]-2*(self.list_r_x_pu[17][0]*P_ij[t,0,17]+self.list_r_x_pu[17][1]*Q_ij[t,0,17])+(self.list_r_x_pu[17][0]*self.list_r_x_pu[17][0]+self.list_r_x_pu[17][1]*self.list_r_x_pu[17][1])*l_ij[t,0,17])==0)

                model.addConstr((v_i[t,0,18]-v_i[t,0,19]-2*(self.list_r_x_pu[18][0]*P_ij[t,0,18]+self.list_r_x_pu[18][1]*Q_ij[t,0,18])+(self.list_r_x_pu[18][0]*self.list_r_x_pu[18][0]+self.list_r_x_pu[18][1]*self.list_r_x_pu[18][1])*l_ij[t,0,18])==0)

                model.addConstr((v_i[t,0,19]-v_i[t,0,20]-2*(self.list_r_x_pu[19][0]*P_ij[t,0,19]+self.list_r_x_pu[19][1]*Q_ij[t,0,19])+(self.list_r_x_pu[19][0]*self.list_r_x_pu[19][0]+self.list_r_x_pu[19][1]*self.list_r_x_pu[19][1])*l_ij[t,0,19])==0)

                model.addConstr((v_i[t,0,20]-v_i[t,0,21]-2*(self.list_r_x_pu[20][0]*P_ij[t,0,20]+self.list_r_x_pu[20][1]*Q_ij[t,0,20])+(self.list_r_x_pu[20][0]*self.list_r_x_pu[20][0]+self.list_r_x_pu[20][1]*self.list_r_x_pu[20][1])*l_ij[t,0,20])==0)

                model.addConstr((v_i[t,0,21]-v_i[t,0,22]-2*(self.list_r_x_pu[21][0]*P_ij[t,0,21]+self.list_r_x_pu[21][1]*Q_ij[t,0,21])+(self.list_r_x_pu[21][0]*self.list_r_x_pu[21][0]+self.list_r_x_pu[21][1]*self.list_r_x_pu[21][1])*l_ij[t,0,21])==0)

                model.addConstr((v_i[t,0,22]-v_i[t,0,23]-2*(self.list_r_x_pu[22][0]*P_ij[t,0,22]+self.list_r_x_pu[22][1]*Q_ij[t,0,22])+(self.list_r_x_pu[22][0]*self.list_r_x_pu[22][0]+self.list_r_x_pu[22][1]*self.list_r_x_pu[22][1])*l_ij[t,0,22])==0)

                model.addConstr((v_i[t,0,23]-v_i[t,0,24]-2*(self.list_r_x_pu[23][0]*P_ij[t,0,23]+self.list_r_x_pu[23][1]*Q_ij[t,0,23])+(self.list_r_x_pu[23][0]*self.list_r_x_pu[23][0]+self.list_r_x_pu[23][1]*self.list_r_x_pu[23][1])*l_ij[t,0,23])==0)

                model.addConstr((v_i[t,0,24]-v_i[t,0,25]-2*(self.list_r_x_pu[24][0]*P_ij[t,0,24]+self.list_r_x_pu[24][1]*Q_ij[t,0,24])+(self.list_r_x_pu[24][0]*self.list_r_x_pu[24][0]+self.list_r_x_pu[24][1]*self.list_r_x_pu[24][1])*l_ij[t,0,24])==0)

                model.addConstr((v_i[t,0,25]-v_i[t,0,26]-2*(self.list_r_x_pu[25][0]*P_ij[t,0,25]+self.list_r_x_pu[25][1]*Q_ij[t,0,25])+(self.list_r_x_pu[25][0]*self.list_r_x_pu[25][0]+self.list_r_x_pu[25][1]*self.list_r_x_pu[25][1])*l_ij[t,0,25])==0)

                model.addConstr((v_i[t,0,26]-v_i[t,0,27]-2*(self.list_r_x_pu[26][0]*P_ij[t,0,26]+self.list_r_x_pu[26][1]*Q_ij[t,0,26])+(self.list_r_x_pu[26][0]*self.list_r_x_pu[26][0]+self.list_r_x_pu[26][1]*self.list_r_x_pu[26][1])*l_ij[t,0,26])==0)

                model.addConstr((v_i[t,0,4]-v_i[t,0,28]-2*(self.list_r_x_pu[27][0]*P_ij[t,0,27]+self.list_r_x_pu[27][1]*Q_ij[t,0,27])+(self.list_r_x_pu[27][0]*self.list_r_x_pu[27][0]+self.list_r_x_pu[27][1]*self.list_r_x_pu[27][1])*l_ij[t,0,27])==0)

                model.addConstr((v_i[t,0,28]-v_i[t,0,29]-2*(self.list_r_x_pu[28][0]*P_ij[t,0,28]+self.list_r_x_pu[28][1]*Q_ij[t,0,28])+(self.list_r_x_pu[28][0]*self.list_r_x_pu[28][0]+self.list_r_x_pu[28][1]*self.list_r_x_pu[28][1])*l_ij[t,0,28])==0)

                model.addConstr((v_i[t,0,29]-v_i[t,0,30]-2*(self.list_r_x_pu[29][0]*P_ij[t,0,29]+self.list_r_x_pu[29][1]*Q_ij[t,0,29])+(self.list_r_x_pu[29][0]*self.list_r_x_pu[29][0]+self.list_r_x_pu[29][1]*self.list_r_x_pu[29][1])*l_ij[t,0,29])==0)

                model.addConstr((v_i[t,0,30]-v_i[t,0,31]-2*(self.list_r_x_pu[30][0]*P_ij[t,0,30]+self.list_r_x_pu[30][1]*Q_ij[t,0,30])+(self.list_r_x_pu[30][0]*self.list_r_x_pu[30][0]+self.list_r_x_pu[30][1]*self.list_r_x_pu[30][1])*l_ij[t,0,30])==0)

                model.addConstr((v_i[t,0,31]-v_i[t,0,32]-2*(self.list_r_x_pu[31][0]*P_ij[t,0,31]+self.list_r_x_pu[31][1]*Q_ij[t,0,31])+(self.list_r_x_pu[31][0]*self.list_r_x_pu[31][0]+self.list_r_x_pu[31][1]*self.list_r_x_pu[31][1])*l_ij[t,0,31])==0)

                model.addConstr((v_i[t,0,32]-v_i[t,0,33]-2*(self.list_r_x_pu[32][0]*P_ij[t,0,32]+self.list_r_x_pu[32][1]*Q_ij[t,0,32])+(self.list_r_x_pu[32][0]*self.list_r_x_pu[32][0]+self.list_r_x_pu[32][1]*self.list_r_x_pu[32][1])*l_ij[t,0,32])==0)

                model.addConstr((v_i[t,0,33]-v_i[t,0,34]-2*(self.list_r_x_pu[33][0]*P_ij[t,0,33]+self.list_r_x_pu[33][1]*Q_ij[t,0,33])+(self.list_r_x_pu[33][0]*self.list_r_x_pu[33][0]+self.list_r_x_pu[33][1]*self.list_r_x_pu[33][1])*l_ij[t,0,33])==0)

                model.addConstr((v_i[t,0,34]-v_i[t,0,35]-2*(self.list_r_x_pu[34][0]*P_ij[t,0,34]+self.list_r_x_pu[34][1]*Q_ij[t,0,34])+(self.list_r_x_pu[34][0]*self.list_r_x_pu[34][0]+self.list_r_x_pu[34][1]*self.list_r_x_pu[34][1])*l_ij[t,0,34])==0)

                model.addConstr((v_i[t,0,30]-v_i[t,0,36]-2*(self.list_r_x_pu[35][0]*P_ij[t,0,35]+self.list_r_x_pu[35][1]*Q_ij[t,0,35])+(self.list_r_x_pu[35][0]*self.list_r_x_pu[35][0]+self.list_r_x_pu[35][1]*self.list_r_x_pu[35][1])*l_ij[t,0,35])==0)

                model.addConstr((v_i[t,0,36]-v_i[t,0,37]-2*(self.list_r_x_pu[36][0]*P_ij[t,0,36]+self.list_r_x_pu[36][1]*Q_ij[t,0,36])+(self.list_r_x_pu[36][0]*self.list_r_x_pu[36][0]+self.list_r_x_pu[36][1]*self.list_r_x_pu[36][1])*l_ij[t,0,36])==0)

                model.addConstr((v_i[t,0,29]-v_i[t,0,38]-2*(self.list_r_x_pu[37][0]*P_ij[t,0,37]+self.list_r_x_pu[37][1]*Q_ij[t,0,37])+(self.list_r_x_pu[37][0]*self.list_r_x_pu[37][0]+self.list_r_x_pu[37][1]*self.list_r_x_pu[37][1])*l_ij[t,0,37])==0)

                model.addConstr((v_i[t,0,38]-v_i[t,0,39]-2*(self.list_r_x_pu[38][0]*P_ij[t,0,38]+self.list_r_x_pu[38][1]*Q_ij[t,0,38])+(self.list_r_x_pu[38][0]*self.list_r_x_pu[38][0]+self.list_r_x_pu[38][1]*self.list_r_x_pu[38][1])*l_ij[t,0,38])==0)

                model.addConstr((v_i[t,0,39]-v_i[t,0,40]-2*(self.list_r_x_pu[39][0]*P_ij[t,0,39]+self.list_r_x_pu[39][1]*Q_ij[t,0,39])+(self.list_r_x_pu[39][0]*self.list_r_x_pu[39][0]+self.list_r_x_pu[39][1]*self.list_r_x_pu[39][1])*l_ij[t,0,39])==0)

                model.addConstr((v_i[t,0,40]-v_i[t,0,41]-2*(self.list_r_x_pu[40][0]*P_ij[t,0,40]+self.list_r_x_pu[40][1]*Q_ij[t,0,40])+(self.list_r_x_pu[40][0]*self.list_r_x_pu[40][0]+self.list_r_x_pu[40][1]*self.list_r_x_pu[40][1])*l_ij[t,0,40])==0)

                model.addConstr((v_i[t,0,41]-v_i[t,0,42]-2*(self.list_r_x_pu[41][0]*P_ij[t,0,41]+self.list_r_x_pu[41][1]*Q_ij[t,0,41])+(self.list_r_x_pu[41][0]*self.list_r_x_pu[41][0]+self.list_r_x_pu[41][1]*self.list_r_x_pu[41][1])*l_ij[t,0,41])==0)

                model.addConstr((v_i[t,0,42]-v_i[t,0,43]-2*(self.list_r_x_pu[42][0]*P_ij[t,0,42]+self.list_r_x_pu[42][1]*Q_ij[t,0,42])+(self.list_r_x_pu[42][0]*self.list_r_x_pu[42][0]+self.list_r_x_pu[42][1]*self.list_r_x_pu[42][1])*l_ij[t,0,42])==0)

                model.addConstr((v_i[t,0,43]-v_i[t,0,44]-2*(self.list_r_x_pu[43][0]*P_ij[t,0,43]+self.list_r_x_pu[43][1]*Q_ij[t,0,43])+(self.list_r_x_pu[43][0]*self.list_r_x_pu[43][0]+self.list_r_x_pu[43][1]*self.list_r_x_pu[43][1])*l_ij[t,0,43])==0)

                model.addConstr((v_i[t,0,44]-v_i[t,0,45]-2*(self.list_r_x_pu[44][0]*P_ij[t,0,44]+self.list_r_x_pu[44][1]*Q_ij[t,0,44])+(self.list_r_x_pu[44][0]*self.list_r_x_pu[44][0]+self.list_r_x_pu[44][1]*self.list_r_x_pu[44][1])*l_ij[t,0,44])==0)

                model.addConstr((v_i[t,0,45]-v_i[t,0,46]-2*(self.list_r_x_pu[45][0]*P_ij[t,0,45]+self.list_r_x_pu[45][1]*Q_ij[t,0,45])+(self.list_r_x_pu[45][0]*self.list_r_x_pu[45][0]+self.list_r_x_pu[45][1]*self.list_r_x_pu[45][1])*l_ij[t,0,45])==0)

                model.addConstr((v_i[t,0,35]-v_i[t,0,47]-2*(self.list_r_x_pu[46][0]*P_ij[t,0,46]+self.list_r_x_pu[46][1]*Q_ij[t,0,46])+(self.list_r_x_pu[46][0]*self.list_r_x_pu[46][0]+self.list_r_x_pu[46][1]*self.list_r_x_pu[46][1])*l_ij[t,0,46])==0)

                model.addConstr((v_i[t,0,47]-v_i[t,0,48]-2*(self.list_r_x_pu[47][0]*P_ij[t,0,47]+self.list_r_x_pu[47][1]*Q_ij[t,0,47])+(self.list_r_x_pu[47][0]*self.list_r_x_pu[47][0]+self.list_r_x_pu[47][1]*self.list_r_x_pu[47][1])*l_ij[t,0,47])==0)

                model.addConstr((v_i[t,0,48]-v_i[t,0,49]-2*(self.list_r_x_pu[48][0]*P_ij[t,0,48]+self.list_r_x_pu[48][1]*Q_ij[t,0,48])+(self.list_r_x_pu[48][0]*self.list_r_x_pu[48][0]+self.list_r_x_pu[48][1]*self.list_r_x_pu[48][1])*l_ij[t,0,48])==0)

                model.addConstr((v_i[t,0,49]-v_i[t,0,50]-2*(self.list_r_x_pu[49][0]*P_ij[t,0,49]+self.list_r_x_pu[49][1]*Q_ij[t,0,49])+(self.list_r_x_pu[49][0]*self.list_r_x_pu[49][0]+self.list_r_x_pu[49][1]*self.list_r_x_pu[49][1])*l_ij[t,0,49])==0)

                model.addConstr((v_i[t,0,50]-v_i[t,0,51]-2*(self.list_r_x_pu[50][0]*P_ij[t,0,50]+self.list_r_x_pu[50][1]*Q_ij[t,0,50])+(self.list_r_x_pu[50][0]*self.list_r_x_pu[50][0]+self.list_r_x_pu[50][1]*self.list_r_x_pu[50][1])*l_ij[t,0,50])==0)

                model.addConstr((v_i[t,0,51]-v_i[t,0,52]-2*(self.list_r_x_pu[51][0]*P_ij[t,0,51]+self.list_r_x_pu[51][1]*Q_ij[t,0,51])+(self.list_r_x_pu[51][0]*self.list_r_x_pu[51][0]+self.list_r_x_pu[51][1]*self.list_r_x_pu[51][1])*l_ij[t,0,51])==0)

                model.addConstr((v_i[t,0,52]-v_i[t,0,53]-2*(self.list_r_x_pu[52][0]*P_ij[t,0,52]+self.list_r_x_pu[52][1]*Q_ij[t,0,52])+(self.list_r_x_pu[52][0]*self.list_r_x_pu[52][0]+self.list_r_x_pu[52][1]*self.list_r_x_pu[52][1])*l_ij[t,0,52])==0)

                model.addConstr((v_i[t,0,53]-v_i[t,0,54]-2*(self.list_r_x_pu[53][0]*P_ij[t,0,53]+self.list_r_x_pu[53][1]*Q_ij[t,0,53])+(self.list_r_x_pu[53][0]*self.list_r_x_pu[53][0]+self.list_r_x_pu[53][1]*self.list_r_x_pu[53][1])*l_ij[t,0,53])==0)

                model.addConstr((v_i[t,0,29]-v_i[t,0,55]-2*(self.list_r_x_pu[54][0]*P_ij[t,0,54]+self.list_r_x_pu[54][1]*Q_ij[t,0,54])+(self.list_r_x_pu[54][0]*self.list_r_x_pu[54][0]+self.list_r_x_pu[54][1]*self.list_r_x_pu[54][1])*l_ij[t,0,54])==0)

                model.addConstr((v_i[t,0,55]-v_i[t,0,56]-2*(self.list_r_x_pu[55][0]*P_ij[t,0,55]+self.list_r_x_pu[55][1]*Q_ij[t,0,55])+(self.list_r_x_pu[55][0]*self.list_r_x_pu[55][0]+self.list_r_x_pu[55][1]*self.list_r_x_pu[55][1])*l_ij[t,0,55])==0)

                model.addConstr((v_i[t,0,56]-v_i[t,0,57]-2*(self.list_r_x_pu[56][0]*P_ij[t,0,56]+self.list_r_x_pu[56][1]*Q_ij[t,0,56])+(self.list_r_x_pu[56][0]*self.list_r_x_pu[56][0]+self.list_r_x_pu[56][1]*self.list_r_x_pu[56][1])*l_ij[t,0,56])==0)

                model.addConstr((v_i[t,0,57]-v_i[t,0,58]-2*(self.list_r_x_pu[57][0]*P_ij[t,0,57]+self.list_r_x_pu[57][1]*Q_ij[t,0,57])+(self.list_r_x_pu[57][0]*self.list_r_x_pu[57][0]+self.list_r_x_pu[57][1]*self.list_r_x_pu[57][1])*l_ij[t,0,57])==0)

                model.addConstr((v_i[t,0,58]-v_i[t,0,59]-2*(self.list_r_x_pu[58][0]*P_ij[t,0,58]+self.list_r_x_pu[58][1]*Q_ij[t,0,58])+(self.list_r_x_pu[58][0]*self.list_r_x_pu[58][0]+self.list_r_x_pu[58][1]*self.list_r_x_pu[58][1])*l_ij[t,0,58])==0)

                model.addConstr((v_i[t,0,59]-v_i[t,0,60]-2*(self.list_r_x_pu[59][0]*P_ij[t,0,59]+self.list_r_x_pu[59][1]*Q_ij[t,0,59])+(self.list_r_x_pu[59][0]*self.list_r_x_pu[59][0]+self.list_r_x_pu[59][1]*self.list_r_x_pu[59][1])*l_ij[t,0,59])==0)

                model.addConstr((v_i[t,0,60]-v_i[t,0,61]-2*(self.list_r_x_pu[60][0]*P_ij[t,0,60]+self.list_r_x_pu[60][1]*Q_ij[t,0,60])+(self.list_r_x_pu[60][0]*self.list_r_x_pu[60][0]+self.list_r_x_pu[60][1]*self.list_r_x_pu[60][1])*l_ij[t,0,60])==0)

                model.addConstr((v_i[t,0,61]-v_i[t,0,62]-2*(self.list_r_x_pu[61][0]*P_ij[t,0,61]+self.list_r_x_pu[61][1]*Q_ij[t,0,61])+(self.list_r_x_pu[61][0]*self.list_r_x_pu[61][0]+self.list_r_x_pu[61][1]*self.list_r_x_pu[61][1])*l_ij[t,0,61])==0)

                model.addConstr((v_i[t,0,1]-v_i[t,0,63]-2*(self.list_r_x_pu[62][0]*P_ij[t,0,62]+self.list_r_x_pu[62][1]*Q_ij[t,0,62])+(self.list_r_x_pu[62][0]*self.list_r_x_pu[62][0]+self.list_r_x_pu[62][1]*self.list_r_x_pu[62][1])*l_ij[t,0,62])==0)

                model.addConstr((v_i[t,0,63]-v_i[t,0,64]-2*(self.list_r_x_pu[63][0]*P_ij[t,0,63]+self.list_r_x_pu[63][1]*Q_ij[t,0,63])+(self.list_r_x_pu[63][0]*self.list_r_x_pu[63][0]+self.list_r_x_pu[63][1]*self.list_r_x_pu[63][1])*l_ij[t,0,63])==0)

                model.addConstr((v_i[t,0,64]-v_i[t,0,65]-2*(self.list_r_x_pu[64][0]*P_ij[t,0,64]+self.list_r_x_pu[64][1]*Q_ij[t,0,64])+(self.list_r_x_pu[64][0]*self.list_r_x_pu[64][0]+self.list_r_x_pu[64][1]*self.list_r_x_pu[64][1])*l_ij[t,0,64])==0)

                model.addConstr((v_i[t,0,65]-v_i[t,0,66]-2*(self.list_r_x_pu[65][0]*P_ij[t,0,65]+self.list_r_x_pu[65][1]*Q_ij[t,0,65])+(self.list_r_x_pu[65][0]*self.list_r_x_pu[65][0]+self.list_r_x_pu[65][1]*self.list_r_x_pu[65][1])*l_ij[t,0,65])==0)

                model.addConstr((v_i[t,0,66]-v_i[t,0,67]-2*(self.list_r_x_pu[66][0]*P_ij[t,0,66]+self.list_r_x_pu[66][1]*Q_ij[t,0,66])+(self.list_r_x_pu[66][0]*self.list_r_x_pu[66][0]+self.list_r_x_pu[66][1]*self.list_r_x_pu[66][1])*l_ij[t,0,66])==0)

                model.addConstr((v_i[t,0,67]-v_i[t,0,68]-2*(self.list_r_x_pu[67][0]*P_ij[t,0,67]+self.list_r_x_pu[67][1]*Q_ij[t,0,67])+(self.list_r_x_pu[67][0]*self.list_r_x_pu[67][0]+self.list_r_x_pu[67][1]*self.list_r_x_pu[67][1])*l_ij[t,0,67])==0)

                model.addConstr((v_i[t,0,68]-v_i[t,0,69]-2*(self.list_r_x_pu[68][0]*P_ij[t,0,68]+self.list_r_x_pu[68][1]*Q_ij[t,0,68])+(self.list_r_x_pu[68][0]*self.list_r_x_pu[68][0]+self.list_r_x_pu[68][1]*self.list_r_x_pu[68][1])*l_ij[t,0,68])==0)

                model.addConstr((v_i[t,0,69]-v_i[t,0,70]-2*(self.list_r_x_pu[69][0]*P_ij[t,0,69]+self.list_r_x_pu[69][1]*Q_ij[t,0,69])+(self.list_r_x_pu[69][0]*self.list_r_x_pu[69][0]+self.list_r_x_pu[69][1]*self.list_r_x_pu[69][1])*l_ij[t,0,69])==0)

                model.addConstr((v_i[t,0,70]-v_i[t,0,71]-2*(self.list_r_x_pu[70][0]*P_ij[t,0,70]+self.list_r_x_pu[70][1]*Q_ij[t,0,70])+(self.list_r_x_pu[70][0]*self.list_r_x_pu[70][0]+self.list_r_x_pu[70][1]*self.list_r_x_pu[70][1])*l_ij[t,0,70])==0)

                model.addConstr((v_i[t,0,71]-v_i[t,0,72]-2*(self.list_r_x_pu[71][0]*P_ij[t,0,71]+self.list_r_x_pu[71][1]*Q_ij[t,0,71])+(self.list_r_x_pu[71][0]*self.list_r_x_pu[71][0]+self.list_r_x_pu[71][1]*self.list_r_x_pu[71][1])*l_ij[t,0,71])==0)

                model.addConstr((v_i[t,0,72]-v_i[t,0,73]-2*(self.list_r_x_pu[72][0]*P_ij[t,0,72]+self.list_r_x_pu[72][1]*Q_ij[t,0,72])+(self.list_r_x_pu[72][0]*self.list_r_x_pu[72][0]+self.list_r_x_pu[72][1]*self.list_r_x_pu[72][1])*l_ij[t,0,72])==0)

                model.addConstr((v_i[t,0,73]-v_i[t,0,74]-2*(self.list_r_x_pu[73][0]*P_ij[t,0,73]+self.list_r_x_pu[73][1]*Q_ij[t,0,73])+(self.list_r_x_pu[73][0]*self.list_r_x_pu[73][0]+self.list_r_x_pu[73][1]*self.list_r_x_pu[73][1])*l_ij[t,0,73])==0)

                model.addConstr((v_i[t,0,74]-v_i[t,0,75]-2*(self.list_r_x_pu[74][0]*P_ij[t,0,74]+self.list_r_x_pu[74][1]*Q_ij[t,0,74])+(self.list_r_x_pu[74][0]*self.list_r_x_pu[74][0]+self.list_r_x_pu[74][1]*self.list_r_x_pu[74][1])*l_ij[t,0,74])==0)

                model.addConstr((v_i[t,0,75]-v_i[t,0,76]-2*(self.list_r_x_pu[75][0]*P_ij[t,0,75]+self.list_r_x_pu[75][1]*Q_ij[t,0,75])+(self.list_r_x_pu[75][0]*self.list_r_x_pu[75][0]+self.list_r_x_pu[75][1]*self.list_r_x_pu[75][1])*l_ij[t,0,75])==0)

                model.addConstr((v_i[t,0,76]-v_i[t,0,77]-2*(self.list_r_x_pu[76][0]*P_ij[t,0,76]+self.list_r_x_pu[76][1]*Q_ij[t,0,76])+(self.list_r_x_pu[76][0]*self.list_r_x_pu[76][0]+self.list_r_x_pu[76][1]*self.list_r_x_pu[76][1])*l_ij[t,0,76])==0)

                model.addConstr((v_i[t,0,64]-v_i[t,0,78]-2*(self.list_r_x_pu[77][0]*P_ij[t,0,77]+self.list_r_x_pu[77][1]*Q_ij[t,0,77])+(self.list_r_x_pu[77][0]*self.list_r_x_pu[77][0]+self.list_r_x_pu[77][1]*self.list_r_x_pu[77][1])*l_ij[t,0,77])==0)

                model.addConstr((v_i[t,0,78]-v_i[t,0,79]-2*(self.list_r_x_pu[78][0]*P_ij[t,0,78]+self.list_r_x_pu[78][1]*Q_ij[t,0,78])+(self.list_r_x_pu[78][0]*self.list_r_x_pu[78][0]+self.list_r_x_pu[78][1]*self.list_r_x_pu[78][1])*l_ij[t,0,78])==0)

                model.addConstr((v_i[t,0,79]-v_i[t,0,80]-2*(self.list_r_x_pu[79][0]*P_ij[t,0,79]+self.list_r_x_pu[79][1]*Q_ij[t,0,79])+(self.list_r_x_pu[79][0]*self.list_r_x_pu[79][0]+self.list_r_x_pu[79][1]*self.list_r_x_pu[79][1])*l_ij[t,0,79])==0)

                model.addConstr((v_i[t,0,80]-v_i[t,0,81]-2*(self.list_r_x_pu[80][0]*P_ij[t,0,80]+self.list_r_x_pu[80][1]*Q_ij[t,0,80])+(self.list_r_x_pu[80][0]*self.list_r_x_pu[80][0]+self.list_r_x_pu[80][1]*self.list_r_x_pu[80][1])*l_ij[t,0,80])==0)

                model.addConstr((v_i[t,0,81]-v_i[t,0,82]-2*(self.list_r_x_pu[81][0]*P_ij[t,0,81]+self.list_r_x_pu[81][1]*Q_ij[t,0,81])+(self.list_r_x_pu[81][0]*self.list_r_x_pu[81][0]+self.list_r_x_pu[81][1]*self.list_r_x_pu[81][1])*l_ij[t,0,81])==0)

                model.addConstr((v_i[t,0,82]-v_i[t,0,83]-2*(self.list_r_x_pu[82][0]*P_ij[t,0,82]+self.list_r_x_pu[82][1]*Q_ij[t,0,82])+(self.list_r_x_pu[82][0]*self.list_r_x_pu[82][0]+self.list_r_x_pu[82][1]*self.list_r_x_pu[82][1])*l_ij[t,0,82])==0)

                model.addConstr((v_i[t,0,83]-v_i[t,0,84]-2*(self.list_r_x_pu[83][0]*P_ij[t,0,83]+self.list_r_x_pu[83][1]*Q_ij[t,0,83])+(self.list_r_x_pu[83][0]*self.list_r_x_pu[83][0]+self.list_r_x_pu[83][1]*self.list_r_x_pu[83][1])*l_ij[t,0,83])==0)

                model.addConstr((v_i[t,0,84]-v_i[t,0,85]-2*(self.list_r_x_pu[84][0]*P_ij[t,0,84]+self.list_r_x_pu[84][1]*Q_ij[t,0,84])+(self.list_r_x_pu[84][0]*self.list_r_x_pu[84][0]+self.list_r_x_pu[84][1]*self.list_r_x_pu[84][1])*l_ij[t,0,84])==0)

                model.addConstr((v_i[t,0,79]-v_i[t,0,86]-2*(self.list_r_x_pu[85][0]*P_ij[t,0,85]+self.list_r_x_pu[85][1]*Q_ij[t,0,85])+(self.list_r_x_pu[85][0]*self.list_r_x_pu[85][0]+self.list_r_x_pu[85][1]*self.list_r_x_pu[85][1])*l_ij[t,0,85])==0)

                model.addConstr((v_i[t,0,86]-v_i[t,0,87]-2*(self.list_r_x_pu[86][0]*P_ij[t,0,86]+self.list_r_x_pu[86][1]*Q_ij[t,0,86])+(self.list_r_x_pu[86][0]*self.list_r_x_pu[86][0]+self.list_r_x_pu[86][1]*self.list_r_x_pu[86][1])*l_ij[t,0,86])==0)

                model.addConstr((v_i[t,0,87]-v_i[t,0,88]-2*(self.list_r_x_pu[87][0]*P_ij[t,0,87]+self.list_r_x_pu[87][1]*Q_ij[t,0,87])+(self.list_r_x_pu[87][0]*self.list_r_x_pu[87][0]+self.list_r_x_pu[87][1]*self.list_r_x_pu[87][1])*l_ij[t,0,87])==0)

                model.addConstr((v_i[t,0,65]-v_i[t,0,89]-2*(self.list_r_x_pu[88][0]*P_ij[t,0,88]+self.list_r_x_pu[88][1]*Q_ij[t,0,88])+(self.list_r_x_pu[88][0]*self.list_r_x_pu[88][0]+self.list_r_x_pu[88][1]*self.list_r_x_pu[88][1])*l_ij[t,0,88])==0)

                model.addConstr((v_i[t,0,89]-v_i[t,0,90]-2*(self.list_r_x_pu[89][0]*P_ij[t,0,89]+self.list_r_x_pu[89][1]*Q_ij[t,0,89])+(self.list_r_x_pu[89][0]*self.list_r_x_pu[89][0]+self.list_r_x_pu[89][1]*self.list_r_x_pu[89][1])*l_ij[t,0,89])==0)

                model.addConstr((v_i[t,0,90]-v_i[t,0,91]-2*(self.list_r_x_pu[90][0]*P_ij[t,0,90]+self.list_r_x_pu[90][1]*Q_ij[t,0,90])+(self.list_r_x_pu[90][0]*self.list_r_x_pu[90][0]+self.list_r_x_pu[90][1]*self.list_r_x_pu[90][1])*l_ij[t,0,90])==0)

                model.addConstr((v_i[t,0,91]-v_i[t,0,92]-2*(self.list_r_x_pu[91][0]*P_ij[t,0,91]+self.list_r_x_pu[91][1]*Q_ij[t,0,91])+(self.list_r_x_pu[91][0]*self.list_r_x_pu[91][0]+self.list_r_x_pu[91][1]*self.list_r_x_pu[91][1])*l_ij[t,0,91])==0)

                model.addConstr((v_i[t,0,92]-v_i[t,0,93]-2*(self.list_r_x_pu[92][0]*P_ij[t,0,92]+self.list_r_x_pu[92][1]*Q_ij[t,0,92])+(self.list_r_x_pu[92][0]*self.list_r_x_pu[92][0]+self.list_r_x_pu[92][1]*self.list_r_x_pu[92][1])*l_ij[t,0,92])==0)

                model.addConstr((v_i[t,0,93]-v_i[t,0,94]-2*(self.list_r_x_pu[93][0]*P_ij[t,0,93]+self.list_r_x_pu[93][1]*Q_ij[t,0,93])+(self.list_r_x_pu[93][0]*self.list_r_x_pu[93][0]+self.list_r_x_pu[93][1]*self.list_r_x_pu[93][1])*l_ij[t,0,93])==0)

                model.addConstr((v_i[t,0,94]-v_i[t,0,95]-2*(self.list_r_x_pu[94][0]*P_ij[t,0,94]+self.list_r_x_pu[94][1]*Q_ij[t,0,94])+(self.list_r_x_pu[94][0]*self.list_r_x_pu[94][0]+self.list_r_x_pu[94][1]*self.list_r_x_pu[94][1])*l_ij[t,0,94])==0)

                model.addConstr((v_i[t,0,91]-v_i[t,0,96]-2*(self.list_r_x_pu[95][0]*P_ij[t,0,95]+self.list_r_x_pu[95][1]*Q_ij[t,0,95])+(self.list_r_x_pu[95][0]*self.list_r_x_pu[95][0]+self.list_r_x_pu[95][1]*self.list_r_x_pu[95][1])*l_ij[t,0,95])==0)

                model.addConstr((v_i[t,0,96]-v_i[t,0,97]-2*(self.list_r_x_pu[96][0]*P_ij[t,0,96]+self.list_r_x_pu[96][1]*Q_ij[t,0,96])+(self.list_r_x_pu[96][0]*self.list_r_x_pu[96][0]+self.list_r_x_pu[96][1]*self.list_r_x_pu[96][1])*l_ij[t,0,96])==0)

                model.addConstr((v_i[t,0,97]-v_i[t,0,98]-2*(self.list_r_x_pu[97][0]*P_ij[t,0,97]+self.list_r_x_pu[97][1]*Q_ij[t,0,97])+(self.list_r_x_pu[97][0]*self.list_r_x_pu[97][0]+self.list_r_x_pu[97][1]*self.list_r_x_pu[97][1])*l_ij[t,0,97])==0)

                model.addConstr((v_i[t,0,98]-v_i[t,0,99]-2*(self.list_r_x_pu[98][0]*P_ij[t,0,98]+self.list_r_x_pu[98][1]*Q_ij[t,0,98])+(self.list_r_x_pu[98][0]*self.list_r_x_pu[98][0]+self.list_r_x_pu[98][1]*self.list_r_x_pu[98][1])*l_ij[t,0,98])==0)

                model.addConstr((v_i[t,0,1]-v_i[t,0,100]-2*(self.list_r_x_pu[99][0]*P_ij[t,0,99]+self.list_r_x_pu[99][1]*Q_ij[t,0,99])+(self.list_r_x_pu[99][0]*self.list_r_x_pu[99][0]+self.list_r_x_pu[99][1]*self.list_r_x_pu[99][1])*l_ij[t,0,99])==0)

                model.addConstr((v_i[t,0,100]-v_i[t,0,101]-2*(self.list_r_x_pu[100][0]*P_ij[t,0,100]+self.list_r_x_pu[100][1]*Q_ij[t,0,100])+(self.list_r_x_pu[100][0]*self.list_r_x_pu[100][0]+self.list_r_x_pu[100][1]*self.list_r_x_pu[100][1])*l_ij[t,0,100])==0)

                model.addConstr((v_i[t,0,101]-v_i[t,0,102]-2*(self.list_r_x_pu[101][0]*P_ij[t,0,101]+self.list_r_x_pu[101][1]*Q_ij[t,0,101])+(self.list_r_x_pu[101][0]*self.list_r_x_pu[101][0]+self.list_r_x_pu[101][1]*self.list_r_x_pu[101][1])*l_ij[t,0,101])==0)

                model.addConstr((v_i[t,0,102]-v_i[t,0,103]-2*(self.list_r_x_pu[102][0]*P_ij[t,0,102]+self.list_r_x_pu[102][1]*Q_ij[t,0,102])+(self.list_r_x_pu[102][0]*self.list_r_x_pu[102][0]+self.list_r_x_pu[102][1]*self.list_r_x_pu[102][1])*l_ij[t,0,102])==0)

                model.addConstr((v_i[t,0,103]-v_i[t,0,104]-2*(self.list_r_x_pu[103][0]*P_ij[t,0,103]+self.list_r_x_pu[103][1]*Q_ij[t,0,103])+(self.list_r_x_pu[103][0]*self.list_r_x_pu[103][0]+self.list_r_x_pu[103][1]*self.list_r_x_pu[103][1])*l_ij[t,0,103])==0)

                model.addConstr((v_i[t,0,104]-v_i[t,0,105]-2*(self.list_r_x_pu[104][0]*P_ij[t,0,104]+self.list_r_x_pu[104][1]*Q_ij[t,0,104])+(self.list_r_x_pu[104][0]*self.list_r_x_pu[104][0]+self.list_r_x_pu[104][1]*self.list_r_x_pu[104][1])*l_ij[t,0,104])==0)

                model.addConstr((v_i[t,0,105]-v_i[t,0,106]-2*(self.list_r_x_pu[105][0]*P_ij[t,0,105]+self.list_r_x_pu[105][1]*Q_ij[t,0,105])+(self.list_r_x_pu[105][0]*self.list_r_x_pu[105][0]+self.list_r_x_pu[105][1]*self.list_r_x_pu[105][1])*l_ij[t,0,105])==0)

                model.addConstr((v_i[t,0,106]-v_i[t,0,107]-2*(self.list_r_x_pu[106][0]*P_ij[t,0,106]+self.list_r_x_pu[106][1]*Q_ij[t,0,106])+(self.list_r_x_pu[106][0]*self.list_r_x_pu[106][0]+self.list_r_x_pu[106][1]*self.list_r_x_pu[106][1])*l_ij[t,0,106])==0)

                model.addConstr((v_i[t,0,107]-v_i[t,0,108]-2*(self.list_r_x_pu[107][0]*P_ij[t,0,107]+self.list_r_x_pu[107][1]*Q_ij[t,0,107])+(self.list_r_x_pu[107][0]*self.list_r_x_pu[107][0]+self.list_r_x_pu[107][1]*self.list_r_x_pu[107][1])*l_ij[t,0,107])==0)

                model.addConstr((v_i[t,0,108]-v_i[t,0,109]-2*(self.list_r_x_pu[108][0]*P_ij[t,0,108]+self.list_r_x_pu[108][1]*Q_ij[t,0,108])+(self.list_r_x_pu[108][0]*self.list_r_x_pu[108][0]+self.list_r_x_pu[108][1]*self.list_r_x_pu[108][1])*l_ij[t,0,108])==0)

                model.addConstr((v_i[t,0,109]-v_i[t,0,110]-2*(self.list_r_x_pu[109][0]*P_ij[t,0,109]+self.list_r_x_pu[109][1]*Q_ij[t,0,109])+(self.list_r_x_pu[109][0]*self.list_r_x_pu[109][0]+self.list_r_x_pu[109][1]*self.list_r_x_pu[109][1])*l_ij[t,0,109])==0)

                model.addConstr((v_i[t,0,110]-v_i[t,0,111]-2*(self.list_r_x_pu[110][0]*P_ij[t,0,110]+self.list_r_x_pu[110][1]*Q_ij[t,0,110])+(self.list_r_x_pu[110][0]*self.list_r_x_pu[110][0]+self.list_r_x_pu[110][1]*self.list_r_x_pu[110][1])*l_ij[t,0,110])==0)

                model.addConstr((v_i[t,0,110]-v_i[t,0,112]-2*(self.list_r_x_pu[111][0]*P_ij[t,0,111]+self.list_r_x_pu[111][1]*Q_ij[t,0,111])+(self.list_r_x_pu[111][0]*self.list_r_x_pu[111][0]+self.list_r_x_pu[111][1]*self.list_r_x_pu[111][1])*l_ij[t,0,111])==0)

                model.addConstr((v_i[t,0,112]-v_i[t,0,113]-2*(self.list_r_x_pu[112][0]*P_ij[t,0,112]+self.list_r_x_pu[112][1]*Q_ij[t,0,112])+(self.list_r_x_pu[112][0]*self.list_r_x_pu[112][0]+self.list_r_x_pu[112][1]*self.list_r_x_pu[112][1])*l_ij[t,0,112])==0)

                model.addConstr((v_i[t,0,100]-v_i[t,0,114]-2*(self.list_r_x_pu[113][0]*P_ij[t,0,113]+self.list_r_x_pu[113][1]*Q_ij[t,0,113])+(self.list_r_x_pu[113][0]*self.list_r_x_pu[113][0]+self.list_r_x_pu[113][1]*self.list_r_x_pu[113][1])*l_ij[t,0,113])==0)

                model.addConstr((v_i[t,0,114]-v_i[t,0,115]-2*(self.list_r_x_pu[114][0]*P_ij[t,0,114]+self.list_r_x_pu[114][1]*Q_ij[t,0,114])+(self.list_r_x_pu[114][0]*self.list_r_x_pu[114][0]+self.list_r_x_pu[114][1]*self.list_r_x_pu[114][1])*l_ij[t,0,114])==0)

                model.addConstr((v_i[t,0,115]-v_i[t,0,116]-2*(self.list_r_x_pu[115][0]*P_ij[t,0,115]+self.list_r_x_pu[115][1]*Q_ij[t,0,115])+(self.list_r_x_pu[115][0]*self.list_r_x_pu[115][0]+self.list_r_x_pu[115][1]*self.list_r_x_pu[115][1])*l_ij[t,0,115])==0)

                model.addConstr((v_i[t,0,116]-v_i[t,0,117]-2*(self.list_r_x_pu[116][0]*P_ij[t,0,116]+self.list_r_x_pu[116][1]*Q_ij[t,0,116])+(self.list_r_x_pu[116][0]*self.list_r_x_pu[116][0]+self.list_r_x_pu[116][1]*self.list_r_x_pu[116][1])*l_ij[t,0,116])==0)

                model.addConstr((v_i[t,0,117]-v_i[t,0,118]-2*(self.list_r_x_pu[117][0]*P_ij[t,0,117]+self.list_r_x_pu[117][1]*Q_ij[t,0,117])+(self.list_r_x_pu[117][0]*self.list_r_x_pu[117][0]+self.list_r_x_pu[117][1]*self.list_r_x_pu[117][1])*l_ij[t,0,117])==0)

                model.addConstr(lv_ij_i[t, 0, 0] == l_ij[t, 0, 0] * v_i[t, 0, 0])
                model.addConstr((lv_ij_i[t, 0, 0] - PP_ij[t, 0, 0] - QQ_ij[t, 0, 0]) == 0)

                model.addConstr(lv_ij_i[t, 0, 1] == l_ij[t, 0, 1] * v_i[t, 0, 1])
                model.addConstr((lv_ij_i[t, 0, 1] - PP_ij[t, 0, 1] - QQ_ij[t, 0, 1]) == 0)

                model.addConstr(lv_ij_i[t, 0, 2] == l_ij[t, 0, 2] * v_i[t, 0, 2])
                model.addConstr((lv_ij_i[t, 0, 2] - PP_ij[t, 0, 2] - QQ_ij[t, 0, 2]) == 0)

                model.addConstr(lv_ij_i[t, 0, 3] == l_ij[t, 0, 3] * v_i[t, 0, 2])
                model.addConstr((lv_ij_i[t, 0, 3] - PP_ij[t, 0, 3] - QQ_ij[t, 0, 3]) == 0)

                model.addConstr(lv_ij_i[t, 0, 4] == l_ij[t, 0, 4] * v_i[t, 0, 4])
                model.addConstr((lv_ij_i[t, 0, 4] - PP_ij[t, 0, 4] - QQ_ij[t, 0, 4]) == 0)

                model.addConstr(lv_ij_i[t, 0, 5] == l_ij[t, 0, 5] * v_i[t, 0, 5])
                model.addConstr((lv_ij_i[t, 0, 5] - PP_ij[t, 0, 5] - QQ_ij[t, 0, 5]) == 0)

                model.addConstr(lv_ij_i[t, 0, 6] == l_ij[t, 0, 6] * v_i[t, 0, 6])
                model.addConstr((lv_ij_i[t, 0, 6] - PP_ij[t, 0, 6] - QQ_ij[t, 0, 6]) == 0)

                model.addConstr(lv_ij_i[t, 0, 7] == l_ij[t, 0, 7] * v_i[t, 0, 7])
                model.addConstr((lv_ij_i[t, 0, 7] - PP_ij[t, 0, 7] - QQ_ij[t, 0, 7]) == 0)

                model.addConstr(lv_ij_i[t, 0, 8] == l_ij[t, 0, 8] * v_i[t, 0, 8])
                model.addConstr((lv_ij_i[t, 0, 8] - PP_ij[t, 0, 8] - QQ_ij[t, 0, 8]) == 0)

                model.addConstr(lv_ij_i[t, 0, 9] == l_ij[t, 0, 9] * v_i[t, 0, 2])
                model.addConstr((lv_ij_i[t, 0, 9] - PP_ij[t, 0, 9] - QQ_ij[t, 0, 9]) == 0)

                model.addConstr(lv_ij_i[t, 0, 10] == l_ij[t, 0, 10] * v_i[t, 0, 10])
                model.addConstr((lv_ij_i[t, 0, 10] - PP_ij[t, 0, 10] - QQ_ij[t, 0, 10]) == 0)

                model.addConstr(lv_ij_i[t, 0, 11] == l_ij[t, 0, 11] * v_i[t, 0, 11])
                model.addConstr((lv_ij_i[t, 0, 11] - PP_ij[t, 0, 11] - QQ_ij[t, 0, 11]) == 0)

                model.addConstr(lv_ij_i[t, 0, 12] == l_ij[t, 0, 12] * v_i[t, 0, 12])
                model.addConstr((lv_ij_i[t, 0, 12] - PP_ij[t, 0, 12] - QQ_ij[t, 0, 12]) == 0)

                model.addConstr(lv_ij_i[t, 0, 13] == l_ij[t, 0, 13] * v_i[t, 0, 13])
                model.addConstr((lv_ij_i[t, 0, 13] - PP_ij[t, 0, 13] - QQ_ij[t, 0, 13]) == 0)

                model.addConstr(lv_ij_i[t, 0, 14] == l_ij[t, 0, 14] * v_i[t, 0, 14])
                model.addConstr((lv_ij_i[t, 0, 14] - PP_ij[t, 0, 14] - QQ_ij[t, 0, 14]) == 0)

                model.addConstr(lv_ij_i[t, 0, 15] == l_ij[t, 0, 15] * v_i[t, 0, 15])
                model.addConstr((lv_ij_i[t, 0, 15] - PP_ij[t, 0, 15] - QQ_ij[t, 0, 15]) == 0)

                model.addConstr(lv_ij_i[t, 0, 16] == l_ij[t, 0, 16] * v_i[t, 0, 16])
                model.addConstr((lv_ij_i[t, 0, 16] - PP_ij[t, 0, 16] - QQ_ij[t, 0, 16]) == 0)

                model.addConstr(lv_ij_i[t, 0, 17] == l_ij[t, 0, 17] * v_i[t, 0, 11])
                model.addConstr((lv_ij_i[t, 0, 17] - PP_ij[t, 0, 17] - QQ_ij[t, 0, 17]) == 0)

                model.addConstr(lv_ij_i[t, 0, 18] == l_ij[t, 0, 18] * v_i[t, 0, 18])
                model.addConstr((lv_ij_i[t, 0, 18] - PP_ij[t, 0, 18] - QQ_ij[t, 0, 18]) == 0)

                model.addConstr(lv_ij_i[t, 0, 19] == l_ij[t, 0, 19] * v_i[t, 0, 19])
                model.addConstr((lv_ij_i[t, 0, 19] - PP_ij[t, 0, 19] - QQ_ij[t, 0, 19]) == 0)

                model.addConstr(lv_ij_i[t, 0, 20] == l_ij[t, 0, 20] * v_i[t, 0, 20])
                model.addConstr((lv_ij_i[t, 0, 20] - PP_ij[t, 0, 20] - QQ_ij[t, 0, 20]) == 0)

                model.addConstr(lv_ij_i[t, 0, 21] == l_ij[t, 0, 21] * v_i[t, 0, 21])
                model.addConstr((lv_ij_i[t, 0, 21] - PP_ij[t, 0, 21] - QQ_ij[t, 0, 21]) == 0)

                model.addConstr(lv_ij_i[t, 0, 22] == l_ij[t, 0, 22] * v_i[t, 0, 22])
                model.addConstr((lv_ij_i[t, 0, 22] - PP_ij[t, 0, 22] - QQ_ij[t, 0, 22]) == 0)

                model.addConstr(lv_ij_i[t, 0, 23] == l_ij[t, 0, 23] * v_i[t, 0, 23])
                model.addConstr((lv_ij_i[t, 0, 23] - PP_ij[t, 0, 23] - QQ_ij[t, 0, 23]) == 0)

                model.addConstr(lv_ij_i[t, 0, 24] == l_ij[t, 0, 24] * v_i[t, 0, 24])
                model.addConstr((lv_ij_i[t, 0, 24] - PP_ij[t, 0, 24] - QQ_ij[t, 0, 24]) == 0)

                model.addConstr(lv_ij_i[t, 0, 25] == l_ij[t, 0, 25] * v_i[t, 0, 25])
                model.addConstr((lv_ij_i[t, 0, 25] - PP_ij[t, 0, 25] - QQ_ij[t, 0, 25]) == 0)

                model.addConstr(lv_ij_i[t, 0, 26] == l_ij[t, 0, 26] * v_i[t, 0, 26])
                model.addConstr((lv_ij_i[t, 0, 26] - PP_ij[t, 0, 26] - QQ_ij[t, 0, 26]) == 0)

                model.addConstr(lv_ij_i[t, 0, 27] == l_ij[t, 0, 27] * v_i[t, 0, 4])
                model.addConstr((lv_ij_i[t, 0, 27] - PP_ij[t, 0, 27] - QQ_ij[t, 0, 27]) == 0)

                model.addConstr(lv_ij_i[t, 0, 28] == l_ij[t, 0, 28] * v_i[t, 0, 28])
                model.addConstr((lv_ij_i[t, 0, 28] - PP_ij[t, 0, 28] - QQ_ij[t, 0, 28]) == 0)

                model.addConstr(lv_ij_i[t, 0, 29] == l_ij[t, 0, 29] * v_i[t, 0, 29])
                model.addConstr((lv_ij_i[t, 0, 29] - PP_ij[t, 0, 29] - QQ_ij[t, 0, 29]) == 0)

                model.addConstr(lv_ij_i[t, 0, 30] == l_ij[t, 0, 30] * v_i[t, 0, 30])
                model.addConstr((lv_ij_i[t, 0, 30] - PP_ij[t, 0, 30] - QQ_ij[t, 0, 30]) == 0)

                model.addConstr(lv_ij_i[t, 0, 31] == l_ij[t, 0, 31] * v_i[t, 0, 31])
                model.addConstr((lv_ij_i[t, 0, 31] - PP_ij[t, 0, 31] - QQ_ij[t, 0, 31]) == 0)

                model.addConstr(lv_ij_i[t, 0, 32] == l_ij[t, 0, 32] * v_i[t, 0, 32])
                model.addConstr((lv_ij_i[t, 0, 32] - PP_ij[t, 0, 32] - QQ_ij[t, 0, 32]) == 0)

                model.addConstr(lv_ij_i[t, 0, 33] == l_ij[t, 0, 33] * v_i[t, 0, 33])
                model.addConstr((lv_ij_i[t, 0, 33] - PP_ij[t, 0, 33] - QQ_ij[t, 0, 33]) == 0)

                model.addConstr(lv_ij_i[t, 0, 34] == l_ij[t, 0, 34] * v_i[t, 0, 34])
                model.addConstr((lv_ij_i[t, 0, 34] - PP_ij[t, 0, 34] - QQ_ij[t, 0, 34]) == 0)

                model.addConstr(lv_ij_i[t, 0, 35] == l_ij[t, 0, 35] * v_i[t, 0, 30])
                model.addConstr((lv_ij_i[t, 0, 35] - PP_ij[t, 0, 35] - QQ_ij[t, 0, 35]) == 0)

                model.addConstr(lv_ij_i[t, 0, 36] == l_ij[t, 0, 36] * v_i[t, 0, 36])
                model.addConstr((lv_ij_i[t, 0, 36] - PP_ij[t, 0, 36] - QQ_ij[t, 0, 36]) == 0)

                model.addConstr(lv_ij_i[t, 0, 37] == l_ij[t, 0, 37] * v_i[t, 0, 29])
                model.addConstr((lv_ij_i[t, 0, 37] - PP_ij[t, 0, 37] - QQ_ij[t, 0, 37]) == 0)

                model.addConstr(lv_ij_i[t, 0, 38] == l_ij[t, 0, 38] * v_i[t, 0, 38])
                model.addConstr((lv_ij_i[t, 0, 38] - PP_ij[t, 0, 38] - QQ_ij[t, 0, 38]) == 0)

                model.addConstr(lv_ij_i[t, 0, 39] == l_ij[t, 0, 39] * v_i[t, 0, 39])
                model.addConstr((lv_ij_i[t, 0, 39] - PP_ij[t, 0, 39] - QQ_ij[t, 0, 39]) == 0)

                model.addConstr(lv_ij_i[t, 0, 40] == l_ij[t, 0, 40] * v_i[t, 0, 40])
                model.addConstr((lv_ij_i[t, 0, 40] - PP_ij[t, 0, 40] - QQ_ij[t, 0, 40]) == 0)

                model.addConstr(lv_ij_i[t, 0, 41] == l_ij[t, 0, 41] * v_i[t, 0, 41])
                model.addConstr((lv_ij_i[t, 0, 41] - PP_ij[t, 0, 41] - QQ_ij[t, 0, 41]) == 0)

                model.addConstr(lv_ij_i[t, 0, 42] == l_ij[t, 0, 42] * v_i[t, 0, 42])
                model.addConstr((lv_ij_i[t, 0, 42] - PP_ij[t, 0, 42] - QQ_ij[t, 0, 42]) == 0)

                model.addConstr(lv_ij_i[t, 0, 43] == l_ij[t, 0, 43] * v_i[t, 0, 43])
                model.addConstr((lv_ij_i[t, 0, 43] - PP_ij[t, 0, 43] - QQ_ij[t, 0, 43]) == 0)

                model.addConstr(lv_ij_i[t, 0, 44] == l_ij[t, 0, 44] * v_i[t, 0, 44])
                model.addConstr((lv_ij_i[t, 0, 44] - PP_ij[t, 0, 44] - QQ_ij[t, 0, 44]) == 0)

                model.addConstr(lv_ij_i[t, 0, 45] == l_ij[t, 0, 45] * v_i[t, 0, 45])
                model.addConstr((lv_ij_i[t, 0, 45] - PP_ij[t, 0, 45] - QQ_ij[t, 0, 45]) == 0)

                model.addConstr(lv_ij_i[t, 0, 46] == l_ij[t, 0, 46] * v_i[t, 0, 35])
                model.addConstr((lv_ij_i[t, 0, 46] - PP_ij[t, 0, 46] - QQ_ij[t, 0, 46]) == 0)

                model.addConstr(lv_ij_i[t, 0, 47] == l_ij[t, 0, 47] * v_i[t, 0, 47])
                model.addConstr((lv_ij_i[t, 0, 47] - PP_ij[t, 0, 47] - QQ_ij[t, 0, 47]) == 0)

                model.addConstr(lv_ij_i[t, 0, 48] == l_ij[t, 0, 48] * v_i[t, 0, 48])
                model.addConstr((lv_ij_i[t, 0, 48] - PP_ij[t, 0, 48] - QQ_ij[t, 0, 48]) == 0)

                model.addConstr(lv_ij_i[t, 0, 49] == l_ij[t, 0, 49] * v_i[t, 0, 49])
                model.addConstr((lv_ij_i[t, 0, 49] - PP_ij[t, 0, 49] - QQ_ij[t, 0, 49]) == 0)

                model.addConstr(lv_ij_i[t, 0, 50] == l_ij[t, 0, 50] * v_i[t, 0, 50])
                model.addConstr((lv_ij_i[t, 0, 50] - PP_ij[t, 0, 50] - QQ_ij[t, 0, 50]) == 0)

                model.addConstr(lv_ij_i[t, 0, 51] == l_ij[t, 0, 51] * v_i[t, 0, 51])
                model.addConstr((lv_ij_i[t, 0, 51] - PP_ij[t, 0, 51] - QQ_ij[t, 0, 51]) == 0)

                model.addConstr(lv_ij_i[t, 0, 52] == l_ij[t, 0, 52] * v_i[t, 0, 52])
                model.addConstr((lv_ij_i[t, 0, 52] - PP_ij[t, 0, 52] - QQ_ij[t, 0, 52]) == 0)

                model.addConstr(lv_ij_i[t, 0, 53] == l_ij[t, 0, 53] * v_i[t, 0, 53])
                model.addConstr((lv_ij_i[t, 0, 53] - PP_ij[t, 0, 53] - QQ_ij[t, 0, 53]) == 0)

                model.addConstr(lv_ij_i[t, 0, 54] == l_ij[t, 0, 54] * v_i[t, 0, 29])
                model.addConstr((lv_ij_i[t, 0, 54] - PP_ij[t, 0, 54] - QQ_ij[t, 0, 54]) == 0)

                model.addConstr(lv_ij_i[t, 0, 55] == l_ij[t, 0, 55] * v_i[t, 0, 55])
                model.addConstr((lv_ij_i[t, 0, 55] - PP_ij[t, 0, 55] - QQ_ij[t, 0, 55]) == 0)

                model.addConstr(lv_ij_i[t, 0, 56] == l_ij[t, 0, 56] * v_i[t, 0, 56])
                model.addConstr((lv_ij_i[t, 0, 56] - PP_ij[t, 0, 56] - QQ_ij[t, 0, 56]) == 0)

                model.addConstr(lv_ij_i[t, 0, 57] == l_ij[t, 0, 57] * v_i[t, 0, 57])
                model.addConstr((lv_ij_i[t, 0, 57] - PP_ij[t, 0, 57] - QQ_ij[t, 0, 57]) == 0)

                model.addConstr(lv_ij_i[t, 0, 58] == l_ij[t, 0, 58] * v_i[t, 0, 58])
                model.addConstr((lv_ij_i[t, 0, 58] - PP_ij[t, 0, 58] - QQ_ij[t, 0, 58]) == 0)

                model.addConstr(lv_ij_i[t, 0, 59] == l_ij[t, 0, 59] * v_i[t, 0, 59])
                model.addConstr((lv_ij_i[t, 0, 59] - PP_ij[t, 0, 59] - QQ_ij[t, 0, 59]) == 0)

                model.addConstr(lv_ij_i[t, 0, 60] == l_ij[t, 0, 60] * v_i[t, 0, 60])
                model.addConstr((lv_ij_i[t, 0, 60] - PP_ij[t, 0, 60] - QQ_ij[t, 0, 60]) == 0)

                model.addConstr(lv_ij_i[t, 0, 61] == l_ij[t, 0, 61] * v_i[t, 0, 61])
                model.addConstr((lv_ij_i[t, 0, 61] - PP_ij[t, 0, 61] - QQ_ij[t, 0, 61]) == 0)

                model.addConstr(lv_ij_i[t, 0, 62] == l_ij[t, 0, 62] * v_i[t, 0, 1])
                model.addConstr((lv_ij_i[t, 0, 62] - PP_ij[t, 0, 62] - QQ_ij[t, 0, 62]) == 0)

                model.addConstr(lv_ij_i[t, 0, 63] == l_ij[t, 0, 63] * v_i[t, 0, 63])
                model.addConstr((lv_ij_i[t, 0, 63] - PP_ij[t, 0, 63] - QQ_ij[t, 0, 63]) == 0)

                model.addConstr(lv_ij_i[t, 0, 64] == l_ij[t, 0, 64] * v_i[t, 0, 64])
                model.addConstr((lv_ij_i[t, 0, 64] - PP_ij[t, 0, 64] - QQ_ij[t, 0, 64]) == 0)

                model.addConstr(lv_ij_i[t, 0, 65] == l_ij[t, 0, 65] * v_i[t, 0, 65])
                model.addConstr((lv_ij_i[t, 0, 65] - PP_ij[t, 0, 65] - QQ_ij[t, 0, 65]) == 0)

                model.addConstr(lv_ij_i[t, 0, 66] == l_ij[t, 0, 66] * v_i[t, 0, 66])
                model.addConstr((lv_ij_i[t, 0, 66] - PP_ij[t, 0, 66] - QQ_ij[t, 0, 66]) == 0)

                model.addConstr(lv_ij_i[t, 0, 67] == l_ij[t, 0, 67] * v_i[t, 0, 67])
                model.addConstr((lv_ij_i[t, 0, 67] - PP_ij[t, 0, 67] - QQ_ij[t, 0, 67]) == 0)

                model.addConstr(lv_ij_i[t, 0, 68] == l_ij[t, 0, 68] * v_i[t, 0, 68])
                model.addConstr((lv_ij_i[t, 0, 68] - PP_ij[t, 0, 68] - QQ_ij[t, 0, 68]) == 0)

                model.addConstr(lv_ij_i[t, 0, 69] == l_ij[t, 0, 69] * v_i[t, 0, 69])
                model.addConstr((lv_ij_i[t, 0, 69] - PP_ij[t, 0, 69] - QQ_ij[t, 0, 69]) == 0)

                model.addConstr(lv_ij_i[t, 0, 70] == l_ij[t, 0, 70] * v_i[t, 0, 70])
                model.addConstr((lv_ij_i[t, 0, 70] - PP_ij[t, 0, 70] - QQ_ij[t, 0, 70]) == 0)

                model.addConstr(lv_ij_i[t, 0, 71] == l_ij[t, 0, 71] * v_i[t, 0, 71])
                model.addConstr((lv_ij_i[t, 0, 71] - PP_ij[t, 0, 71] - QQ_ij[t, 0, 71]) == 0)

                model.addConstr(lv_ij_i[t, 0, 72] == l_ij[t, 0, 72] * v_i[t, 0, 72])
                model.addConstr((lv_ij_i[t, 0, 72] - PP_ij[t, 0, 72] - QQ_ij[t, 0, 72]) == 0)

                model.addConstr(lv_ij_i[t, 0, 73] == l_ij[t, 0, 73] * v_i[t, 0, 73])
                model.addConstr((lv_ij_i[t, 0, 73] - PP_ij[t, 0, 73] - QQ_ij[t, 0, 73]) == 0)

                model.addConstr(lv_ij_i[t, 0, 74] == l_ij[t, 0, 74] * v_i[t, 0, 74])
                model.addConstr((lv_ij_i[t, 0, 74] - PP_ij[t, 0, 74] - QQ_ij[t, 0, 74]) == 0)

                model.addConstr(lv_ij_i[t, 0, 75] == l_ij[t, 0, 75] * v_i[t, 0, 75])
                model.addConstr((lv_ij_i[t, 0, 75] - PP_ij[t, 0, 75] - QQ_ij[t, 0, 75]) == 0)

                model.addConstr(lv_ij_i[t, 0, 76] == l_ij[t, 0, 76] * v_i[t, 0, 76])
                model.addConstr((lv_ij_i[t, 0, 76] - PP_ij[t, 0, 76] - QQ_ij[t, 0, 76]) == 0)

                model.addConstr(lv_ij_i[t, 0, 77] == l_ij[t, 0, 77] * v_i[t, 0, 64])
                model.addConstr((lv_ij_i[t, 0, 77] - PP_ij[t, 0, 77] - QQ_ij[t, 0, 77]) == 0)

                model.addConstr(lv_ij_i[t, 0, 78] == l_ij[t, 0, 78] * v_i[t, 0, 78])
                model.addConstr((lv_ij_i[t, 0, 78] - PP_ij[t, 0, 78] - QQ_ij[t, 0, 78]) == 0)

                model.addConstr(lv_ij_i[t, 0, 79] == l_ij[t, 0, 79] * v_i[t, 0, 79])
                model.addConstr((lv_ij_i[t, 0, 79] - PP_ij[t, 0, 79] - QQ_ij[t, 0, 79]) == 0)

                model.addConstr(lv_ij_i[t, 0, 80] == l_ij[t, 0, 80] * v_i[t, 0, 80])
                model.addConstr((lv_ij_i[t, 0, 80] - PP_ij[t, 0, 80] - QQ_ij[t, 0, 80]) == 0)

                model.addConstr(lv_ij_i[t, 0, 81] == l_ij[t, 0, 81] * v_i[t, 0, 81])
                model.addConstr((lv_ij_i[t, 0, 81] - PP_ij[t, 0, 81] - QQ_ij[t, 0, 81]) == 0)

                model.addConstr(lv_ij_i[t, 0, 82] == l_ij[t, 0, 82] * v_i[t, 0, 82])
                model.addConstr((lv_ij_i[t, 0, 82] - PP_ij[t, 0, 82] - QQ_ij[t, 0, 82]) == 0)

                model.addConstr(lv_ij_i[t, 0, 83] == l_ij[t, 0, 83] * v_i[t, 0, 83])
                model.addConstr((lv_ij_i[t, 0, 83] - PP_ij[t, 0, 83] - QQ_ij[t, 0, 83]) == 0)

                model.addConstr(lv_ij_i[t, 0, 84] == l_ij[t, 0, 84] * v_i[t, 0, 84])
                model.addConstr((lv_ij_i[t, 0, 84] - PP_ij[t, 0, 84] - QQ_ij[t, 0, 84]) == 0)

                model.addConstr(lv_ij_i[t, 0, 85] == l_ij[t, 0, 85] * v_i[t, 0, 79])
                model.addConstr((lv_ij_i[t, 0, 85] - PP_ij[t, 0, 85] - QQ_ij[t, 0, 85]) == 0)

                model.addConstr(lv_ij_i[t, 0, 86] == l_ij[t, 0, 86] * v_i[t, 0, 86])
                model.addConstr((lv_ij_i[t, 0, 86] - PP_ij[t, 0, 86] - QQ_ij[t, 0, 86]) == 0)

                model.addConstr(lv_ij_i[t, 0, 87] == l_ij[t, 0, 87] * v_i[t, 0, 87])
                model.addConstr((lv_ij_i[t, 0, 87] - PP_ij[t, 0, 87] - QQ_ij[t, 0, 87]) == 0)

                model.addConstr(lv_ij_i[t, 0, 88] == l_ij[t, 0, 88] * v_i[t, 0, 65])
                model.addConstr((lv_ij_i[t, 0, 88] - PP_ij[t, 0, 88] - QQ_ij[t, 0, 88]) == 0)

                model.addConstr(lv_ij_i[t, 0, 89] == l_ij[t, 0, 89] * v_i[t, 0, 89])
                model.addConstr((lv_ij_i[t, 0, 89] - PP_ij[t, 0, 89] - QQ_ij[t, 0, 89]) == 0)

                model.addConstr(lv_ij_i[t, 0, 90] == l_ij[t, 0, 90] * v_i[t, 0, 90])
                model.addConstr((lv_ij_i[t, 0, 90] - PP_ij[t, 0, 90] - QQ_ij[t, 0, 90]) == 0)

                model.addConstr(lv_ij_i[t, 0, 91] == l_ij[t, 0, 91] * v_i[t, 0, 91])
                model.addConstr((lv_ij_i[t, 0, 91] - PP_ij[t, 0, 91] - QQ_ij[t, 0, 91]) == 0)

                model.addConstr(lv_ij_i[t, 0, 92] == l_ij[t, 0, 92] * v_i[t, 0, 92])
                model.addConstr((lv_ij_i[t, 0, 92] - PP_ij[t, 0, 92] - QQ_ij[t, 0, 92]) == 0)

                model.addConstr(lv_ij_i[t, 0, 93] == l_ij[t, 0, 93] * v_i[t, 0, 93])
                model.addConstr((lv_ij_i[t, 0, 93] - PP_ij[t, 0, 93] - QQ_ij[t, 0, 93]) == 0)

                model.addConstr(lv_ij_i[t, 0, 94] == l_ij[t, 0, 94] * v_i[t, 0, 94])
                model.addConstr((lv_ij_i[t, 0, 94] - PP_ij[t, 0, 94] - QQ_ij[t, 0, 94]) == 0)

                model.addConstr(lv_ij_i[t, 0, 95] == l_ij[t, 0, 95] * v_i[t, 0, 91])
                model.addConstr((lv_ij_i[t, 0, 95] - PP_ij[t, 0, 95] - QQ_ij[t, 0, 95]) == 0)

                model.addConstr(lv_ij_i[t, 0, 96] == l_ij[t, 0, 96] * v_i[t, 0, 96])
                model.addConstr((lv_ij_i[t, 0, 96] - PP_ij[t, 0, 96] - QQ_ij[t, 0, 96]) == 0)

                model.addConstr(lv_ij_i[t, 0, 97] == l_ij[t, 0, 97] * v_i[t, 0, 97])
                model.addConstr((lv_ij_i[t, 0, 97] - PP_ij[t, 0, 97] - QQ_ij[t, 0, 97]) == 0)

                model.addConstr(lv_ij_i[t, 0, 98] == l_ij[t, 0, 98] * v_i[t, 0, 98])
                model.addConstr((lv_ij_i[t, 0, 98] - PP_ij[t, 0, 98] - QQ_ij[t, 0, 98]) == 0)

                model.addConstr(lv_ij_i[t, 0, 99] == l_ij[t, 0, 99] * v_i[t, 0, 1])
                model.addConstr((lv_ij_i[t, 0, 99] - PP_ij[t, 0, 99] - QQ_ij[t, 0, 99]) == 0)

                model.addConstr(lv_ij_i[t, 0, 100] == l_ij[t, 0, 100] * v_i[t, 0, 100])
                model.addConstr((lv_ij_i[t, 0, 100] - PP_ij[t, 0, 100] - QQ_ij[t, 0, 100]) == 0)

                model.addConstr(lv_ij_i[t, 0, 101] == l_ij[t, 0, 101] * v_i[t, 0, 101])
                model.addConstr((lv_ij_i[t, 0, 101] - PP_ij[t, 0, 101] - QQ_ij[t, 0, 101]) == 0)

                model.addConstr(lv_ij_i[t, 0, 102] == l_ij[t, 0, 102] * v_i[t, 0, 102])
                model.addConstr((lv_ij_i[t, 0, 102] - PP_ij[t, 0, 102] - QQ_ij[t, 0, 102]) == 0)

                model.addConstr(lv_ij_i[t, 0, 103] == l_ij[t, 0, 103] * v_i[t, 0, 103])
                model.addConstr((lv_ij_i[t, 0, 103] - PP_ij[t, 0, 103] - QQ_ij[t, 0, 103]) == 0)

                model.addConstr(lv_ij_i[t, 0, 104] == l_ij[t, 0, 104] * v_i[t, 0, 104])
                model.addConstr((lv_ij_i[t, 0, 104] - PP_ij[t, 0, 104] - QQ_ij[t, 0, 104]) == 0)

                model.addConstr(lv_ij_i[t, 0, 105] == l_ij[t, 0, 105] * v_i[t, 0, 105])
                model.addConstr((lv_ij_i[t, 0, 105] - PP_ij[t, 0, 105] - QQ_ij[t, 0, 105]) == 0)

                model.addConstr(lv_ij_i[t, 0, 106] == l_ij[t, 0, 106] * v_i[t, 0, 106])
                model.addConstr((lv_ij_i[t, 0, 106] - PP_ij[t, 0, 106] - QQ_ij[t, 0, 106]) == 0)

                model.addConstr(lv_ij_i[t, 0, 107] == l_ij[t, 0, 107] * v_i[t, 0, 107])
                model.addConstr((lv_ij_i[t, 0, 107] - PP_ij[t, 0, 107] - QQ_ij[t, 0, 107]) == 0)

                model.addConstr(lv_ij_i[t, 0, 108] == l_ij[t, 0, 108] * v_i[t, 0, 108])
                model.addConstr((lv_ij_i[t, 0, 108] - PP_ij[t, 0, 108] - QQ_ij[t, 0, 108]) == 0)

                model.addConstr(lv_ij_i[t, 0, 109] == l_ij[t, 0, 109] * v_i[t, 0, 109])
                model.addConstr((lv_ij_i[t, 0, 109] - PP_ij[t, 0, 109] - QQ_ij[t, 0, 109]) == 0)

                model.addConstr(lv_ij_i[t, 0, 110] == l_ij[t, 0, 110] * v_i[t, 0, 110])
                model.addConstr((lv_ij_i[t, 0, 110] - PP_ij[t, 0, 110] - QQ_ij[t, 0, 110]) == 0)

                model.addConstr(lv_ij_i[t, 0, 111] == l_ij[t, 0, 111] * v_i[t, 0, 110])
                model.addConstr((lv_ij_i[t, 0, 111] - PP_ij[t, 0, 111] - QQ_ij[t, 0, 111]) == 0)

                model.addConstr(lv_ij_i[t, 0, 112] == l_ij[t, 0, 112] * v_i[t, 0, 112])
                model.addConstr((lv_ij_i[t, 0, 112] - PP_ij[t, 0, 112] - QQ_ij[t, 0, 112]) == 0)

                model.addConstr(lv_ij_i[t, 0, 113] == l_ij[t, 0, 113] * v_i[t, 0, 100])
                model.addConstr((lv_ij_i[t, 0, 113] - PP_ij[t, 0, 113] - QQ_ij[t, 0, 113]) == 0)

                model.addConstr(lv_ij_i[t, 0, 114] == l_ij[t, 0, 114] * v_i[t, 0, 114])
                model.addConstr((lv_ij_i[t, 0, 114] - PP_ij[t, 0, 114] - QQ_ij[t, 0, 114]) == 0)

                model.addConstr(lv_ij_i[t, 0, 115] == l_ij[t, 0, 115] * v_i[t, 0, 115])
                model.addConstr((lv_ij_i[t, 0, 115] - PP_ij[t, 0, 115] - QQ_ij[t, 0, 115]) == 0)

                model.addConstr(lv_ij_i[t, 0, 116] == l_ij[t, 0, 116] * v_i[t, 0, 116])
                model.addConstr((lv_ij_i[t, 0, 116] - PP_ij[t, 0, 116] - QQ_ij[t, 0, 116]) == 0)

                model.addConstr(lv_ij_i[t, 0, 117] == l_ij[t, 0, 117] * v_i[t, 0, 117])
                model.addConstr((lv_ij_i[t, 0, 117] - PP_ij[t, 0, 117] - QQ_ij[t, 0, 117]) == 0)

            model.setParam('outPutFlag', 0)
            model.setParam(GRB.Param.TimeLimit, 300)
            model.Params.MIPGap = 0.001
            model.optimize()

            print("", model.Runtime)

            obj=model.objval
            dict_value=dict()
            for v in model.getVars():
                dict_value[v.varName]=v.x
        except GurobiError as e:
            print('Error code ' + str(e.errno) + ':' + str(e))
            dict_value=dict()
            obj = 'error'
            dict_value['P_DG_29[0]'] = 'error'
            dict_value['P_DG_64[0]'] = 'error'
            dict_value['SOC_BSS_12[0]'] = 'error'
            dict_value['SOC_BSS_34[0]'] = 'error'
            dict_value['SOC_BSS_68[0]'] = 'error'
            dict_value['SOC_BSS_103[0]'] = 'error'
            dict_value['cost_hour[0]'] = 'error'
        except AttributeError:
            print('Encountered an attribute error')
            dict_value=dict()
            obj = 'error'
            dict_value['P_DG_29[0]'] = 'error'
            dict_value['P_DG_64[0]'] = 'error'
            dict_value['SOC_BSS_12[0]'] = 'error'
            dict_value['SOC_BSS_34[0]'] = 'error'
            dict_value['SOC_BSS_68[0]'] = 'error'
            dict_value['SOC_BSS_103[0]'] = 'error'
            dict_value['cost_hour[0]'] = 'error'

        return (obj, dict_value['P_DG_29[0]'], dict_value['P_DG_64[0]'],
                dict_value['SOC_BSS_12[0]'], dict_value['SOC_BSS_34[0]'], dict_value['SOC_BSS_68[0]'], dict_value['SOC_BSS_103[0]'],
                dict_value['cost_hour[0]'], model.Runtime)

    def sol_pro_remainder(self, current_time, window_time,
                        load_list, load_list_Q,
                        P_DG_29_init, P_DG_64_init,
                        SOC_BSS_12_init, SOC_BSS_34_init, SOC_BSS_68_init, SOC_BSS_103_init,
                        wind_11_avail_list, wind_32_avail_list, wind_66_avail_list, wind_101_avail_list,
                        pv_22_avail_list, pv_36_avail_list, pv_70_avail_list, pv_107_avail_list):
        self.load_list = load_list
        self.load_list_Q = load_list_Q
        self.wind_11_avail_list = wind_11_avail_list
        self.wind_32_avail_list = wind_32_avail_list
        self.wind_66_avail_list = wind_66_avail_list
        self.wind_101_avail_list = wind_101_avail_list
        self.pv_22_avail_list = pv_22_avail_list
        self.pv_36_avail_list = pv_36_avail_list
        self.pv_70_avail_list = pv_70_avail_list
        self.pv_107_avail_list = pv_107_avail_list
        self.list_p_q_pu_t = self._get_pq_t()

        obj = "error"
        try:
            model = Model('MPC')

            model.setParam('MIPFocus',0)

            P_ij = model.addMVar(shape=(window_time, 1, 118), lb=-100, ub=100, vtype=GRB.CONTINUOUS, name='P_ij')
            Q_ij = model.addMVar(shape=(window_time, 1, 118), lb=-100, ub=100, vtype=GRB.CONTINUOUS, name='Q_ij')
            l_ij = model.addMVar(shape=(window_time, 1, 118), lb=0, vtype=GRB.CONTINUOUS, name='l_ij')
            v_i = model.addMVar(shape=(window_time, 1, 119), lb=0.9 * 0.9, ub=1.1 * 1.1, vtype=GRB.CONTINUOUS, name='v_i')
            lv_ij_i = model.addMVar(shape=(window_time, 2, 118), vtype=GRB.CONTINUOUS, name='lv_ij_i')
            PP_ij = model.addMVar(shape=(window_time, 1, 118), vtype=GRB.CONTINUOUS, name='PP_ij')
            QQ_ij = model.addMVar(shape=(window_time, 1, 118), vtype=GRB.CONTINUOUS, name='QQ_ij')

            Loss = model.addMVar(shape=(window_time,), lb=0, vtype=GRB.CONTINUOUS, name='Loss')
            Loss_i = model.addMVar(shape=(window_time, 1, 118), lb=0, vtype=GRB.CONTINUOUS, name='Loss_i')

            Q_svc_20 = model.addMVar(shape=(window_time,), lb=-0.3, ub=0.3, vtype=GRB.CONTINUOUS, name='Q_svc_20')
            Q_svc_56 = model.addMVar(shape=(window_time,), lb=-0.3, ub=0.3, vtype=GRB.CONTINUOUS, name='Q_svc_56')
            Q_svc_79 = model.addMVar(shape=(window_time,), lb=-0.3, ub=0.3, vtype=GRB.CONTINUOUS, name='Q_svc_79')
            Q_svc_105 = model.addMVar(shape=(window_time,), lb=-0.3, ub=0.3, vtype=GRB.CONTINUOUS, name='Q_svc_105')

            Q_wt_11 = model.addMVar(shape=(window_time,), vtype=GRB.CONTINUOUS, name='Q_wt_11')
            Q_wt_32 = model.addMVar(shape=(window_time,), vtype=GRB.CONTINUOUS, name='Q_wt_32')
            Q_wt_66 = model.addMVar(shape=(window_time,), vtype=GRB.CONTINUOUS, name='Q_wt_66')
            Q_wt_101 = model.addMVar(shape=(window_time,), vtype=GRB.CONTINUOUS, name='Q_wt_101')

            Q_pv_22 = model.addMVar(shape=(window_time,), vtype=GRB.CONTINUOUS, name='Q_pv_22')
            Q_pv_36 = model.addMVar(shape=(window_time,), vtype=GRB.CONTINUOUS, name='Q_pv_36')
            Q_pv_70 = model.addMVar(shape=(window_time,), vtype=GRB.CONTINUOUS, name='Q_pv_70')
            Q_pv_107 = model.addMVar(shape=(window_time,), vtype=GRB.CONTINUOUS, name='Q_pv_107')

            P_DG_29 = model.addMVar(shape=(window_time,), lb=self.gen_min, ub=self.gen_max, vtype=GRB.CONTINUOUS, name='P_DG_29')
            P_DG_64 = model.addMVar(shape=(window_time,), lb=self.gen_min, ub=self.gen_max, vtype=GRB.CONTINUOUS, name='P_DG_64')

            SOC_BSS_12 = model.addMVar(shape=(window_time,), lb=self.SOC_min, ub=self.SOC_max, vtype=GRB.CONTINUOUS, name='SOC_BSS_12')
            SOC_BSS_34 = model.addMVar(shape=(window_time,), lb=self.SOC_min, ub=self.SOC_max, vtype=GRB.CONTINUOUS, name='SOC_BSS_34')
            SOC_BSS_68 = model.addMVar(shape=(window_time,), lb=self.SOC_min, ub=self.SOC_max, vtype=GRB.CONTINUOUS, name='SOC_BSS_68')
            SOC_BSS_103 = model.addMVar(shape=(window_time,), lb=self.SOC_min, ub=self.SOC_max, vtype=GRB.CONTINUOUS, name='SOC_BSS_103')

            P_BSS_ch_12 = model.addMVar(shape=(window_time,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS, name='P_BSS_ch_12')
            P_BSS_ch_34 = model.addMVar(shape=(window_time,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS, name='P_BSS_ch_34')
            P_BSS_ch_68 = model.addMVar(shape=(window_time,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS, name='P_BSS_ch_68')
            P_BSS_ch_103 = model.addMVar(shape=(window_time,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS, name='P_BSS_ch_103')
            P_BSS_dch_12 = model.addMVar(shape=(window_time,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS, name='P_BSS_dch_12')
            P_BSS_dch_34 = model.addMVar(shape=(window_time,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS, name='P_BSS_dch_34')
            P_BSS_dch_68 = model.addMVar(shape=(window_time,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS, name='P_BSS_dch_68')
            P_BSS_dch_103 = model.addMVar(shape=(window_time,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS, name='P_BSS_dch_103')

            cost_balance = model.addMVar(shape=(window_time,), vtype=GRB.CONTINUOUS, name='cost_balance')
            P_balance = model.addMVar(shape=(window_time,), vtype=GRB.CONTINUOUS, name='P_balance')
            cost_window = model.addVar(vtype=GRB.CONTINUOUS, name='cost_window')
            cost_hour = model.addMVar(shape=(window_time,), vtype=GRB.CONTINUOUS, name='cost_hour')
            cost_genera = model.addMVar(shape=(window_time,), vtype=GRB.CONTINUOUS, name='cost_genera')

            model.setObjective(cost_window, GRB.MINIMIZE)

            model.addConstr(cost_window == cost_hour.sum())
            for t in range(window_time):
                model.addConstr(cost_hour[t] == cost_genera[t] + cost_balance[t])
                model.addConstr(cost_balance[t] == (P_balance[t] * self.Prices[t + current_time]) / 2)
                model.addConstr(P_balance[t] == grb.max_(P_ij[t, 0, 0], 0))
                model.addConstr(cost_genera[t] == (self.a_29_2 * P_DG_29[t] * P_DG_29[t] + self.a_29_1 * P_DG_29[t] + self.a_29_0 +
                                self.a_64_2 * P_DG_64[t] * P_DG_64[t] + self.a_64_1 * P_DG_64[t] + self.a_64_0) / 2)

            model.addConstr(P_DG_29[0] <= P_DG_29_init + self.gen_ramp / 2)
            model.addConstr(P_DG_64[0] <= P_DG_64_init + self.gen_ramp / 2)
            model.addConstr(P_DG_29[0] >= P_DG_29_init - self.gen_ramp / 2)
            model.addConstr(P_DG_64[0] >= P_DG_64_init - self.gen_ramp / 2)
            if window_time > 1:
                for t in range(window_time - 1):
                    model.addConstr(P_DG_29[t + 1] <= P_DG_29[t] + self.gen_ramp / 2)
                    model.addConstr(P_DG_64[t + 1] <= P_DG_64[t] + self.gen_ramp / 2)
                    model.addConstr(P_DG_29[t + 1] >= P_DG_29[t] - self.gen_ramp / 2)
                    model.addConstr(P_DG_64[t + 1] >= P_DG_64[t] - self.gen_ramp / 2)

            model.addConstr(SOC_BSS_12[0] == SOC_BSS_12_init + (P_BSS_ch_12[0] - P_BSS_dch_12[0]) / 2)
            if window_time > 1:
                for t in range(window_time - 1):
                    model.addConstr(SOC_BSS_12[t + 1] == SOC_BSS_12[t] + (P_BSS_ch_12[t + 1] - P_BSS_dch_12[t + 1]) / 2)
            model.addConstr(SOC_BSS_34[0] == SOC_BSS_34_init + (P_BSS_ch_34[0] - P_BSS_dch_34[0]) / 2)
            if window_time > 1:
                for t in range(window_time - 1):
                    model.addConstr(SOC_BSS_34[t + 1] == SOC_BSS_34[t] + (P_BSS_ch_34[t + 1] - P_BSS_dch_34[t + 1]) / 2)
            model.addConstr(SOC_BSS_68[0] == SOC_BSS_68_init + (P_BSS_ch_68[0] - P_BSS_dch_68[0]) / 2)
            if window_time > 1:
                for t in range(window_time - 1):
                    model.addConstr(SOC_BSS_68[t + 1] == SOC_BSS_68[t] + (P_BSS_ch_68[t + 1] - P_BSS_dch_68[t + 1]) / 2)
            model.addConstr(SOC_BSS_103[0] == SOC_BSS_103_init + (P_BSS_ch_103[0] - P_BSS_dch_103[0]) / 2)
            if window_time > 1:
                for t in range(window_time - 1):
                    model.addConstr(SOC_BSS_103[t + 1] == SOC_BSS_103[t] + (P_BSS_ch_103[t + 1] - P_BSS_dch_103[t + 1]) / 2)

            for t in range(window_time):
                pass
                pass
            for t in range(window_time):
                pass
                pass
            for t in range(window_time):
                pass
                pass
            for t in range(window_time):
                pass
                pass
            print("current_time", current_time)

            for t in range(window_time):
                model.addConstr(v_i[t, 0, 0] == 1)

                for i in range(118):
                    model.addConstr(Loss_i[t, 0, i] == self.list_r_x_pu[i][0] * l_ij[t, 0, i])
                model.addConstr(Loss[t] == Loss_i[t].sum())

                print("t", t)
                for i in range(118):
                    model.addConstr(PP_ij[t, 0, i] == P_ij[t, 0, i] * P_ij[t, 0, i])
                    model.addConstr(QQ_ij[t, 0, i] == Q_ij[t, 0, i] * Q_ij[t, 0, i])

                model.addConstr(-self.list_p_q_pu_t[t + current_time][0][0] ==
                                (P_ij[t, 0, 1] + P_ij[t, 0, 62] + P_ij[t, 0, 99])
                                - (P_ij[t, 0, 0] - self.list_r_x_pu[0][0] * l_ij[t, 0, 0]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][0][1] ==
                                (Q_ij[t, 0, 1] + Q_ij[t, 0, 62] + Q_ij[t, 0, 99])
                                - (Q_ij[t, 0, 0] - self.list_r_x_pu[0][1] * l_ij[t, 0, 0]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][1][0] ==
                                (P_ij[t, 0, 2] + P_ij[t, 0, 3] + P_ij[t, 0, 9])
                                - (P_ij[t, 0, 1] - self.list_r_x_pu[1][0] * l_ij[t, 0, 1]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][1][1] ==
                                (Q_ij[t, 0, 2] + Q_ij[t, 0, 3] + Q_ij[t, 0, 9])
                                - (Q_ij[t, 0, 1] - self.list_r_x_pu[1][1] * l_ij[t, 0, 1]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][2][0] ==
                                -(P_ij[t, 0, 2] - self.list_r_x_pu[2][0] * l_ij[t, 0, 2]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][2][1] ==
                                -(Q_ij[t, 0, 2] - self.list_r_x_pu[2][1] * l_ij[t, 0, 2]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][3][0] ==
                                (P_ij[t, 0, 4] + P_ij[t, 0, 27])
                                - (P_ij[t, 0, 3] - self.list_r_x_pu[3][0] * l_ij[t, 0, 3]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][3][1] ==
                                (Q_ij[t, 0, 4] + Q_ij[t, 0, 27])
                                - (Q_ij[t, 0, 3] - self.list_r_x_pu[3][1] * l_ij[t, 0, 3]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][4][0] ==
                                (P_ij[t, 0, 5])
                                - (P_ij[t, 0, 4] - self.list_r_x_pu[4][0] * l_ij[t, 0, 4]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][4][1] ==
                                (Q_ij[t, 0, 5])
                                - (Q_ij[t, 0, 4] - self.list_r_x_pu[4][1] * l_ij[t, 0, 4]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][5][0] ==
                                (P_ij[t, 0, 6])
                                - (P_ij[t, 0, 5] - self.list_r_x_pu[5][0] * l_ij[t, 0, 5]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][5][1] ==
                                (Q_ij[t, 0, 6])
                                - (Q_ij[t, 0, 5] - self.list_r_x_pu[5][1] * l_ij[t, 0, 5]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][6][0] ==
                                (P_ij[t, 0, 7])
                                - (P_ij[t, 0, 6] - self.list_r_x_pu[6][0] * l_ij[t, 0, 6]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][6][1] ==
                                (Q_ij[t, 0, 7])
                                - (Q_ij[t, 0, 6] - self.list_r_x_pu[6][1] * l_ij[t, 0, 6]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][7][0] ==
                                (P_ij[t, 0, 8])
                                - (P_ij[t, 0, 7] - self.list_r_x_pu[7][0] * l_ij[t, 0, 7]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][7][1] ==
                                (Q_ij[t, 0, 8])
                                - (Q_ij[t, 0, 7] - self.list_r_x_pu[7][1] * l_ij[t, 0, 7]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][8][0] ==
                                -(P_ij[t, 0, 8] - self.list_r_x_pu[8][0] * l_ij[t, 0, 8]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][8][1] ==
                                -(Q_ij[t, 0, 8] - self.list_r_x_pu[8][1] * l_ij[t, 0, 8]))

                model.addConstr(
                    -self.list_p_q_pu_t[t + current_time][9][0] + self.wind_11_avail_list[t + current_time] ==
                    (P_ij[t, 0, 10])
                    - (P_ij[t, 0, 9] - self.list_r_x_pu[9][0] * l_ij[t, 0, 9]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][9][1] + Q_wt_11[t] ==
                                (Q_ij[t, 0, 10])
                                - (Q_ij[t, 0, 9] - self.list_r_x_pu[9][1] * l_ij[t, 0, 9]))

                model.addConstr(
                    -self.list_p_q_pu_t[t + current_time][10][0] - P_BSS_ch_12[t] * 1.02 + P_BSS_dch_12[t] * 0.98 ==
                    (P_ij[t, 0, 11] + P_ij[t, 0, 17])
                    - (P_ij[t, 0, 10] - self.list_r_x_pu[10][0] * l_ij[t, 0, 10]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][10][1] ==
                                (Q_ij[t, 0, 11] + Q_ij[t, 0, 17])
                                - (Q_ij[t, 0, 10] - self.list_r_x_pu[10][1] * l_ij[t, 0, 10]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][11][0] ==
                                (P_ij[t, 0, 12])
                                - (P_ij[t, 0, 11] - self.list_r_x_pu[11][0] * l_ij[t, 0, 11]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][11][1] ==
                                (Q_ij[t, 0, 12])
                                - (Q_ij[t, 0, 11] - self.list_r_x_pu[11][1] * l_ij[t, 0, 11]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][12][0] ==
                                (P_ij[t, 0, 13])
                                - (P_ij[t, 0, 12] - self.list_r_x_pu[12][0] * l_ij[t, 0, 12]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][12][1] ==
                                (Q_ij[t, 0, 13])
                                - (Q_ij[t, 0, 12] - self.list_r_x_pu[12][1] * l_ij[t, 0, 12]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][13][0] ==
                                (P_ij[t, 0, 14])
                                - (P_ij[t, 0, 13] - self.list_r_x_pu[13][0] * l_ij[t, 0, 13]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][13][1] ==
                                (Q_ij[t, 0, 14])
                                - (Q_ij[t, 0, 13] - self.list_r_x_pu[13][1] * l_ij[t, 0, 13]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][14][0] ==
                                (P_ij[t, 0, 15])
                                - (P_ij[t, 0, 14] - self.list_r_x_pu[14][0] * l_ij[t, 0, 14]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][14][1] ==
                                (Q_ij[t, 0, 15])
                                - (Q_ij[t, 0, 14] - self.list_r_x_pu[14][1] * l_ij[t, 0, 14]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][15][0] ==
                                (P_ij[t, 0, 16])
                                - (P_ij[t, 0, 15] - self.list_r_x_pu[15][0] * l_ij[t, 0, 15]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][15][1] ==
                                (Q_ij[t, 0, 16])
                                - (Q_ij[t, 0, 15] - self.list_r_x_pu[15][1] * l_ij[t, 0, 15]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][16][0] ==
                                -(P_ij[t, 0, 16] - self.list_r_x_pu[16][0] * l_ij[t, 0, 16]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][16][1] ==
                                -(Q_ij[t, 0, 16] - self.list_r_x_pu[16][1] * l_ij[t, 0, 16]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][17][0] ==
                                (P_ij[t, 0, 18])
                                - (P_ij[t, 0, 17] - self.list_r_x_pu[17][0] * l_ij[t, 0, 17]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][17][1] ==
                                (Q_ij[t, 0, 18])
                                - (Q_ij[t, 0, 17] - self.list_r_x_pu[17][1] * l_ij[t, 0, 17]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][18][0] ==
                                (P_ij[t, 0, 19])
                                - (P_ij[t, 0, 18] - self.list_r_x_pu[18][0] * l_ij[t, 0, 18]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][18][1] + Q_svc_20[t] ==
                                (Q_ij[t, 0, 19])
                                - (Q_ij[t, 0, 18] - self.list_r_x_pu[18][1] * l_ij[t, 0, 18]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][19][0] ==
                                (P_ij[t, 0, 20])
                                - (P_ij[t, 0, 19] - self.list_r_x_pu[19][0] * l_ij[t, 0, 19]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][19][1] ==
                                (Q_ij[t, 0, 20])
                                - (Q_ij[t, 0, 19] - self.list_r_x_pu[19][1] * l_ij[t, 0, 19]))

                model.addConstr(
                    -self.list_p_q_pu_t[t + current_time][20][0] + self.pv_22_avail_list[t + current_time] ==
                    (P_ij[t, 0, 21])
                    - (P_ij[t, 0, 20] - self.list_r_x_pu[20][0] * l_ij[t, 0, 20]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][20][1] + Q_pv_22[t] ==
                                (Q_ij[t, 0, 21])
                                * (Q_ij[t, 0, 20] - self.list_r_x_pu[20][1] * l_ij[t, 0, 20]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][21][0] ==
                                (P_ij[t, 0, 22])
                                - (P_ij[t, 0, 21] - self.list_r_x_pu[21][0] * l_ij[t, 0, 21]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][21][1] ==
                                (Q_ij[t, 0, 22])
                                - (Q_ij[t, 0, 21] - self.list_r_x_pu[21][1] * l_ij[t, 0, 21]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][22][0] ==
                                (P_ij[t, 0, 23])
                                - (P_ij[t, 0, 22] - self.list_r_x_pu[22][0] * l_ij[t, 0, 22]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][22][1] ==
                                (Q_ij[t, 0, 23])
                                - (Q_ij[t, 0, 22] - self.list_r_x_pu[22][1] * l_ij[t, 0, 22]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][23][0] ==
                                (P_ij[t, 0, 24])
                                - (P_ij[t, 0, 23] - self.list_r_x_pu[23][0] * l_ij[t, 0, 23]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][23][1] ==
                                (Q_ij[t, 0, 24])
                                - (Q_ij[t, 0, 23] - self.list_r_x_pu[23][1] * l_ij[t, 0, 23]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][24][0] ==
                                (P_ij[t, 0, 25])
                                - (P_ij[t, 0, 24] - self.list_r_x_pu[24][0] * l_ij[t, 0, 24]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][24][1] ==
                                (Q_ij[t, 0, 25])
                                - (Q_ij[t, 0, 24] - self.list_r_x_pu[24][1] * l_ij[t, 0, 24]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][25][0] ==
                                (P_ij[t, 0, 26])
                                - (P_ij[t, 0, 25] - self.list_r_x_pu[25][0] * l_ij[t, 0, 25]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][25][1] ==
                                (Q_ij[t, 0, 26])
                                - (Q_ij[t, 0, 25] - self.list_r_x_pu[25][1] * l_ij[t, 0, 25]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][26][0] ==
                                -(P_ij[t, 0, 26] - self.list_r_x_pu[26][0] * l_ij[t, 0, 26]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][26][1] ==
                                -(Q_ij[t, 0, 26] - self.list_r_x_pu[26][1] * l_ij[t, 0, 26]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][27][0] + P_DG_29[t] ==
                                (P_ij[t, 0, 28])
                                - (P_ij[t, 0, 27] - self.list_r_x_pu[27][0] * l_ij[t, 0, 27]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][27][1] ==
                                (Q_ij[t, 0, 28])
                                - (Q_ij[t, 0, 27] - self.list_r_x_pu[27][1] * l_ij[t, 0, 27]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][28][0] ==
                                (P_ij[t, 0, 29] + P_ij[t, 0, 37] + P_ij[t, 0, 54])
                                - (P_ij[t, 0, 28] - self.list_r_x_pu[28][0] * l_ij[t, 0, 28]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][28][1] ==
                                (Q_ij[t, 0, 29] + Q_ij[t, 0, 37] + Q_ij[t, 0, 54])
                                - (Q_ij[t, 0, 28] - self.list_r_x_pu[28][1] * l_ij[t, 0, 28]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][29][0] ==
                                (P_ij[t, 0, 30] + P_ij[t, 0, 35])
                                - (P_ij[t, 0, 29] - self.list_r_x_pu[29][0] * l_ij[t, 0, 29]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][29][1] ==
                                (Q_ij[t, 0, 30] + Q_ij[t, 0, 35])
                                - (Q_ij[t, 0, 29] - self.list_r_x_pu[29][1] * l_ij[t, 0, 29]))

                model.addConstr(
                    -self.list_p_q_pu_t[t + current_time][30][0] + self.wind_32_avail_list[t + current_time] ==
                    (P_ij[t, 0, 31])
                    - (P_ij[t, 0, 30] - self.list_r_x_pu[30][0] * l_ij[t, 0, 30]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][30][1] + Q_wt_32[t] ==
                                (Q_ij[t, 0, 31])
                                - (Q_ij[t, 0, 30] - self.list_r_x_pu[30][1] * l_ij[t, 0, 30]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][31][0] ==
                                (P_ij[t, 0, 32])
                                - (P_ij[t, 0, 31] - self.list_r_x_pu[31][0] * l_ij[t, 0, 31]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][31][1] ==
                                (Q_ij[t, 0, 32])
                                - (Q_ij[t, 0, 31] - self.list_r_x_pu[31][1] * l_ij[t, 0, 31]))

                model.addConstr(
                    -self.list_p_q_pu_t[t + current_time][32][0] - P_BSS_ch_34[t] * 1.02 + P_BSS_dch_34[t] * 0.98 ==
                    (P_ij[t, 0, 33])
                    - (P_ij[t, 0, 32] - self.list_r_x_pu[32][0] * l_ij[t, 0, 32]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][32][1] ==
                                (Q_ij[t, 0, 33])
                                - (Q_ij[t, 0, 32] - self.list_r_x_pu[32][1] * l_ij[t, 0, 32]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][33][0] ==
                                (P_ij[t, 0, 34])
                                - (P_ij[t, 0, 33] - self.list_r_x_pu[33][0] * l_ij[t, 0, 33]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][33][1] ==
                                (Q_ij[t, 0, 34])
                                - (Q_ij[t, 0, 33] - self.list_r_x_pu[33][1] * l_ij[t, 0, 33]))

                model.addConstr(
                    -self.list_p_q_pu_t[t + current_time][34][0] + self.pv_36_avail_list[t + current_time] ==
                    (P_ij[t, 0, 46])
                    - (P_ij[t, 0, 34] - self.list_r_x_pu[34][0] * l_ij[t, 0, 34]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][34][1] + Q_pv_36[t] ==
                                (Q_ij[t, 0, 46])
                                - (Q_ij[t, 0, 34] - self.list_r_x_pu[34][1] * l_ij[t, 0, 34]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][35][0] ==
                                (P_ij[t, 0, 36])
                                - (P_ij[t, 0, 35] - self.list_r_x_pu[35][0] * l_ij[t, 0, 35]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][35][1] ==
                                (Q_ij[t, 0, 36])
                                - (Q_ij[t, 0, 35] - self.list_r_x_pu[35][1] * l_ij[t, 0, 35]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][36][0] ==
                                -(P_ij[t, 0, 36] - self.list_r_x_pu[36][0] * l_ij[t, 0, 36]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][36][1] ==
                                -(Q_ij[t, 0, 36] - self.list_r_x_pu[36][1] * l_ij[t, 0, 36]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][37][0] ==
                                (P_ij[t, 0, 38])
                                - (P_ij[t, 0, 37] - self.list_r_x_pu[37][0] * l_ij[t, 0, 37]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][37][1] ==
                                (Q_ij[t, 0, 38])
                                - (Q_ij[t, 0, 37] - self.list_r_x_pu[37][1] * l_ij[t, 0, 37]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][38][0] ==
                                (P_ij[t, 0, 39])
                                - (P_ij[t, 0, 38] - self.list_r_x_pu[38][0] * l_ij[t, 0, 38]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][38][1] ==
                                (Q_ij[t, 0, 39])
                                - (Q_ij[t, 0, 38] - self.list_r_x_pu[38][1] * l_ij[t, 0, 38]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][39][0] ==
                                (P_ij[t, 0, 40])
                                - (P_ij[t, 0, 39] - self.list_r_x_pu[39][0] * l_ij[t, 0, 39]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][39][1] ==
                                (Q_ij[t, 0, 40])
                                - (Q_ij[t, 0, 39] - self.list_r_x_pu[39][1] * l_ij[t, 0, 39]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][40][0] ==
                                (P_ij[t, 0, 41])
                                - (P_ij[t, 0, 40] - self.list_r_x_pu[40][0] * l_ij[t, 0, 40]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][40][1] ==
                                (Q_ij[t, 0, 41])
                                - (Q_ij[t, 0, 40] - self.list_r_x_pu[40][1] * l_ij[t, 0, 40]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][41][0] ==
                                (P_ij[t, 0, 42])
                                - (P_ij[t, 0, 41] - self.list_r_x_pu[41][0] * l_ij[t, 0, 41]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][41][1] ==
                                (Q_ij[t, 0, 42])
                                - (Q_ij[t, 0, 41] - self.list_r_x_pu[41][1] * l_ij[t, 0, 41]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][42][0] ==
                                (P_ij[t, 0, 43])
                                - (P_ij[t, 0, 42] - self.list_r_x_pu[42][0] * l_ij[t, 0, 42]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][42][1] ==
                                (Q_ij[t, 0, 43])
                                - (Q_ij[t, 0, 42] - self.list_r_x_pu[42][1] * l_ij[t, 0, 42]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][43][0] ==
                                (P_ij[t, 0, 44])
                                - (P_ij[t, 0, 43] - self.list_r_x_pu[43][0] * l_ij[t, 0, 43]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][43][1] ==
                                (Q_ij[t, 0, 44])
                                - (Q_ij[t, 0, 43] - self.list_r_x_pu[43][1] * l_ij[t, 0, 43]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][44][0] ==
                                (P_ij[t, 0, 45])
                                - (P_ij[t, 0, 44] - self.list_r_x_pu[44][0] * l_ij[t, 0, 44]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][44][1] ==
                                (Q_ij[t, 0, 45])
                                - (Q_ij[t, 0, 44] - self.list_r_x_pu[44][1] * l_ij[t, 0, 44]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][45][0] ==
                                -(P_ij[t, 0, 45] - self.list_r_x_pu[45][0] * l_ij[t, 0, 45]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][45][1] ==
                                -(Q_ij[t, 0, 45] - self.list_r_x_pu[45][1] * l_ij[t, 0, 45]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][46][0] ==
                                (P_ij[t, 0, 47])
                                - (P_ij[t, 0, 46] - self.list_r_x_pu[46][0] * l_ij[t, 0, 46]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][46][1] ==
                                (Q_ij[t, 0, 47])
                                - (Q_ij[t, 0, 46] - self.list_r_x_pu[46][1] * l_ij[t, 0, 46]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][47][0] ==
                                (P_ij[t, 0, 48])
                                - (P_ij[t, 0, 47] - self.list_r_x_pu[47][0] * l_ij[t, 0, 47]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][47][1] ==
                                (Q_ij[t, 0, 48])
                                - (Q_ij[t, 0, 47] - self.list_r_x_pu[47][1] * l_ij[t, 0, 47]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][48][0] ==
                                (P_ij[t, 0, 49])
                                - (P_ij[t, 0, 48] - self.list_r_x_pu[48][0] * l_ij[t, 0, 48]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][48][1] ==
                                (Q_ij[t, 0, 49])
                                - (Q_ij[t, 0, 48] - self.list_r_x_pu[48][1] * l_ij[t, 0, 48]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][49][0] ==
                                (P_ij[t, 0, 50])
                                - (P_ij[t, 0, 49] - self.list_r_x_pu[49][0] * l_ij[t, 0, 49]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][49][1] ==
                                (Q_ij[t, 0, 50])
                                - (Q_ij[t, 0, 49] - self.list_r_x_pu[49][1] * l_ij[t, 0, 49]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][50][0] ==
                                (P_ij[t, 0, 51])
                                - (P_ij[t, 0, 50] - self.list_r_x_pu[50][0] * l_ij[t, 0, 50]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][50][1] ==
                                (Q_ij[t, 0, 51])
                                - (Q_ij[t, 0, 50] - self.list_r_x_pu[50][1] * l_ij[t, 0, 50]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][51][0] ==
                                (P_ij[t, 0, 52])
                                - (P_ij[t, 0, 51] - self.list_r_x_pu[51][0] * l_ij[t, 0, 51]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][51][1] ==
                                (Q_ij[t, 0, 52])
                                - (Q_ij[t, 0, 51] - self.list_r_x_pu[51][1] * l_ij[t, 0, 51]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][52][0] ==
                                (P_ij[t, 0, 53])
                                - (P_ij[t, 0, 52] - self.list_r_x_pu[52][0] * l_ij[t, 0, 52]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][52][1] ==
                                (Q_ij[t, 0, 53])
                                - (Q_ij[t, 0, 52] - self.list_r_x_pu[52][1] * l_ij[t, 0, 52]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][53][0] ==
                                -(P_ij[t, 0, 53] - self.list_r_x_pu[53][0] * l_ij[t, 0, 53]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][53][1] ==
                                -(Q_ij[t, 0, 53] - self.list_r_x_pu[53][1] * l_ij[t, 0, 53]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][54][0] ==
                                (P_ij[t, 0, 55])
                                - (P_ij[t, 0, 54] - self.list_r_x_pu[54][0] * l_ij[t, 0, 54]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][54][1] + Q_svc_56[t] ==
                                (Q_ij[t, 0, 55])
                                - (Q_ij[t, 0, 54] - self.list_r_x_pu[54][1] * l_ij[t, 0, 54]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][55][0] ==
                                (P_ij[t, 0, 56])
                                - (P_ij[t, 0, 55] - self.list_r_x_pu[55][0] * l_ij[t, 0, 55]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][55][1] ==
                                (Q_ij[t, 0, 56])
                                - (Q_ij[t, 0, 55] - self.list_r_x_pu[55][1] * l_ij[t, 0, 55]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][56][0] ==
                                (P_ij[t, 0, 57])
                                - (P_ij[t, 0, 56] - self.list_r_x_pu[56][0] * l_ij[t, 0, 56]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][56][1] ==
                                (Q_ij[t, 0, 57])
                                - (Q_ij[t, 0, 56] - self.list_r_x_pu[56][1] * l_ij[t, 0, 56]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][57][0] ==
                                (P_ij[t, 0, 58])
                                - (P_ij[t, 0, 57] - self.list_r_x_pu[57][0] * l_ij[t, 0, 57]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][57][1] ==
                                (Q_ij[t, 0, 58])
                                - (Q_ij[t, 0, 57] - self.list_r_x_pu[57][1] * l_ij[t, 0, 57]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][58][0] ==
                                (P_ij[t, 0, 59])
                                - (P_ij[t, 0, 58] - self.list_r_x_pu[58][0] * l_ij[t, 0, 58]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][58][1] ==
                                (Q_ij[t, 0, 59])
                                - (Q_ij[t, 0, 58] - self.list_r_x_pu[58][1] * l_ij[t, 0, 58]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][59][0] ==
                                (P_ij[t, 0, 60])
                                - (P_ij[t, 0, 59] - self.list_r_x_pu[59][0] * l_ij[t, 0, 59]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][59][1] ==
                                (Q_ij[t, 0, 60])
                                - (Q_ij[t, 0, 59] - self.list_r_x_pu[59][1] * l_ij[t, 0, 59]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][60][0] ==
                                (P_ij[t, 0, 61])
                                - (P_ij[t, 0, 60] - self.list_r_x_pu[60][0] * l_ij[t, 0, 60]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][60][1] ==
                                (Q_ij[t, 0, 61])
                                - (Q_ij[t, 0, 60] - self.list_r_x_pu[60][1] * l_ij[t, 0, 60]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][61][0] ==
                                -(P_ij[t, 0, 61] - self.list_r_x_pu[61][0] * l_ij[t, 0, 61]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][61][1] ==
                                -(Q_ij[t, 0, 61] - self.list_r_x_pu[61][1] * l_ij[t, 0, 61]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][62][0] + P_DG_64[t] ==
                                (P_ij[t, 0, 63])
                                - (P_ij[t, 0, 62] - self.list_r_x_pu[62][0] * l_ij[t, 0, 62]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][62][1] ==
                                (Q_ij[t, 0, 63])
                                - (Q_ij[t, 0, 62] - self.list_r_x_pu[62][1] * l_ij[t, 0, 62]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][63][0] ==
                                (P_ij[t, 0, 64] + P_ij[t, 0, 77])
                                - (P_ij[t, 0, 63] - self.list_r_x_pu[63][0] * l_ij[t, 0, 63]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][63][1] ==
                                (Q_ij[t, 0, 64] + Q_ij[t, 0, 77])
                                - (Q_ij[t, 0, 63] - self.list_r_x_pu[63][1] * l_ij[t, 0, 63]))

                model.addConstr(
                    -self.list_p_q_pu_t[t + current_time][64][0] + self.wind_66_avail_list[t + current_time] ==
                    (P_ij[t, 0, 65] + P_ij[t, 0, 88])
                    - (P_ij[t, 0, 64] - self.list_r_x_pu[64][0] * l_ij[t, 0, 64]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][64][1] + Q_wt_66[t] ==
                                (Q_ij[t, 0, 65] + Q_ij[t, 0, 88])
                                - (Q_ij[t, 0, 64] - self.list_r_x_pu[64][1] * l_ij[t, 0, 64]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][65][0] ==
                                (P_ij[t, 0, 66])
                                - (P_ij[t, 0, 65] - self.list_r_x_pu[65][0] * l_ij[t, 0, 65]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][65][1] ==
                                (Q_ij[t, 0, 66])
                                - (Q_ij[t, 0, 65] - self.list_r_x_pu[65][1] * l_ij[t, 0, 65]))

                model.addConstr(
                    -self.list_p_q_pu_t[t + current_time][66][0] - P_BSS_ch_68[t] * 1.02 + P_BSS_dch_68[t] * 0.98 ==
                    (P_ij[t, 0, 67])
                    - (P_ij[t, 0, 66] - self.list_r_x_pu[66][0] * l_ij[t, 0, 66]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][66][1] ==
                                (Q_ij[t, 0, 67])
                                - (Q_ij[t, 0, 66] - self.list_r_x_pu[66][1] * l_ij[t, 0, 66]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][67][0] ==
                                (P_ij[t, 0, 68])
                                - (P_ij[t, 0, 67] - self.list_r_x_pu[67][0] * l_ij[t, 0, 67]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][67][1] ==
                                (Q_ij[t, 0, 68])
                                - (Q_ij[t, 0, 67] - self.list_r_x_pu[67][1] * l_ij[t, 0, 67]))

                model.addConstr(
                    -self.list_p_q_pu_t[t + current_time][68][0] + self.pv_70_avail_list[t + current_time] ==
                    (P_ij[t, 0, 69])
                    - (P_ij[t, 0, 68] - self.list_r_x_pu[68][0] * l_ij[t, 0, 68]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][68][1] + Q_pv_70[t] ==
                                (Q_ij[t, 0, 69])
                                - (Q_ij[t, 0, 68] - self.list_r_x_pu[68][1] * l_ij[t, 0, 68]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][69][0] ==
                                (P_ij[t, 0, 70])
                                - (P_ij[t, 0, 69] - self.list_r_x_pu[69][0] * l_ij[t, 0, 69]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][69][1] ==
                                (Q_ij[t, 0, 70])
                                - (Q_ij[t, 0, 69] - self.list_r_x_pu[69][1] * l_ij[t, 0, 69]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][70][0] ==
                                (P_ij[t, 0, 71])
                                - (P_ij[t, 0, 70] - self.list_r_x_pu[70][0] * l_ij[t, 0, 70]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][70][1] ==
                                (Q_ij[t, 0, 71])
                                - (Q_ij[t, 0, 70] - self.list_r_x_pu[70][1] * l_ij[t, 0, 70]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][71][0] ==
                                (P_ij[t, 0, 72])
                                - (P_ij[t, 0, 71] - self.list_r_x_pu[71][0] * l_ij[t, 0, 71]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][71][1] ==
                                (Q_ij[t, 0, 72])
                                - (Q_ij[t, 0, 71] - self.list_r_x_pu[71][1] * l_ij[t, 0, 71]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][72][0] ==
                                (P_ij[t, 0, 73])
                                - (P_ij[t, 0, 72] - self.list_r_x_pu[72][0] * l_ij[t, 0, 72]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][72][1] ==
                                (Q_ij[t, 0, 73])
                                - (Q_ij[t, 0, 72] - self.list_r_x_pu[72][1] * l_ij[t, 0, 72]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][73][0] ==
                                (P_ij[t, 0, 74])
                                - (P_ij[t, 0, 73] - self.list_r_x_pu[73][0] * l_ij[t, 0, 73]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][73][1] ==
                                (Q_ij[t, 0, 74])
                                - (Q_ij[t, 0, 73] - self.list_r_x_pu[73][1] * l_ij[t, 0, 73]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][74][0] ==
                                (P_ij[t, 0, 75])
                                - (P_ij[t, 0, 74] - self.list_r_x_pu[74][0] * l_ij[t, 0, 74]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][74][1] ==
                                (Q_ij[t, 0, 75])
                                - (Q_ij[t, 0, 74] - self.list_r_x_pu[74][1] * l_ij[t, 0, 74]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][75][0] ==
                                (P_ij[t, 0, 76])
                                - (P_ij[t, 0, 75] - self.list_r_x_pu[75][0] * l_ij[t, 0, 75]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][75][1] ==
                                (Q_ij[t, 0, 76])
                                - (Q_ij[t, 0, 75] - self.list_r_x_pu[75][1] * l_ij[t, 0, 75]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][76][0] ==
                                -(P_ij[t, 0, 76] - self.list_r_x_pu[76][0] * l_ij[t, 0, 76]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][76][1] ==
                                -(Q_ij[t, 0, 76] - self.list_r_x_pu[76][1] * l_ij[t, 0, 76]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][77][0] ==
                                (P_ij[t, 0, 78])
                                - (P_ij[t, 0, 77] - self.list_r_x_pu[77][0] * l_ij[t, 0, 77]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][77][1] + Q_svc_79[t] ==
                                (Q_ij[t, 0, 78])
                                - (Q_ij[t, 0, 77] - self.list_r_x_pu[77][1] * l_ij[t, 0, 77]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][78][0] ==
                                (P_ij[t, 0, 79] + P_ij[t, 0, 85])
                                - (P_ij[t, 0, 78] - self.list_r_x_pu[78][0] * l_ij[t, 0, 78]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][78][1] ==
                                (Q_ij[t, 0, 79] + Q_ij[t, 0, 85])
                                - (Q_ij[t, 0, 78] - self.list_r_x_pu[78][1] * l_ij[t, 0, 78]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][79][0] ==
                                (P_ij[t, 0, 80])
                                - (P_ij[t, 0, 79] - self.list_r_x_pu[79][0] * l_ij[t, 0, 79]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][79][1] ==
                                (Q_ij[t, 0, 80])
                                - (Q_ij[t, 0, 79] - self.list_r_x_pu[79][1] * l_ij[t, 0, 79]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][80][0] ==
                                (P_ij[t, 0, 81])
                                - (P_ij[t, 0, 80] - self.list_r_x_pu[80][0] * l_ij[t, 0, 80]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][80][1] ==
                                (Q_ij[t, 0, 81])
                                - (Q_ij[t, 0, 80] - self.list_r_x_pu[80][1] * l_ij[t, 0, 80]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][81][0] ==
                                (P_ij[t, 0, 82])
                                - (P_ij[t, 0, 81] - self.list_r_x_pu[81][0] * l_ij[t, 0, 81]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][81][1] ==
                                (Q_ij[t, 0, 82])
                                - (Q_ij[t, 0, 81] - self.list_r_x_pu[81][1] * l_ij[t, 0, 81]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][82][0] ==
                                (P_ij[t, 0, 83])
                                - (P_ij[t, 0, 82] - self.list_r_x_pu[82][0] * l_ij[t, 0, 82]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][82][1] ==
                                (Q_ij[t, 0, 83])
                                - (Q_ij[t, 0, 82] - self.list_r_x_pu[82][1] * l_ij[t, 0, 82]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][83][0] ==
                                (P_ij[t, 0, 84])
                                - (P_ij[t, 0, 83] - self.list_r_x_pu[83][0] * l_ij[t, 0, 83]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][83][1] ==
                                (Q_ij[t, 0, 84])
                                - (Q_ij[t, 0, 83] - self.list_r_x_pu[83][1] * l_ij[t, 0, 83]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][84][0] ==
                                -(P_ij[t, 0, 84] - self.list_r_x_pu[84][0] * l_ij[t, 0, 84]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][84][1] ==
                                -(Q_ij[t, 0, 84] - self.list_r_x_pu[84][1] * l_ij[t, 0, 84]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][85][0] ==
                                (P_ij[t, 0, 86])
                                - (P_ij[t, 0, 85] - self.list_r_x_pu[85][0] * l_ij[t, 0, 85]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][85][1] ==
                                (Q_ij[t, 0, 86])
                                - (Q_ij[t, 0, 85] - self.list_r_x_pu[85][1] * l_ij[t, 0, 85]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][86][0] ==
                                (P_ij[t, 0, 87])
                                - (P_ij[t, 0, 86] - self.list_r_x_pu[86][0] * l_ij[t, 0, 86]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][86][1] ==
                                (Q_ij[t, 0, 87])
                                - (Q_ij[t, 0, 86] - self.list_r_x_pu[86][1] * l_ij[t, 0, 86]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][87][0] ==
                                -(P_ij[t, 0, 87] - self.list_r_x_pu[87][0] * l_ij[t, 0, 87]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][87][1] ==
                                -(Q_ij[t, 0, 87] - self.list_r_x_pu[87][1] * l_ij[t, 0, 87]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][88][0] ==
                                (P_ij[t, 0, 89])
                                - (P_ij[t, 0, 88] - self.list_r_x_pu[88][0] * l_ij[t, 0, 88]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][88][1] ==
                                (Q_ij[t, 0, 89])
                                - (Q_ij[t, 0, 88] - self.list_r_x_pu[88][1] * l_ij[t, 0, 88]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][89][0] ==
                                (P_ij[t, 0, 90])
                                - (P_ij[t, 0, 89] - self.list_r_x_pu[89][0] * l_ij[t, 0, 89]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][89][1] ==
                                (Q_ij[t, 0, 90])
                                - (Q_ij[t, 0, 89] - self.list_r_x_pu[89][1] * l_ij[t, 0, 89]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][90][0] ==
                                (P_ij[t, 0, 91] + P_ij[t, 0, 95])
                                - (P_ij[t, 0, 90] - self.list_r_x_pu[90][0] * l_ij[t, 0, 90]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][90][1] ==
                                (Q_ij[t, 0, 91] + Q_ij[t, 0, 95])
                                - (Q_ij[t, 0, 90] - self.list_r_x_pu[90][1] * l_ij[t, 0, 90]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][91][0] ==
                                (P_ij[t, 0, 92])
                                - (P_ij[t, 0, 91] - self.list_r_x_pu[91][0] * l_ij[t, 0, 91]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][91][1] ==
                                (Q_ij[t, 0, 92])
                                - (Q_ij[t, 0, 91] - self.list_r_x_pu[91][1] * l_ij[t, 0, 91]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][92][0] ==
                                (P_ij[t, 0, 93])
                                - (P_ij[t, 0, 92] - self.list_r_x_pu[92][0] * l_ij[t, 0, 92]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][92][1] ==
                                (Q_ij[t, 0, 93])
                                - (Q_ij[t, 0, 92] - self.list_r_x_pu[92][1] * l_ij[t, 0, 92]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][93][0] == (1 * P_ij[t, 0, 94])
                                - (1 * (P_ij[t, 0, 93] - self.list_r_x_pu[93][0] * l_ij[t, 0, 93])))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][93][1] == (1 * Q_ij[t, 0, 94])
                                - (1 * (Q_ij[t, 0, 93] - self.list_r_x_pu[93][1] * l_ij[t, 0, 93])))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][94][0] ==
                                -(1 * (P_ij[t, 0, 94] - self.list_r_x_pu[94][0] * l_ij[t, 0, 94])))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][94][1] ==
                                -(1 * (Q_ij[t, 0, 94] - self.list_r_x_pu[94][1] * l_ij[t, 0, 94])))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][95][0] ==
                                (P_ij[t, 0, 96])
                                - (P_ij[t, 0, 95] - self.list_r_x_pu[95][0] * l_ij[t, 0, 95]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][95][1] ==
                                (Q_ij[t, 0, 96])
                                - (Q_ij[t, 0, 95] - self.list_r_x_pu[95][1] * l_ij[t, 0, 95]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][96][0] ==
                                (P_ij[t, 0, 97])
                                - (P_ij[t, 0, 96] - self.list_r_x_pu[96][0] * l_ij[t, 0, 96]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][96][1] ==
                                (Q_ij[t, 0, 97])
                                - (Q_ij[t, 0, 96] - self.list_r_x_pu[96][1] * l_ij[t, 0, 96]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][97][0] ==
                                (P_ij[t, 0, 98])
                                - (P_ij[t, 0, 97] - self.list_r_x_pu[97][0] * l_ij[t, 0, 97]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][97][1] ==
                                (Q_ij[t, 0, 98])
                                - (Q_ij[t, 0, 97] - self.list_r_x_pu[97][1] * l_ij[t, 0, 97]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][98][0] ==
                                -(P_ij[t, 0, 98] - self.list_r_x_pu[98][0] * l_ij[t, 0, 98]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][98][1] ==
                                -(Q_ij[t, 0, 98] - self.list_r_x_pu[98][1] * l_ij[t, 0, 98]))

                model.addConstr(
                    -self.list_p_q_pu_t[t + current_time][99][0] + self.wind_101_avail_list[t + current_time] ==
                    (P_ij[t, 0, 100] + P_ij[t, 0, 113])
                    - (P_ij[t, 0, 99] - self.list_r_x_pu[99][0] * l_ij[t, 0, 99]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][99][1] + Q_wt_101[t] ==
                                (Q_ij[t, 0, 100] + Q_ij[t, 0, 113])
                                - (Q_ij[t, 0, 99] - self.list_r_x_pu[99][1] * l_ij[t, 0, 99]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][100][0] ==
                                (P_ij[t, 0, 101])
                                - (P_ij[t, 0, 100] - self.list_r_x_pu[100][0] * l_ij[t, 0, 100]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][100][1] ==
                                (Q_ij[t, 0, 101])
                                - (Q_ij[t, 0, 100] - self.list_r_x_pu[100][1] * l_ij[t, 0, 100]))

                model.addConstr(
                    -self.list_p_q_pu_t[t + current_time][101][0] - P_BSS_ch_103[t] * 1.02 + P_BSS_dch_103[
                        t] * 0.98 ==
                    (P_ij[t, 0, 102])
                    - (P_ij[t, 0, 101] - self.list_r_x_pu[101][0] * l_ij[t, 0, 101]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][101][1] ==
                                (Q_ij[t, 0, 102])
                                - (Q_ij[t, 0, 101] - self.list_r_x_pu[101][1] * l_ij[t, 0, 101]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][102][0] ==
                                (P_ij[t, 0, 103])
                                - (P_ij[t, 0, 102] - self.list_r_x_pu[102][0] * l_ij[t, 0, 102]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][102][1] ==
                                (Q_ij[t, 0, 103])
                                - (Q_ij[t, 0, 102] - self.list_r_x_pu[102][1] * l_ij[t, 0, 102]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][103][0] ==
                                (P_ij[t, 0, 104])
                                - (P_ij[t, 0, 103] - self.list_r_x_pu[103][0] * l_ij[t, 0, 103]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][103][1] + Q_svc_105[t] ==
                                (Q_ij[t, 0, 104])
                                - (Q_ij[t, 0, 103] - self.list_r_x_pu[103][1] * l_ij[t, 0, 103]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][104][0] ==
                                (P_ij[t, 0, 105])
                                - (P_ij[t, 0, 104] - self.list_r_x_pu[104][0] * l_ij[t, 0, 104]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][104][1] ==
                                (Q_ij[t, 0, 105])
                                - (Q_ij[t, 0, 104] - self.list_r_x_pu[104][1] * l_ij[t, 0, 104]))

                model.addConstr(
                    -self.list_p_q_pu_t[t + current_time][105][0] + self.pv_107_avail_list[t + current_time] ==
                    (P_ij[t, 0, 106])
                    - (P_ij[t, 0, 105] - self.list_r_x_pu[105][0] * l_ij[t, 0, 105]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][105][1] + Q_pv_107[t] ==
                                (Q_ij[t, 0, 106])
                                - (Q_ij[t, 0, 105] - self.list_r_x_pu[105][1] * l_ij[t, 0, 105]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][106][0] ==
                                (P_ij[t, 0, 107])
                                - (P_ij[t, 0, 106] - self.list_r_x_pu[106][0] * l_ij[t, 0, 106]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][106][1] ==
                                (Q_ij[t, 0, 107])
                                - (Q_ij[t, 0, 106] - self.list_r_x_pu[106][1] * l_ij[t, 0, 106]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][107][0] ==
                                (P_ij[t, 0, 108])
                                - (P_ij[t, 0, 107] - self.list_r_x_pu[107][0] * l_ij[t, 0, 107]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][107][1] ==
                                (Q_ij[t, 0, 108])
                                - (Q_ij[t, 0, 107] - self.list_r_x_pu[107][1] * l_ij[t, 0, 107]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][108][0] ==
                                (P_ij[t, 0, 109])
                                - (P_ij[t, 0, 108] - self.list_r_x_pu[108][0] * l_ij[t, 0, 108]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][108][1] ==
                                (Q_ij[t, 0, 109])
                                - (Q_ij[t, 0, 108] - self.list_r_x_pu[108][1] * l_ij[t, 0, 108]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][109][0] ==
                                (P_ij[t, 0, 110] + P_ij[t, 0, 111])
                                - (P_ij[t, 0, 109] - self.list_r_x_pu[109][0] * l_ij[t, 0, 109]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][109][1] ==
                                (Q_ij[t, 0, 110] + Q_ij[t, 0, 111])
                                - (Q_ij[t, 0, 109] - self.list_r_x_pu[109][1] * l_ij[t, 0, 109]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][110][0] ==
                                -(P_ij[t, 0, 110] - self.list_r_x_pu[110][0] * l_ij[t, 0, 110]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][110][1] ==
                                -(Q_ij[t, 0, 110] - self.list_r_x_pu[110][1] * l_ij[t, 0, 110]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][111][0] ==
                                (P_ij[t, 0, 112])
                                - (P_ij[t, 0, 111] - self.list_r_x_pu[111][0] * l_ij[t, 0, 111]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][111][1] ==
                                (Q_ij[t, 0, 112])
                                - (Q_ij[t, 0, 111] - self.list_r_x_pu[111][1] * l_ij[t, 0, 111]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][112][0] ==
                                -(P_ij[t, 0, 112] - self.list_r_x_pu[112][0] * l_ij[t, 0, 112]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][112][1] ==
                                -(Q_ij[t, 0, 112] - self.list_r_x_pu[112][1] * l_ij[t, 0, 112]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][113][0] ==
                                (P_ij[t, 0, 114])
                                - (P_ij[t, 0, 113] - self.list_r_x_pu[113][0] * l_ij[t, 0, 113]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][113][1] ==
                                (Q_ij[t, 0, 114])
                                - (Q_ij[t, 0, 113] - self.list_r_x_pu[113][1] * l_ij[t, 0, 113]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][114][0] ==
                                (P_ij[t, 0, 115])
                                - (P_ij[t, 0, 114] - self.list_r_x_pu[114][0] * l_ij[t, 0, 114]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][114][1] ==
                                (Q_ij[t, 0, 115])
                                - (Q_ij[t, 0, 114] - self.list_r_x_pu[114][1] * l_ij[t, 0, 114]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][115][0] ==
                                (P_ij[t, 0, 116])
                                - (P_ij[t, 0, 115] - self.list_r_x_pu[115][0] * l_ij[t, 0, 115]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][115][1] ==
                                (Q_ij[t, 0, 116])
                                - (Q_ij[t, 0, 115] - self.list_r_x_pu[115][1] * l_ij[t, 0, 115]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][116][0] ==
                                (P_ij[t, 0, 117])
                                - (P_ij[t, 0, 116] - self.list_r_x_pu[116][0] * l_ij[t, 0, 116]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][116][1] ==
                                (Q_ij[t, 0, 117])
                                - (Q_ij[t, 0, 116] - self.list_r_x_pu[116][1] * l_ij[t, 0, 116]))

                model.addConstr(-self.list_p_q_pu_t[t + current_time][117][0] ==
                                -(P_ij[t, 0, 117] - self.list_r_x_pu[117][0] * l_ij[t, 0, 117]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][117][1] ==
                                -(Q_ij[t, 0, 117] - self.list_r_x_pu[117][1] * l_ij[t, 0, 117]))

                model.addConstr((v_i[t, 0, 0] - v_i[t, 0, 1] - 2 * (
                            self.list_r_x_pu[0][0] * P_ij[t, 0, 0] + self.list_r_x_pu[0][1] * Q_ij[t, 0, 0]) + (
                                             self.list_r_x_pu[0][0] * self.list_r_x_pu[0][0] + self.list_r_x_pu[0][
                                         1] * self.list_r_x_pu[0][1]) * l_ij[t, 0, 0]) == 0)

                model.addConstr((v_i[t, 0, 1] - v_i[t, 0, 2] - 2 * (
                            self.list_r_x_pu[1][0] * P_ij[t, 0, 1] + self.list_r_x_pu[1][1] * Q_ij[t, 0, 1]) + (
                                             self.list_r_x_pu[1][0] * self.list_r_x_pu[1][0] + self.list_r_x_pu[1][
                                         1] * self.list_r_x_pu[1][1]) * l_ij[t, 0, 1]) == 0)

                model.addConstr((v_i[t, 0, 2] - v_i[t, 0, 3] - 2 * (
                            self.list_r_x_pu[2][0] * P_ij[t, 0, 2] + self.list_r_x_pu[2][1] * Q_ij[t, 0, 2]) + (
                                             self.list_r_x_pu[2][0] * self.list_r_x_pu[2][0] + self.list_r_x_pu[2][
                                         1] * self.list_r_x_pu[2][1]) * l_ij[t, 0, 2]) == 0)

                model.addConstr((v_i[t, 0, 2] - v_i[t, 0, 4] - 2 * (
                            self.list_r_x_pu[3][0] * P_ij[t, 0, 3] + self.list_r_x_pu[3][1] * Q_ij[t, 0, 3]) + (
                                             self.list_r_x_pu[3][0] * self.list_r_x_pu[3][0] + self.list_r_x_pu[3][
                                         1] * self.list_r_x_pu[3][1]) * l_ij[t, 0, 3]) == 0)

                model.addConstr((v_i[t, 0, 4] - v_i[t, 0, 5] - 2 * (
                            self.list_r_x_pu[4][0] * P_ij[t, 0, 4] + self.list_r_x_pu[4][1] * Q_ij[t, 0, 4]) + (
                                             self.list_r_x_pu[4][0] * self.list_r_x_pu[4][0] + self.list_r_x_pu[4][
                                         1] * self.list_r_x_pu[4][1]) * l_ij[t, 0, 4]) == 0)

                model.addConstr((v_i[t, 0, 5] - v_i[t, 0, 6] - 2 * (
                            self.list_r_x_pu[5][0] * P_ij[t, 0, 5] + self.list_r_x_pu[5][1] * Q_ij[t, 0, 5]) + (
                                             self.list_r_x_pu[5][0] * self.list_r_x_pu[5][0] + self.list_r_x_pu[5][
                                         1] * self.list_r_x_pu[5][1]) * l_ij[t, 0, 5]) == 0)

                model.addConstr((v_i[t, 0, 6] - v_i[t, 0, 7] - 2 * (
                            self.list_r_x_pu[6][0] * P_ij[t, 0, 6] + self.list_r_x_pu[6][1] * Q_ij[t, 0, 6]) + (
                                             self.list_r_x_pu[6][0] * self.list_r_x_pu[6][0] + self.list_r_x_pu[6][
                                         1] * self.list_r_x_pu[6][1]) * l_ij[t, 0, 6]) == 0)

                model.addConstr((v_i[t, 0, 7] - v_i[t, 0, 8] - 2 * (
                            self.list_r_x_pu[7][0] * P_ij[t, 0, 7] + self.list_r_x_pu[7][1] * Q_ij[t, 0, 7]) + (
                                             self.list_r_x_pu[7][0] * self.list_r_x_pu[7][0] + self.list_r_x_pu[7][
                                         1] * self.list_r_x_pu[7][1]) * l_ij[t, 0, 7]) == 0)

                model.addConstr((v_i[t, 0, 8] - v_i[t, 0, 9] - 2 * (
                            self.list_r_x_pu[8][0] * P_ij[t, 0, 8] + self.list_r_x_pu[8][1] * Q_ij[t, 0, 8]) + (
                                             self.list_r_x_pu[8][0] * self.list_r_x_pu[8][0] + self.list_r_x_pu[8][
                                         1] * self.list_r_x_pu[8][1]) * l_ij[t, 0, 8]) == 0)

                model.addConstr((v_i[t, 0, 2] - v_i[t, 0, 10] - 2 * (
                            self.list_r_x_pu[9][0] * P_ij[t, 0, 9] + self.list_r_x_pu[9][1] * Q_ij[t, 0, 9]) + (
                                             self.list_r_x_pu[9][0] * self.list_r_x_pu[9][0] + self.list_r_x_pu[9][
                                         1] * self.list_r_x_pu[9][1]) * l_ij[t, 0, 9]) == 0)

                model.addConstr((v_i[t, 0, 10] - v_i[t, 0, 11] - 2 * (
                            self.list_r_x_pu[10][0] * P_ij[t, 0, 10] + self.list_r_x_pu[10][1] * Q_ij[t, 0, 10]) + (
                                             self.list_r_x_pu[10][0] * self.list_r_x_pu[10][0] +
                                             self.list_r_x_pu[10][1] * self.list_r_x_pu[10][1]) * l_ij[
                                     t, 0, 10]) == 0)

                model.addConstr((v_i[t, 0, 11] - v_i[t, 0, 12] - 2 * (
                            self.list_r_x_pu[11][0] * P_ij[t, 0, 11] + self.list_r_x_pu[11][1] * Q_ij[t, 0, 11]) + (
                                             self.list_r_x_pu[11][0] * self.list_r_x_pu[11][0] +
                                             self.list_r_x_pu[11][1] * self.list_r_x_pu[11][1]) * l_ij[
                                     t, 0, 11]) == 0)

                model.addConstr((v_i[t, 0, 12] - v_i[t, 0, 13] - 2 * (
                            self.list_r_x_pu[12][0] * P_ij[t, 0, 12] + self.list_r_x_pu[12][1] * Q_ij[t, 0, 12]) + (
                                             self.list_r_x_pu[12][0] * self.list_r_x_pu[12][0] +
                                             self.list_r_x_pu[12][1] * self.list_r_x_pu[12][1]) * l_ij[
                                     t, 0, 12]) == 0)

                model.addConstr((v_i[t, 0, 13] - v_i[t, 0, 14] - 2 * (
                            self.list_r_x_pu[13][0] * P_ij[t, 0, 13] + self.list_r_x_pu[13][1] * Q_ij[t, 0, 13]) + (
                                             self.list_r_x_pu[13][0] * self.list_r_x_pu[13][0] +
                                             self.list_r_x_pu[13][1] * self.list_r_x_pu[13][1]) * l_ij[
                                     t, 0, 13]) == 0)

                model.addConstr((v_i[t, 0, 14] - v_i[t, 0, 15] - 2 * (
                            self.list_r_x_pu[14][0] * P_ij[t, 0, 14] + self.list_r_x_pu[14][1] * Q_ij[t, 0, 14]) + (
                                             self.list_r_x_pu[14][0] * self.list_r_x_pu[14][0] +
                                             self.list_r_x_pu[14][1] * self.list_r_x_pu[14][1]) * l_ij[
                                     t, 0, 14]) == 0)

                model.addConstr((v_i[t, 0, 15] - v_i[t, 0, 16] - 2 * (
                            self.list_r_x_pu[15][0] * P_ij[t, 0, 15] + self.list_r_x_pu[15][1] * Q_ij[t, 0, 15]) + (
                                             self.list_r_x_pu[15][0] * self.list_r_x_pu[15][0] +
                                             self.list_r_x_pu[15][1] * self.list_r_x_pu[15][1]) * l_ij[
                                     t, 0, 15]) == 0)

                model.addConstr((v_i[t, 0, 16] - v_i[t, 0, 17] - 2 * (
                            self.list_r_x_pu[16][0] * P_ij[t, 0, 16] + self.list_r_x_pu[16][1] * Q_ij[t, 0, 16]) + (
                                             self.list_r_x_pu[16][0] * self.list_r_x_pu[16][0] +
                                             self.list_r_x_pu[16][1] * self.list_r_x_pu[16][1]) * l_ij[
                                     t, 0, 16]) == 0)

                model.addConstr((v_i[t, 0, 11] - v_i[t, 0, 18] - 2 * (
                            self.list_r_x_pu[17][0] * P_ij[t, 0, 17] + self.list_r_x_pu[17][1] * Q_ij[t, 0, 17]) + (
                                             self.list_r_x_pu[17][0] * self.list_r_x_pu[17][0] +
                                             self.list_r_x_pu[17][1] * self.list_r_x_pu[17][1]) * l_ij[
                                     t, 0, 17]) == 0)

                model.addConstr((v_i[t, 0, 18] - v_i[t, 0, 19] - 2 * (
                            self.list_r_x_pu[18][0] * P_ij[t, 0, 18] + self.list_r_x_pu[18][1] * Q_ij[t, 0, 18]) + (
                                             self.list_r_x_pu[18][0] * self.list_r_x_pu[18][0] +
                                             self.list_r_x_pu[18][1] * self.list_r_x_pu[18][1]) * l_ij[
                                     t, 0, 18]) == 0)

                model.addConstr((v_i[t, 0, 19] - v_i[t, 0, 20] - 2 * (
                            self.list_r_x_pu[19][0] * P_ij[t, 0, 19] + self.list_r_x_pu[19][1] * Q_ij[t, 0, 19]) + (
                                             self.list_r_x_pu[19][0] * self.list_r_x_pu[19][0] +
                                             self.list_r_x_pu[19][1] * self.list_r_x_pu[19][1]) * l_ij[
                                     t, 0, 19]) == 0)

                model.addConstr((v_i[t, 0, 20] - v_i[t, 0, 21] - 2 * (
                            self.list_r_x_pu[20][0] * P_ij[t, 0, 20] + self.list_r_x_pu[20][1] * Q_ij[t, 0, 20]) + (
                                             self.list_r_x_pu[20][0] * self.list_r_x_pu[20][0] +
                                             self.list_r_x_pu[20][1] * self.list_r_x_pu[20][1]) * l_ij[
                                     t, 0, 20]) == 0)

                model.addConstr((v_i[t, 0, 21] - v_i[t, 0, 22] - 2 * (
                            self.list_r_x_pu[21][0] * P_ij[t, 0, 21] + self.list_r_x_pu[21][1] * Q_ij[t, 0, 21]) + (
                                             self.list_r_x_pu[21][0] * self.list_r_x_pu[21][0] +
                                             self.list_r_x_pu[21][1] * self.list_r_x_pu[21][1]) * l_ij[
                                     t, 0, 21]) == 0)

                model.addConstr((v_i[t, 0, 22] - v_i[t, 0, 23] - 2 * (
                            self.list_r_x_pu[22][0] * P_ij[t, 0, 22] + self.list_r_x_pu[22][1] * Q_ij[t, 0, 22]) + (
                                             self.list_r_x_pu[22][0] * self.list_r_x_pu[22][0] +
                                             self.list_r_x_pu[22][1] * self.list_r_x_pu[22][1]) * l_ij[
                                     t, 0, 22]) == 0)

                model.addConstr((v_i[t, 0, 23] - v_i[t, 0, 24] - 2 * (
                            self.list_r_x_pu[23][0] * P_ij[t, 0, 23] + self.list_r_x_pu[23][1] * Q_ij[t, 0, 23]) + (
                                             self.list_r_x_pu[23][0] * self.list_r_x_pu[23][0] +
                                             self.list_r_x_pu[23][1] * self.list_r_x_pu[23][1]) * l_ij[
                                     t, 0, 23]) == 0)

                model.addConstr((v_i[t, 0, 24] - v_i[t, 0, 25] - 2 * (
                            self.list_r_x_pu[24][0] * P_ij[t, 0, 24] + self.list_r_x_pu[24][1] * Q_ij[t, 0, 24]) + (
                                             self.list_r_x_pu[24][0] * self.list_r_x_pu[24][0] +
                                             self.list_r_x_pu[24][1] * self.list_r_x_pu[24][1]) * l_ij[
                                     t, 0, 24]) == 0)

                model.addConstr((v_i[t, 0, 25] - v_i[t, 0, 26] - 2 * (
                            self.list_r_x_pu[25][0] * P_ij[t, 0, 25] + self.list_r_x_pu[25][1] * Q_ij[t, 0, 25]) + (
                                             self.list_r_x_pu[25][0] * self.list_r_x_pu[25][0] +
                                             self.list_r_x_pu[25][1] * self.list_r_x_pu[25][1]) * l_ij[
                                     t, 0, 25]) == 0)

                model.addConstr((v_i[t, 0, 26] - v_i[t, 0, 27] - 2 * (
                            self.list_r_x_pu[26][0] * P_ij[t, 0, 26] + self.list_r_x_pu[26][1] * Q_ij[t, 0, 26]) + (
                                             self.list_r_x_pu[26][0] * self.list_r_x_pu[26][0] +
                                             self.list_r_x_pu[26][1] * self.list_r_x_pu[26][1]) * l_ij[
                                     t, 0, 26]) == 0)

                model.addConstr((v_i[t, 0, 4] - v_i[t, 0, 28] - 2 * (
                            self.list_r_x_pu[27][0] * P_ij[t, 0, 27] + self.list_r_x_pu[27][1] * Q_ij[t, 0, 27]) + (
                                             self.list_r_x_pu[27][0] * self.list_r_x_pu[27][0] +
                                             self.list_r_x_pu[27][1] * self.list_r_x_pu[27][1]) * l_ij[
                                     t, 0, 27]) == 0)

                model.addConstr((v_i[t, 0, 28] - v_i[t, 0, 29] - 2 * (
                            self.list_r_x_pu[28][0] * P_ij[t, 0, 28] + self.list_r_x_pu[28][1] * Q_ij[t, 0, 28]) + (
                                             self.list_r_x_pu[28][0] * self.list_r_x_pu[28][0] +
                                             self.list_r_x_pu[28][1] * self.list_r_x_pu[28][1]) * l_ij[
                                     t, 0, 28]) == 0)

                model.addConstr((v_i[t, 0, 29] - v_i[t, 0, 30] - 2 * (
                            self.list_r_x_pu[29][0] * P_ij[t, 0, 29] + self.list_r_x_pu[29][1] * Q_ij[t, 0, 29]) + (
                                             self.list_r_x_pu[29][0] * self.list_r_x_pu[29][0] +
                                             self.list_r_x_pu[29][1] * self.list_r_x_pu[29][1]) * l_ij[
                                     t, 0, 29]) == 0)

                model.addConstr((v_i[t, 0, 30] - v_i[t, 0, 31] - 2 * (
                            self.list_r_x_pu[30][0] * P_ij[t, 0, 30] + self.list_r_x_pu[30][1] * Q_ij[t, 0, 30]) + (
                                             self.list_r_x_pu[30][0] * self.list_r_x_pu[30][0] +
                                             self.list_r_x_pu[30][1] * self.list_r_x_pu[30][1]) * l_ij[
                                     t, 0, 30]) == 0)

                model.addConstr((v_i[t, 0, 31] - v_i[t, 0, 32] - 2 * (
                            self.list_r_x_pu[31][0] * P_ij[t, 0, 31] + self.list_r_x_pu[31][1] * Q_ij[t, 0, 31]) + (
                                             self.list_r_x_pu[31][0] * self.list_r_x_pu[31][0] +
                                             self.list_r_x_pu[31][1] * self.list_r_x_pu[31][1]) * l_ij[
                                     t, 0, 31]) == 0)

                model.addConstr((v_i[t, 0, 32] - v_i[t, 0, 33] - 2 * (
                            self.list_r_x_pu[32][0] * P_ij[t, 0, 32] + self.list_r_x_pu[32][1] * Q_ij[t, 0, 32]) + (
                                             self.list_r_x_pu[32][0] * self.list_r_x_pu[32][0] +
                                             self.list_r_x_pu[32][1] * self.list_r_x_pu[32][1]) * l_ij[
                                     t, 0, 32]) == 0)

                model.addConstr((v_i[t, 0, 33] - v_i[t, 0, 34] - 2 * (
                            self.list_r_x_pu[33][0] * P_ij[t, 0, 33] + self.list_r_x_pu[33][1] * Q_ij[t, 0, 33]) + (
                                             self.list_r_x_pu[33][0] * self.list_r_x_pu[33][0] +
                                             self.list_r_x_pu[33][1] * self.list_r_x_pu[33][1]) * l_ij[
                                     t, 0, 33]) == 0)

                model.addConstr((v_i[t, 0, 34] - v_i[t, 0, 35] - 2 * (
                            self.list_r_x_pu[34][0] * P_ij[t, 0, 34] + self.list_r_x_pu[34][1] * Q_ij[t, 0, 34]) + (
                                             self.list_r_x_pu[34][0] * self.list_r_x_pu[34][0] +
                                             self.list_r_x_pu[34][1] * self.list_r_x_pu[34][1]) * l_ij[
                                     t, 0, 34]) == 0)

                model.addConstr((v_i[t, 0, 30] - v_i[t, 0, 36] - 2 * (
                            self.list_r_x_pu[35][0] * P_ij[t, 0, 35] + self.list_r_x_pu[35][1] * Q_ij[t, 0, 35]) + (
                                             self.list_r_x_pu[35][0] * self.list_r_x_pu[35][0] +
                                             self.list_r_x_pu[35][1] * self.list_r_x_pu[35][1]) * l_ij[
                                     t, 0, 35]) == 0)

                model.addConstr((v_i[t, 0, 36] - v_i[t, 0, 37] - 2 * (
                            self.list_r_x_pu[36][0] * P_ij[t, 0, 36] + self.list_r_x_pu[36][1] * Q_ij[t, 0, 36]) + (
                                             self.list_r_x_pu[36][0] * self.list_r_x_pu[36][0] +
                                             self.list_r_x_pu[36][1] * self.list_r_x_pu[36][1]) * l_ij[
                                     t, 0, 36]) == 0)

                model.addConstr((v_i[t, 0, 29] - v_i[t, 0, 38] - 2 * (
                            self.list_r_x_pu[37][0] * P_ij[t, 0, 37] + self.list_r_x_pu[37][1] * Q_ij[t, 0, 37]) + (
                                             self.list_r_x_pu[37][0] * self.list_r_x_pu[37][0] +
                                             self.list_r_x_pu[37][1] * self.list_r_x_pu[37][1]) * l_ij[
                                     t, 0, 37]) == 0)

                model.addConstr((v_i[t, 0, 38] - v_i[t, 0, 39] - 2 * (
                            self.list_r_x_pu[38][0] * P_ij[t, 0, 38] + self.list_r_x_pu[38][1] * Q_ij[t, 0, 38]) + (
                                             self.list_r_x_pu[38][0] * self.list_r_x_pu[38][0] +
                                             self.list_r_x_pu[38][1] * self.list_r_x_pu[38][1]) * l_ij[
                                     t, 0, 38]) == 0)

                model.addConstr((v_i[t, 0, 39] - v_i[t, 0, 40] - 2 * (
                            self.list_r_x_pu[39][0] * P_ij[t, 0, 39] + self.list_r_x_pu[39][1] * Q_ij[t, 0, 39]) + (
                                             self.list_r_x_pu[39][0] * self.list_r_x_pu[39][0] +
                                             self.list_r_x_pu[39][1] * self.list_r_x_pu[39][1]) * l_ij[
                                     t, 0, 39]) == 0)

                model.addConstr((v_i[t, 0, 40] - v_i[t, 0, 41] - 2 * (
                            self.list_r_x_pu[40][0] * P_ij[t, 0, 40] + self.list_r_x_pu[40][1] * Q_ij[t, 0, 40]) + (
                                             self.list_r_x_pu[40][0] * self.list_r_x_pu[40][0] +
                                             self.list_r_x_pu[40][1] * self.list_r_x_pu[40][1]) * l_ij[
                                     t, 0, 40]) == 0)

                model.addConstr((v_i[t, 0, 41] - v_i[t, 0, 42] - 2 * (
                            self.list_r_x_pu[41][0] * P_ij[t, 0, 41] + self.list_r_x_pu[41][1] * Q_ij[t, 0, 41]) + (
                                             self.list_r_x_pu[41][0] * self.list_r_x_pu[41][0] +
                                             self.list_r_x_pu[41][1] * self.list_r_x_pu[41][1]) * l_ij[
                                     t, 0, 41]) == 0)

                model.addConstr((v_i[t, 0, 42] - v_i[t, 0, 43] - 2 * (
                            self.list_r_x_pu[42][0] * P_ij[t, 0, 42] + self.list_r_x_pu[42][1] * Q_ij[t, 0, 42]) + (
                                             self.list_r_x_pu[42][0] * self.list_r_x_pu[42][0] +
                                             self.list_r_x_pu[42][1] * self.list_r_x_pu[42][1]) * l_ij[
                                     t, 0, 42]) == 0)

                model.addConstr((v_i[t, 0, 43] - v_i[t, 0, 44] - 2 * (
                            self.list_r_x_pu[43][0] * P_ij[t, 0, 43] + self.list_r_x_pu[43][1] * Q_ij[t, 0, 43]) + (
                                             self.list_r_x_pu[43][0] * self.list_r_x_pu[43][0] +
                                             self.list_r_x_pu[43][1] * self.list_r_x_pu[43][1]) * l_ij[
                                     t, 0, 43]) == 0)

                model.addConstr((v_i[t, 0, 44] - v_i[t, 0, 45] - 2 * (
                            self.list_r_x_pu[44][0] * P_ij[t, 0, 44] + self.list_r_x_pu[44][1] * Q_ij[t, 0, 44]) + (
                                             self.list_r_x_pu[44][0] * self.list_r_x_pu[44][0] +
                                             self.list_r_x_pu[44][1] * self.list_r_x_pu[44][1]) * l_ij[
                                     t, 0, 44]) == 0)

                model.addConstr((v_i[t, 0, 45] - v_i[t, 0, 46] - 2 * (
                            self.list_r_x_pu[45][0] * P_ij[t, 0, 45] + self.list_r_x_pu[45][1] * Q_ij[t, 0, 45]) + (
                                             self.list_r_x_pu[45][0] * self.list_r_x_pu[45][0] +
                                             self.list_r_x_pu[45][1] * self.list_r_x_pu[45][1]) * l_ij[
                                     t, 0, 45]) == 0)

                model.addConstr((v_i[t, 0, 35] - v_i[t, 0, 47] - 2 * (
                            self.list_r_x_pu[46][0] * P_ij[t, 0, 46] + self.list_r_x_pu[46][1] * Q_ij[t, 0, 46]) + (
                                             self.list_r_x_pu[46][0] * self.list_r_x_pu[46][0] +
                                             self.list_r_x_pu[46][1] * self.list_r_x_pu[46][1]) * l_ij[
                                     t, 0, 46]) == 0)

                model.addConstr((v_i[t, 0, 47] - v_i[t, 0, 48] - 2 * (
                            self.list_r_x_pu[47][0] * P_ij[t, 0, 47] + self.list_r_x_pu[47][1] * Q_ij[t, 0, 47]) + (
                                             self.list_r_x_pu[47][0] * self.list_r_x_pu[47][0] +
                                             self.list_r_x_pu[47][1] * self.list_r_x_pu[47][1]) * l_ij[
                                     t, 0, 47]) == 0)

                model.addConstr((v_i[t, 0, 48] - v_i[t, 0, 49] - 2 * (
                            self.list_r_x_pu[48][0] * P_ij[t, 0, 48] + self.list_r_x_pu[48][1] * Q_ij[t, 0, 48]) + (
                                             self.list_r_x_pu[48][0] * self.list_r_x_pu[48][0] +
                                             self.list_r_x_pu[48][1] * self.list_r_x_pu[48][1]) * l_ij[
                                     t, 0, 48]) == 0)

                model.addConstr((v_i[t, 0, 49] - v_i[t, 0, 50] - 2 * (
                            self.list_r_x_pu[49][0] * P_ij[t, 0, 49] + self.list_r_x_pu[49][1] * Q_ij[t, 0, 49]) + (
                                             self.list_r_x_pu[49][0] * self.list_r_x_pu[49][0] +
                                             self.list_r_x_pu[49][1] * self.list_r_x_pu[49][1]) * l_ij[
                                     t, 0, 49]) == 0)

                model.addConstr((v_i[t, 0, 50] - v_i[t, 0, 51] - 2 * (
                            self.list_r_x_pu[50][0] * P_ij[t, 0, 50] + self.list_r_x_pu[50][1] * Q_ij[t, 0, 50]) + (
                                             self.list_r_x_pu[50][0] * self.list_r_x_pu[50][0] +
                                             self.list_r_x_pu[50][1] * self.list_r_x_pu[50][1]) * l_ij[
                                     t, 0, 50]) == 0)

                model.addConstr((v_i[t, 0, 51] - v_i[t, 0, 52] - 2 * (
                            self.list_r_x_pu[51][0] * P_ij[t, 0, 51] + self.list_r_x_pu[51][1] * Q_ij[t, 0, 51]) + (
                                             self.list_r_x_pu[51][0] * self.list_r_x_pu[51][0] +
                                             self.list_r_x_pu[51][1] * self.list_r_x_pu[51][1]) * l_ij[
                                     t, 0, 51]) == 0)

                model.addConstr((v_i[t, 0, 52] - v_i[t, 0, 53] - 2 * (
                            self.list_r_x_pu[52][0] * P_ij[t, 0, 52] + self.list_r_x_pu[52][1] * Q_ij[t, 0, 52]) + (
                                             self.list_r_x_pu[52][0] * self.list_r_x_pu[52][0] +
                                             self.list_r_x_pu[52][1] * self.list_r_x_pu[52][1]) * l_ij[
                                     t, 0, 52]) == 0)

                model.addConstr((v_i[t, 0, 53] - v_i[t, 0, 54] - 2 * (
                            self.list_r_x_pu[53][0] * P_ij[t, 0, 53] + self.list_r_x_pu[53][1] * Q_ij[t, 0, 53]) + (
                                             self.list_r_x_pu[53][0] * self.list_r_x_pu[53][0] +
                                             self.list_r_x_pu[53][1] * self.list_r_x_pu[53][1]) * l_ij[
                                     t, 0, 53]) == 0)

                model.addConstr((v_i[t, 0, 29] - v_i[t, 0, 55] - 2 * (
                            self.list_r_x_pu[54][0] * P_ij[t, 0, 54] + self.list_r_x_pu[54][1] * Q_ij[t, 0, 54]) + (
                                             self.list_r_x_pu[54][0] * self.list_r_x_pu[54][0] +
                                             self.list_r_x_pu[54][1] * self.list_r_x_pu[54][1]) * l_ij[
                                     t, 0, 54]) == 0)

                model.addConstr((v_i[t, 0, 55] - v_i[t, 0, 56] - 2 * (
                            self.list_r_x_pu[55][0] * P_ij[t, 0, 55] + self.list_r_x_pu[55][1] * Q_ij[t, 0, 55]) + (
                                             self.list_r_x_pu[55][0] * self.list_r_x_pu[55][0] +
                                             self.list_r_x_pu[55][1] * self.list_r_x_pu[55][1]) * l_ij[
                                     t, 0, 55]) == 0)

                model.addConstr((v_i[t, 0, 56] - v_i[t, 0, 57] - 2 * (
                            self.list_r_x_pu[56][0] * P_ij[t, 0, 56] + self.list_r_x_pu[56][1] * Q_ij[t, 0, 56]) + (
                                             self.list_r_x_pu[56][0] * self.list_r_x_pu[56][0] +
                                             self.list_r_x_pu[56][1] * self.list_r_x_pu[56][1]) * l_ij[
                                     t, 0, 56]) == 0)

                model.addConstr((v_i[t, 0, 57] - v_i[t, 0, 58] - 2 * (
                            self.list_r_x_pu[57][0] * P_ij[t, 0, 57] + self.list_r_x_pu[57][1] * Q_ij[t, 0, 57]) + (
                                             self.list_r_x_pu[57][0] * self.list_r_x_pu[57][0] +
                                             self.list_r_x_pu[57][1] * self.list_r_x_pu[57][1]) * l_ij[
                                     t, 0, 57]) == 0)

                model.addConstr((v_i[t, 0, 58] - v_i[t, 0, 59] - 2 * (
                            self.list_r_x_pu[58][0] * P_ij[t, 0, 58] + self.list_r_x_pu[58][1] * Q_ij[t, 0, 58]) + (
                                             self.list_r_x_pu[58][0] * self.list_r_x_pu[58][0] +
                                             self.list_r_x_pu[58][1] * self.list_r_x_pu[58][1]) * l_ij[
                                     t, 0, 58]) == 0)

                model.addConstr((v_i[t, 0, 59] - v_i[t, 0, 60] - 2 * (
                            self.list_r_x_pu[59][0] * P_ij[t, 0, 59] + self.list_r_x_pu[59][1] * Q_ij[t, 0, 59]) + (
                                             self.list_r_x_pu[59][0] * self.list_r_x_pu[59][0] +
                                             self.list_r_x_pu[59][1] * self.list_r_x_pu[59][1]) * l_ij[
                                     t, 0, 59]) == 0)

                model.addConstr((v_i[t, 0, 60] - v_i[t, 0, 61] - 2 * (
                            self.list_r_x_pu[60][0] * P_ij[t, 0, 60] + self.list_r_x_pu[60][1] * Q_ij[t, 0, 60]) + (
                                             self.list_r_x_pu[60][0] * self.list_r_x_pu[60][0] +
                                             self.list_r_x_pu[60][1] * self.list_r_x_pu[60][1]) * l_ij[
                                     t, 0, 60]) == 0)

                model.addConstr((v_i[t, 0, 61] - v_i[t, 0, 62] - 2 * (
                            self.list_r_x_pu[61][0] * P_ij[t, 0, 61] + self.list_r_x_pu[61][1] * Q_ij[t, 0, 61]) + (
                                             self.list_r_x_pu[61][0] * self.list_r_x_pu[61][0] +
                                             self.list_r_x_pu[61][1] * self.list_r_x_pu[61][1]) * l_ij[
                                     t, 0, 61]) == 0)

                model.addConstr((v_i[t, 0, 1] - v_i[t, 0, 63] - 2 * (
                            self.list_r_x_pu[62][0] * P_ij[t, 0, 62] + self.list_r_x_pu[62][1] * Q_ij[t, 0, 62]) + (
                                             self.list_r_x_pu[62][0] * self.list_r_x_pu[62][0] +
                                             self.list_r_x_pu[62][1] * self.list_r_x_pu[62][1]) * l_ij[
                                     t, 0, 62]) == 0)

                model.addConstr((v_i[t, 0, 63] - v_i[t, 0, 64] - 2 * (
                            self.list_r_x_pu[63][0] * P_ij[t, 0, 63] + self.list_r_x_pu[63][1] * Q_ij[t, 0, 63]) + (
                                             self.list_r_x_pu[63][0] * self.list_r_x_pu[63][0] +
                                             self.list_r_x_pu[63][1] * self.list_r_x_pu[63][1]) * l_ij[
                                     t, 0, 63]) == 0)

                model.addConstr((v_i[t, 0, 64] - v_i[t, 0, 65] - 2 * (
                            self.list_r_x_pu[64][0] * P_ij[t, 0, 64] + self.list_r_x_pu[64][1] * Q_ij[t, 0, 64]) + (
                                             self.list_r_x_pu[64][0] * self.list_r_x_pu[64][0] +
                                             self.list_r_x_pu[64][1] * self.list_r_x_pu[64][1]) * l_ij[
                                     t, 0, 64]) == 0)

                model.addConstr((v_i[t, 0, 65] - v_i[t, 0, 66] - 2 * (
                            self.list_r_x_pu[65][0] * P_ij[t, 0, 65] + self.list_r_x_pu[65][1] * Q_ij[t, 0, 65]) + (
                                             self.list_r_x_pu[65][0] * self.list_r_x_pu[65][0] +
                                             self.list_r_x_pu[65][1] * self.list_r_x_pu[65][1]) * l_ij[
                                     t, 0, 65]) == 0)

                model.addConstr((v_i[t, 0, 66] - v_i[t, 0, 67] - 2 * (
                            self.list_r_x_pu[66][0] * P_ij[t, 0, 66] + self.list_r_x_pu[66][1] * Q_ij[t, 0, 66]) + (
                                             self.list_r_x_pu[66][0] * self.list_r_x_pu[66][0] +
                                             self.list_r_x_pu[66][1] * self.list_r_x_pu[66][1]) * l_ij[
                                     t, 0, 66]) == 0)

                model.addConstr((v_i[t, 0, 67] - v_i[t, 0, 68] - 2 * (
                            self.list_r_x_pu[67][0] * P_ij[t, 0, 67] + self.list_r_x_pu[67][1] * Q_ij[t, 0, 67]) + (
                                             self.list_r_x_pu[67][0] * self.list_r_x_pu[67][0] +
                                             self.list_r_x_pu[67][1] * self.list_r_x_pu[67][1]) * l_ij[
                                     t, 0, 67]) == 0)

                model.addConstr((v_i[t, 0, 68] - v_i[t, 0, 69] - 2 * (
                            self.list_r_x_pu[68][0] * P_ij[t, 0, 68] + self.list_r_x_pu[68][1] * Q_ij[t, 0, 68]) + (
                                             self.list_r_x_pu[68][0] * self.list_r_x_pu[68][0] +
                                             self.list_r_x_pu[68][1] * self.list_r_x_pu[68][1]) * l_ij[
                                     t, 0, 68]) == 0)

                model.addConstr((v_i[t, 0, 69] - v_i[t, 0, 70] - 2 * (
                            self.list_r_x_pu[69][0] * P_ij[t, 0, 69] + self.list_r_x_pu[69][1] * Q_ij[t, 0, 69]) + (
                                             self.list_r_x_pu[69][0] * self.list_r_x_pu[69][0] +
                                             self.list_r_x_pu[69][1] * self.list_r_x_pu[69][1]) * l_ij[
                                     t, 0, 69]) == 0)

                model.addConstr((v_i[t, 0, 70] - v_i[t, 0, 71] - 2 * (
                            self.list_r_x_pu[70][0] * P_ij[t, 0, 70] + self.list_r_x_pu[70][1] * Q_ij[t, 0, 70]) + (
                                             self.list_r_x_pu[70][0] * self.list_r_x_pu[70][0] +
                                             self.list_r_x_pu[70][1] * self.list_r_x_pu[70][1]) * l_ij[
                                     t, 0, 70]) == 0)

                model.addConstr((v_i[t, 0, 71] - v_i[t, 0, 72] - 2 * (
                            self.list_r_x_pu[71][0] * P_ij[t, 0, 71] + self.list_r_x_pu[71][1] * Q_ij[t, 0, 71]) + (
                                             self.list_r_x_pu[71][0] * self.list_r_x_pu[71][0] +
                                             self.list_r_x_pu[71][1] * self.list_r_x_pu[71][1]) * l_ij[
                                     t, 0, 71]) == 0)

                model.addConstr((v_i[t, 0, 72] - v_i[t, 0, 73] - 2 * (
                            self.list_r_x_pu[72][0] * P_ij[t, 0, 72] + self.list_r_x_pu[72][1] * Q_ij[t, 0, 72]) + (
                                             self.list_r_x_pu[72][0] * self.list_r_x_pu[72][0] +
                                             self.list_r_x_pu[72][1] * self.list_r_x_pu[72][1]) * l_ij[
                                     t, 0, 72]) == 0)

                model.addConstr((v_i[t, 0, 73] - v_i[t, 0, 74] - 2 * (
                            self.list_r_x_pu[73][0] * P_ij[t, 0, 73] + self.list_r_x_pu[73][1] * Q_ij[t, 0, 73]) + (
                                             self.list_r_x_pu[73][0] * self.list_r_x_pu[73][0] +
                                             self.list_r_x_pu[73][1] * self.list_r_x_pu[73][1]) * l_ij[
                                     t, 0, 73]) == 0)

                model.addConstr((v_i[t, 0, 74] - v_i[t, 0, 75] - 2 * (
                            self.list_r_x_pu[74][0] * P_ij[t, 0, 74] + self.list_r_x_pu[74][1] * Q_ij[t, 0, 74]) + (
                                             self.list_r_x_pu[74][0] * self.list_r_x_pu[74][0] +
                                             self.list_r_x_pu[74][1] * self.list_r_x_pu[74][1]) * l_ij[
                                     t, 0, 74]) == 0)

                model.addConstr((v_i[t, 0, 75] - v_i[t, 0, 76] - 2 * (
                            self.list_r_x_pu[75][0] * P_ij[t, 0, 75] + self.list_r_x_pu[75][1] * Q_ij[t, 0, 75]) + (
                                             self.list_r_x_pu[75][0] * self.list_r_x_pu[75][0] +
                                             self.list_r_x_pu[75][1] * self.list_r_x_pu[75][1]) * l_ij[
                                     t, 0, 75]) == 0)

                model.addConstr((v_i[t, 0, 76] - v_i[t, 0, 77] - 2 * (
                            self.list_r_x_pu[76][0] * P_ij[t, 0, 76] + self.list_r_x_pu[76][1] * Q_ij[t, 0, 76]) + (
                                             self.list_r_x_pu[76][0] * self.list_r_x_pu[76][0] +
                                             self.list_r_x_pu[76][1] * self.list_r_x_pu[76][1]) * l_ij[
                                     t, 0, 76]) == 0)

                model.addConstr((v_i[t, 0, 64] - v_i[t, 0, 78] - 2 * (
                            self.list_r_x_pu[77][0] * P_ij[t, 0, 77] + self.list_r_x_pu[77][1] * Q_ij[t, 0, 77]) + (
                                             self.list_r_x_pu[77][0] * self.list_r_x_pu[77][0] +
                                             self.list_r_x_pu[77][1] * self.list_r_x_pu[77][1]) * l_ij[
                                     t, 0, 77]) == 0)

                model.addConstr((v_i[t, 0, 78] - v_i[t, 0, 79] - 2 * (
                            self.list_r_x_pu[78][0] * P_ij[t, 0, 78] + self.list_r_x_pu[78][1] * Q_ij[t, 0, 78]) + (
                                             self.list_r_x_pu[78][0] * self.list_r_x_pu[78][0] +
                                             self.list_r_x_pu[78][1] * self.list_r_x_pu[78][1]) * l_ij[
                                     t, 0, 78]) == 0)

                model.addConstr((v_i[t, 0, 79] - v_i[t, 0, 80] - 2 * (
                            self.list_r_x_pu[79][0] * P_ij[t, 0, 79] + self.list_r_x_pu[79][1] * Q_ij[t, 0, 79]) + (
                                             self.list_r_x_pu[79][0] * self.list_r_x_pu[79][0] +
                                             self.list_r_x_pu[79][1] * self.list_r_x_pu[79][1]) * l_ij[
                                     t, 0, 79]) == 0)

                model.addConstr((v_i[t, 0, 80] - v_i[t, 0, 81] - 2 * (
                            self.list_r_x_pu[80][0] * P_ij[t, 0, 80] + self.list_r_x_pu[80][1] * Q_ij[t, 0, 80]) + (
                                             self.list_r_x_pu[80][0] * self.list_r_x_pu[80][0] +
                                             self.list_r_x_pu[80][1] * self.list_r_x_pu[80][1]) * l_ij[
                                     t, 0, 80]) == 0)

                model.addConstr((v_i[t, 0, 81] - v_i[t, 0, 82] - 2 * (
                            self.list_r_x_pu[81][0] * P_ij[t, 0, 81] + self.list_r_x_pu[81][1] * Q_ij[t, 0, 81]) + (
                                             self.list_r_x_pu[81][0] * self.list_r_x_pu[81][0] +
                                             self.list_r_x_pu[81][1] * self.list_r_x_pu[81][1]) * l_ij[
                                     t, 0, 81]) == 0)

                model.addConstr((v_i[t, 0, 82] - v_i[t, 0, 83] - 2 * (
                            self.list_r_x_pu[82][0] * P_ij[t, 0, 82] + self.list_r_x_pu[82][1] * Q_ij[t, 0, 82]) + (
                                             self.list_r_x_pu[82][0] * self.list_r_x_pu[82][0] +
                                             self.list_r_x_pu[82][1] * self.list_r_x_pu[82][1]) * l_ij[
                                     t, 0, 82]) == 0)

                model.addConstr((v_i[t, 0, 83] - v_i[t, 0, 84] - 2 * (
                            self.list_r_x_pu[83][0] * P_ij[t, 0, 83] + self.list_r_x_pu[83][1] * Q_ij[t, 0, 83]) + (
                                             self.list_r_x_pu[83][0] * self.list_r_x_pu[83][0] +
                                             self.list_r_x_pu[83][1] * self.list_r_x_pu[83][1]) * l_ij[
                                     t, 0, 83]) == 0)

                model.addConstr((v_i[t, 0, 84] - v_i[t, 0, 85] - 2 * (
                            self.list_r_x_pu[84][0] * P_ij[t, 0, 84] + self.list_r_x_pu[84][1] * Q_ij[t, 0, 84]) + (
                                             self.list_r_x_pu[84][0] * self.list_r_x_pu[84][0] +
                                             self.list_r_x_pu[84][1] * self.list_r_x_pu[84][1]) * l_ij[
                                     t, 0, 84]) == 0)

                model.addConstr((v_i[t, 0, 79] - v_i[t, 0, 86] - 2 * (
                            self.list_r_x_pu[85][0] * P_ij[t, 0, 85] + self.list_r_x_pu[85][1] * Q_ij[t, 0, 85]) + (
                                             self.list_r_x_pu[85][0] * self.list_r_x_pu[85][0] +
                                             self.list_r_x_pu[85][1] * self.list_r_x_pu[85][1]) * l_ij[
                                     t, 0, 85]) == 0)

                model.addConstr((v_i[t, 0, 86] - v_i[t, 0, 87] - 2 * (
                            self.list_r_x_pu[86][0] * P_ij[t, 0, 86] + self.list_r_x_pu[86][1] * Q_ij[t, 0, 86]) + (
                                             self.list_r_x_pu[86][0] * self.list_r_x_pu[86][0] +
                                             self.list_r_x_pu[86][1] * self.list_r_x_pu[86][1]) * l_ij[
                                     t, 0, 86]) == 0)

                model.addConstr((v_i[t, 0, 87] - v_i[t, 0, 88] - 2 * (
                            self.list_r_x_pu[87][0] * P_ij[t, 0, 87] + self.list_r_x_pu[87][1] * Q_ij[t, 0, 87]) + (
                                             self.list_r_x_pu[87][0] * self.list_r_x_pu[87][0] +
                                             self.list_r_x_pu[87][1] * self.list_r_x_pu[87][1]) * l_ij[
                                     t, 0, 87]) == 0)

                model.addConstr((v_i[t, 0, 65] - v_i[t, 0, 89] - 2 * (
                            self.list_r_x_pu[88][0] * P_ij[t, 0, 88] + self.list_r_x_pu[88][1] * Q_ij[t, 0, 88]) + (
                                             self.list_r_x_pu[88][0] * self.list_r_x_pu[88][0] +
                                             self.list_r_x_pu[88][1] * self.list_r_x_pu[88][1]) * l_ij[
                                     t, 0, 88]) == 0)

                model.addConstr((v_i[t, 0, 89] - v_i[t, 0, 90] - 2 * (
                            self.list_r_x_pu[89][0] * P_ij[t, 0, 89] + self.list_r_x_pu[89][1] * Q_ij[t, 0, 89]) + (
                                             self.list_r_x_pu[89][0] * self.list_r_x_pu[89][0] +
                                             self.list_r_x_pu[89][1] * self.list_r_x_pu[89][1]) * l_ij[
                                     t, 0, 89]) == 0)

                model.addConstr((v_i[t, 0, 90] - v_i[t, 0, 91] - 2 * (
                            self.list_r_x_pu[90][0] * P_ij[t, 0, 90] + self.list_r_x_pu[90][1] * Q_ij[t, 0, 90]) + (
                                             self.list_r_x_pu[90][0] * self.list_r_x_pu[90][0] +
                                             self.list_r_x_pu[90][1] * self.list_r_x_pu[90][1]) * l_ij[
                                     t, 0, 90]) == 0)

                model.addConstr((v_i[t, 0, 91] - v_i[t, 0, 92] - 2 * (
                            self.list_r_x_pu[91][0] * P_ij[t, 0, 91] + self.list_r_x_pu[91][1] * Q_ij[t, 0, 91]) + (
                                             self.list_r_x_pu[91][0] * self.list_r_x_pu[91][0] +
                                             self.list_r_x_pu[91][1] * self.list_r_x_pu[91][1]) * l_ij[
                                     t, 0, 91]) == 0)

                model.addConstr((v_i[t, 0, 92] - v_i[t, 0, 93] - 2 * (
                            self.list_r_x_pu[92][0] * P_ij[t, 0, 92] + self.list_r_x_pu[92][1] * Q_ij[t, 0, 92]) + (
                                             self.list_r_x_pu[92][0] * self.list_r_x_pu[92][0] +
                                             self.list_r_x_pu[92][1] * self.list_r_x_pu[92][1]) * l_ij[
                                     t, 0, 92]) == 0)

                model.addConstr((v_i[t, 0, 93] - v_i[t, 0, 94] - 2 * (
                            self.list_r_x_pu[93][0] * P_ij[t, 0, 93] + self.list_r_x_pu[93][1] * Q_ij[t, 0, 93]) + (
                                             self.list_r_x_pu[93][0] * self.list_r_x_pu[93][0] +
                                             self.list_r_x_pu[93][1] * self.list_r_x_pu[93][1]) * l_ij[
                                     t, 0, 93]) == 0)

                model.addConstr((v_i[t, 0, 94] - v_i[t, 0, 95] - 2 * (
                            self.list_r_x_pu[94][0] * P_ij[t, 0, 94] + self.list_r_x_pu[94][1] * Q_ij[t, 0, 94]) + (
                                             self.list_r_x_pu[94][0] * self.list_r_x_pu[94][0] +
                                             self.list_r_x_pu[94][1] * self.list_r_x_pu[94][1]) * l_ij[
                                     t, 0, 94]) == 0)

                model.addConstr((v_i[t, 0, 91] - v_i[t, 0, 96] - 2 * (
                            self.list_r_x_pu[95][0] * P_ij[t, 0, 95] + self.list_r_x_pu[95][1] * Q_ij[t, 0, 95]) + (
                                             self.list_r_x_pu[95][0] * self.list_r_x_pu[95][0] +
                                             self.list_r_x_pu[95][1] * self.list_r_x_pu[95][1]) * l_ij[
                                     t, 0, 95]) == 0)

                model.addConstr((v_i[t, 0, 96] - v_i[t, 0, 97] - 2 * (
                            self.list_r_x_pu[96][0] * P_ij[t, 0, 96] + self.list_r_x_pu[96][1] * Q_ij[t, 0, 96]) + (
                                             self.list_r_x_pu[96][0] * self.list_r_x_pu[96][0] +
                                             self.list_r_x_pu[96][1] * self.list_r_x_pu[96][1]) * l_ij[
                                     t, 0, 96]) == 0)

                model.addConstr((v_i[t, 0, 97] - v_i[t, 0, 98] - 2 * (
                            self.list_r_x_pu[97][0] * P_ij[t, 0, 97] + self.list_r_x_pu[97][1] * Q_ij[t, 0, 97]) + (
                                             self.list_r_x_pu[97][0] * self.list_r_x_pu[97][0] +
                                             self.list_r_x_pu[97][1] * self.list_r_x_pu[97][1]) * l_ij[
                                     t, 0, 97]) == 0)

                model.addConstr((v_i[t, 0, 98] - v_i[t, 0, 99] - 2 * (
                            self.list_r_x_pu[98][0] * P_ij[t, 0, 98] + self.list_r_x_pu[98][1] * Q_ij[t, 0, 98]) + (
                                             self.list_r_x_pu[98][0] * self.list_r_x_pu[98][0] +
                                             self.list_r_x_pu[98][1] * self.list_r_x_pu[98][1]) * l_ij[
                                     t, 0, 98]) == 0)

                model.addConstr((v_i[t, 0, 1] - v_i[t, 0, 100] - 2 * (
                            self.list_r_x_pu[99][0] * P_ij[t, 0, 99] + self.list_r_x_pu[99][1] * Q_ij[t, 0, 99]) + (
                                             self.list_r_x_pu[99][0] * self.list_r_x_pu[99][0] +
                                             self.list_r_x_pu[99][1] * self.list_r_x_pu[99][1]) * l_ij[
                                     t, 0, 99]) == 0)

                model.addConstr((v_i[t, 0, 100] - v_i[t, 0, 101] - 2 * (
                            self.list_r_x_pu[100][0] * P_ij[t, 0, 100] + self.list_r_x_pu[100][1] * Q_ij[
                        t, 0, 100]) + (self.list_r_x_pu[100][0] * self.list_r_x_pu[100][0] + self.list_r_x_pu[100][
                    1] * self.list_r_x_pu[100][1]) * l_ij[t, 0, 100]) == 0)

                model.addConstr((v_i[t, 0, 101] - v_i[t, 0, 102] - 2 * (
                            self.list_r_x_pu[101][0] * P_ij[t, 0, 101] + self.list_r_x_pu[101][1] * Q_ij[
                        t, 0, 101]) + (self.list_r_x_pu[101][0] * self.list_r_x_pu[101][0] + self.list_r_x_pu[101][
                    1] * self.list_r_x_pu[101][1]) * l_ij[t, 0, 101]) == 0)

                model.addConstr((v_i[t, 0, 102] - v_i[t, 0, 103] - 2 * (
                            self.list_r_x_pu[102][0] * P_ij[t, 0, 102] + self.list_r_x_pu[102][1] * Q_ij[
                        t, 0, 102]) + (self.list_r_x_pu[102][0] * self.list_r_x_pu[102][0] + self.list_r_x_pu[102][
                    1] * self.list_r_x_pu[102][1]) * l_ij[t, 0, 102]) == 0)

                model.addConstr((v_i[t, 0, 103] - v_i[t, 0, 104] - 2 * (
                            self.list_r_x_pu[103][0] * P_ij[t, 0, 103] + self.list_r_x_pu[103][1] * Q_ij[
                        t, 0, 103]) + (self.list_r_x_pu[103][0] * self.list_r_x_pu[103][0] + self.list_r_x_pu[103][
                    1] * self.list_r_x_pu[103][1]) * l_ij[t, 0, 103]) == 0)

                model.addConstr((v_i[t, 0, 104] - v_i[t, 0, 105] - 2 * (
                            self.list_r_x_pu[104][0] * P_ij[t, 0, 104] + self.list_r_x_pu[104][1] * Q_ij[
                        t, 0, 104]) + (self.list_r_x_pu[104][0] * self.list_r_x_pu[104][0] + self.list_r_x_pu[104][
                    1] * self.list_r_x_pu[104][1]) * l_ij[t, 0, 104]) == 0)

                model.addConstr((v_i[t, 0, 105] - v_i[t, 0, 106] - 2 * (
                            self.list_r_x_pu[105][0] * P_ij[t, 0, 105] + self.list_r_x_pu[105][1] * Q_ij[
                        t, 0, 105]) + (self.list_r_x_pu[105][0] * self.list_r_x_pu[105][0] + self.list_r_x_pu[105][
                    1] * self.list_r_x_pu[105][1]) * l_ij[t, 0, 105]) == 0)

                model.addConstr((v_i[t, 0, 106] - v_i[t, 0, 107] - 2 * (
                            self.list_r_x_pu[106][0] * P_ij[t, 0, 106] + self.list_r_x_pu[106][1] * Q_ij[
                        t, 0, 106]) + (self.list_r_x_pu[106][0] * self.list_r_x_pu[106][0] + self.list_r_x_pu[106][
                    1] * self.list_r_x_pu[106][1]) * l_ij[t, 0, 106]) == 0)

                model.addConstr((v_i[t, 0, 107] - v_i[t, 0, 108] - 2 * (
                            self.list_r_x_pu[107][0] * P_ij[t, 0, 107] + self.list_r_x_pu[107][1] * Q_ij[
                        t, 0, 107]) + (self.list_r_x_pu[107][0] * self.list_r_x_pu[107][0] + self.list_r_x_pu[107][
                    1] * self.list_r_x_pu[107][1]) * l_ij[t, 0, 107]) == 0)

                model.addConstr((v_i[t, 0, 108] - v_i[t, 0, 109] - 2 * (
                            self.list_r_x_pu[108][0] * P_ij[t, 0, 108] + self.list_r_x_pu[108][1] * Q_ij[
                        t, 0, 108]) + (self.list_r_x_pu[108][0] * self.list_r_x_pu[108][0] + self.list_r_x_pu[108][
                    1] * self.list_r_x_pu[108][1]) * l_ij[t, 0, 108]) == 0)

                model.addConstr((v_i[t, 0, 109] - v_i[t, 0, 110] - 2 * (
                            self.list_r_x_pu[109][0] * P_ij[t, 0, 109] + self.list_r_x_pu[109][1] * Q_ij[
                        t, 0, 109]) + (self.list_r_x_pu[109][0] * self.list_r_x_pu[109][0] + self.list_r_x_pu[109][
                    1] * self.list_r_x_pu[109][1]) * l_ij[t, 0, 109]) == 0)

                model.addConstr((v_i[t, 0, 110] - v_i[t, 0, 111] - 2 * (
                            self.list_r_x_pu[110][0] * P_ij[t, 0, 110] + self.list_r_x_pu[110][1] * Q_ij[
                        t, 0, 110]) + (self.list_r_x_pu[110][0] * self.list_r_x_pu[110][0] + self.list_r_x_pu[110][
                    1] * self.list_r_x_pu[110][1]) * l_ij[t, 0, 110]) == 0)

                model.addConstr((v_i[t, 0, 110] - v_i[t, 0, 112] - 2 * (
                            self.list_r_x_pu[111][0] * P_ij[t, 0, 111] + self.list_r_x_pu[111][1] * Q_ij[
                        t, 0, 111]) + (self.list_r_x_pu[111][0] * self.list_r_x_pu[111][0] + self.list_r_x_pu[111][
                    1] * self.list_r_x_pu[111][1]) * l_ij[t, 0, 111]) == 0)

                model.addConstr((v_i[t, 0, 112] - v_i[t, 0, 113] - 2 * (
                            self.list_r_x_pu[112][0] * P_ij[t, 0, 112] + self.list_r_x_pu[112][1] * Q_ij[
                        t, 0, 112]) + (self.list_r_x_pu[112][0] * self.list_r_x_pu[112][0] + self.list_r_x_pu[112][
                    1] * self.list_r_x_pu[112][1]) * l_ij[t, 0, 112]) == 0)

                model.addConstr((v_i[t, 0, 100] - v_i[t, 0, 114] - 2 * (
                            self.list_r_x_pu[113][0] * P_ij[t, 0, 113] + self.list_r_x_pu[113][1] * Q_ij[
                        t, 0, 113]) + (self.list_r_x_pu[113][0] * self.list_r_x_pu[113][0] + self.list_r_x_pu[113][
                    1] * self.list_r_x_pu[113][1]) * l_ij[t, 0, 113]) == 0)

                model.addConstr((v_i[t, 0, 114] - v_i[t, 0, 115] - 2 * (
                            self.list_r_x_pu[114][0] * P_ij[t, 0, 114] + self.list_r_x_pu[114][1] * Q_ij[
                        t, 0, 114]) + (self.list_r_x_pu[114][0] * self.list_r_x_pu[114][0] + self.list_r_x_pu[114][
                    1] * self.list_r_x_pu[114][1]) * l_ij[t, 0, 114]) == 0)

                model.addConstr((v_i[t, 0, 115] - v_i[t, 0, 116] - 2 * (
                            self.list_r_x_pu[115][0] * P_ij[t, 0, 115] + self.list_r_x_pu[115][1] * Q_ij[
                        t, 0, 115]) + (self.list_r_x_pu[115][0] * self.list_r_x_pu[115][0] + self.list_r_x_pu[115][
                    1] * self.list_r_x_pu[115][1]) * l_ij[t, 0, 115]) == 0)

                model.addConstr((v_i[t, 0, 116] - v_i[t, 0, 117] - 2 * (
                            self.list_r_x_pu[116][0] * P_ij[t, 0, 116] + self.list_r_x_pu[116][1] * Q_ij[
                        t, 0, 116]) + (self.list_r_x_pu[116][0] * self.list_r_x_pu[116][0] + self.list_r_x_pu[116][
                    1] * self.list_r_x_pu[116][1]) * l_ij[t, 0, 116]) == 0)

                model.addConstr((v_i[t, 0, 117] - v_i[t, 0, 118] - 2 * (
                            self.list_r_x_pu[117][0] * P_ij[t, 0, 117] + self.list_r_x_pu[117][1] * Q_ij[
                        t, 0, 117]) + (self.list_r_x_pu[117][0] * self.list_r_x_pu[117][0] + self.list_r_x_pu[117][
                    1] * self.list_r_x_pu[117][1]) * l_ij[t, 0, 117]) == 0)

                model.addConstr(lv_ij_i[t, 0, 0] == l_ij[t, 0, 0] * v_i[t, 0, 0])
                model.addConstr((lv_ij_i[t, 0, 0] - PP_ij[t, 0, 0] - QQ_ij[t, 0, 0]) == 0)

                model.addConstr(lv_ij_i[t, 0, 1] == l_ij[t, 0, 1] * v_i[t, 0, 1])
                model.addConstr((lv_ij_i[t, 0, 1] - PP_ij[t, 0, 1] - QQ_ij[t, 0, 1]) == 0)

                model.addConstr(lv_ij_i[t, 0, 2] == l_ij[t, 0, 2] * v_i[t, 0, 2])
                model.addConstr((lv_ij_i[t, 0, 2] - PP_ij[t, 0, 2] - QQ_ij[t, 0, 2]) == 0)

                model.addConstr(lv_ij_i[t, 0, 3] == l_ij[t, 0, 3] * v_i[t, 0, 2])
                model.addConstr((lv_ij_i[t, 0, 3] - PP_ij[t, 0, 3] - QQ_ij[t, 0, 3]) == 0)

                model.addConstr(lv_ij_i[t, 0, 4] == l_ij[t, 0, 4] * v_i[t, 0, 4])
                model.addConstr((lv_ij_i[t, 0, 4] - PP_ij[t, 0, 4] - QQ_ij[t, 0, 4]) == 0)

                model.addConstr(lv_ij_i[t, 0, 5] == l_ij[t, 0, 5] * v_i[t, 0, 5])
                model.addConstr((lv_ij_i[t, 0, 5] - PP_ij[t, 0, 5] - QQ_ij[t, 0, 5]) == 0)

                model.addConstr(lv_ij_i[t, 0, 6] == l_ij[t, 0, 6] * v_i[t, 0, 6])
                model.addConstr((lv_ij_i[t, 0, 6] - PP_ij[t, 0, 6] - QQ_ij[t, 0, 6]) == 0)

                model.addConstr(lv_ij_i[t, 0, 7] == l_ij[t, 0, 7] * v_i[t, 0, 7])
                model.addConstr((lv_ij_i[t, 0, 7] - PP_ij[t, 0, 7] - QQ_ij[t, 0, 7]) == 0)

                model.addConstr(lv_ij_i[t, 0, 8] == l_ij[t, 0, 8] * v_i[t, 0, 8])
                model.addConstr((lv_ij_i[t, 0, 8] - PP_ij[t, 0, 8] - QQ_ij[t, 0, 8]) == 0)

                model.addConstr(lv_ij_i[t, 0, 9] == l_ij[t, 0, 9] * v_i[t, 0, 2])
                model.addConstr((lv_ij_i[t, 0, 9] - PP_ij[t, 0, 9] - QQ_ij[t, 0, 9]) == 0)

                model.addConstr(lv_ij_i[t, 0, 10] == l_ij[t, 0, 10] * v_i[t, 0, 10])
                model.addConstr((lv_ij_i[t, 0, 10] - PP_ij[t, 0, 10] - QQ_ij[t, 0, 10]) == 0)

                model.addConstr(lv_ij_i[t, 0, 11] == l_ij[t, 0, 11] * v_i[t, 0, 11])
                model.addConstr((lv_ij_i[t, 0, 11] - PP_ij[t, 0, 11] - QQ_ij[t, 0, 11]) == 0)

                model.addConstr(lv_ij_i[t, 0, 12] == l_ij[t, 0, 12] * v_i[t, 0, 12])
                model.addConstr((lv_ij_i[t, 0, 12] - PP_ij[t, 0, 12] - QQ_ij[t, 0, 12]) == 0)

                model.addConstr(lv_ij_i[t, 0, 13] == l_ij[t, 0, 13] * v_i[t, 0, 13])
                model.addConstr((lv_ij_i[t, 0, 13] - PP_ij[t, 0, 13] - QQ_ij[t, 0, 13]) == 0)

                model.addConstr(lv_ij_i[t, 0, 14] == l_ij[t, 0, 14] * v_i[t, 0, 14])
                model.addConstr((lv_ij_i[t, 0, 14] - PP_ij[t, 0, 14] - QQ_ij[t, 0, 14]) == 0)

                model.addConstr(lv_ij_i[t, 0, 15] == l_ij[t, 0, 15] * v_i[t, 0, 15])
                model.addConstr((lv_ij_i[t, 0, 15] - PP_ij[t, 0, 15] - QQ_ij[t, 0, 15]) == 0)

                model.addConstr(lv_ij_i[t, 0, 16] == l_ij[t, 0, 16] * v_i[t, 0, 16])
                model.addConstr((lv_ij_i[t, 0, 16] - PP_ij[t, 0, 16] - QQ_ij[t, 0, 16]) == 0)

                model.addConstr(lv_ij_i[t, 0, 17] == l_ij[t, 0, 17] * v_i[t, 0, 11])
                model.addConstr((lv_ij_i[t, 0, 17] - PP_ij[t, 0, 17] - QQ_ij[t, 0, 17]) == 0)

                model.addConstr(lv_ij_i[t, 0, 18] == l_ij[t, 0, 18] * v_i[t, 0, 18])
                model.addConstr((lv_ij_i[t, 0, 18] - PP_ij[t, 0, 18] - QQ_ij[t, 0, 18]) == 0)

                model.addConstr(lv_ij_i[t, 0, 19] == l_ij[t, 0, 19] * v_i[t, 0, 19])
                model.addConstr((lv_ij_i[t, 0, 19] - PP_ij[t, 0, 19] - QQ_ij[t, 0, 19]) == 0)

                model.addConstr(lv_ij_i[t, 0, 20] == l_ij[t, 0, 20] * v_i[t, 0, 20])
                model.addConstr((lv_ij_i[t, 0, 20] - PP_ij[t, 0, 20] - QQ_ij[t, 0, 20]) == 0)

                model.addConstr(lv_ij_i[t, 0, 21] == l_ij[t, 0, 21] * v_i[t, 0, 21])
                model.addConstr((lv_ij_i[t, 0, 21] - PP_ij[t, 0, 21] - QQ_ij[t, 0, 21]) == 0)

                model.addConstr(lv_ij_i[t, 0, 22] == l_ij[t, 0, 22] * v_i[t, 0, 22])
                model.addConstr((lv_ij_i[t, 0, 22] - PP_ij[t, 0, 22] - QQ_ij[t, 0, 22]) == 0)

                model.addConstr(lv_ij_i[t, 0, 23] == l_ij[t, 0, 23] * v_i[t, 0, 23])
                model.addConstr((lv_ij_i[t, 0, 23] - PP_ij[t, 0, 23] - QQ_ij[t, 0, 23]) == 0)

                model.addConstr(lv_ij_i[t, 0, 24] == l_ij[t, 0, 24] * v_i[t, 0, 24])
                model.addConstr((lv_ij_i[t, 0, 24] - PP_ij[t, 0, 24] - QQ_ij[t, 0, 24]) == 0)

                model.addConstr(lv_ij_i[t, 0, 25] == l_ij[t, 0, 25] * v_i[t, 0, 25])
                model.addConstr((lv_ij_i[t, 0, 25] - PP_ij[t, 0, 25] - QQ_ij[t, 0, 25]) == 0)

                model.addConstr(lv_ij_i[t, 0, 26] == l_ij[t, 0, 26] * v_i[t, 0, 26])
                model.addConstr((lv_ij_i[t, 0, 26] - PP_ij[t, 0, 26] - QQ_ij[t, 0, 26]) == 0)

                model.addConstr(lv_ij_i[t, 0, 27] == l_ij[t, 0, 27] * v_i[t, 0, 4])
                model.addConstr((lv_ij_i[t, 0, 27] - PP_ij[t, 0, 27] - QQ_ij[t, 0, 27]) == 0)

                model.addConstr(lv_ij_i[t, 0, 28] == l_ij[t, 0, 28] * v_i[t, 0, 28])
                model.addConstr((lv_ij_i[t, 0, 28] - PP_ij[t, 0, 28] - QQ_ij[t, 0, 28]) == 0)

                model.addConstr(lv_ij_i[t, 0, 29] == l_ij[t, 0, 29] * v_i[t, 0, 29])
                model.addConstr((lv_ij_i[t, 0, 29] - PP_ij[t, 0, 29] - QQ_ij[t, 0, 29]) == 0)

                model.addConstr(lv_ij_i[t, 0, 30] == l_ij[t, 0, 30] * v_i[t, 0, 30])
                model.addConstr((lv_ij_i[t, 0, 30] - PP_ij[t, 0, 30] - QQ_ij[t, 0, 30]) == 0)

                model.addConstr(lv_ij_i[t, 0, 31] == l_ij[t, 0, 31] * v_i[t, 0, 31])
                model.addConstr((lv_ij_i[t, 0, 31] - PP_ij[t, 0, 31] - QQ_ij[t, 0, 31]) == 0)

                model.addConstr(lv_ij_i[t, 0, 32] == l_ij[t, 0, 32] * v_i[t, 0, 32])
                model.addConstr((lv_ij_i[t, 0, 32] - PP_ij[t, 0, 32] - QQ_ij[t, 0, 32]) == 0)

                model.addConstr(lv_ij_i[t, 0, 33] == l_ij[t, 0, 33] * v_i[t, 0, 33])
                model.addConstr((lv_ij_i[t, 0, 33] - PP_ij[t, 0, 33] - QQ_ij[t, 0, 33]) == 0)

                model.addConstr(lv_ij_i[t, 0, 34] == l_ij[t, 0, 34] * v_i[t, 0, 34])
                model.addConstr((lv_ij_i[t, 0, 34] - PP_ij[t, 0, 34] - QQ_ij[t, 0, 34]) == 0)

                model.addConstr(lv_ij_i[t, 0, 35] == l_ij[t, 0, 35] * v_i[t, 0, 30])
                model.addConstr((lv_ij_i[t, 0, 35] - PP_ij[t, 0, 35] - QQ_ij[t, 0, 35]) == 0)

                model.addConstr(lv_ij_i[t, 0, 36] == l_ij[t, 0, 36] * v_i[t, 0, 36])
                model.addConstr((lv_ij_i[t, 0, 36] - PP_ij[t, 0, 36] - QQ_ij[t, 0, 36]) == 0)

                model.addConstr(lv_ij_i[t, 0, 37] == l_ij[t, 0, 37] * v_i[t, 0, 29])
                model.addConstr((lv_ij_i[t, 0, 37] - PP_ij[t, 0, 37] - QQ_ij[t, 0, 37]) == 0)

                model.addConstr(lv_ij_i[t, 0, 38] == l_ij[t, 0, 38] * v_i[t, 0, 38])
                model.addConstr((lv_ij_i[t, 0, 38] - PP_ij[t, 0, 38] - QQ_ij[t, 0, 38]) == 0)

                model.addConstr(lv_ij_i[t, 0, 39] == l_ij[t, 0, 39] * v_i[t, 0, 39])
                model.addConstr((lv_ij_i[t, 0, 39] - PP_ij[t, 0, 39] - QQ_ij[t, 0, 39]) == 0)

                model.addConstr(lv_ij_i[t, 0, 40] == l_ij[t, 0, 40] * v_i[t, 0, 40])
                model.addConstr((lv_ij_i[t, 0, 40] - PP_ij[t, 0, 40] - QQ_ij[t, 0, 40]) == 0)

                model.addConstr(lv_ij_i[t, 0, 41] == l_ij[t, 0, 41] * v_i[t, 0, 41])
                model.addConstr((lv_ij_i[t, 0, 41] - PP_ij[t, 0, 41] - QQ_ij[t, 0, 41]) == 0)

                model.addConstr(lv_ij_i[t, 0, 42] == l_ij[t, 0, 42] * v_i[t, 0, 42])
                model.addConstr((lv_ij_i[t, 0, 42] - PP_ij[t, 0, 42] - QQ_ij[t, 0, 42]) == 0)

                model.addConstr(lv_ij_i[t, 0, 43] == l_ij[t, 0, 43] * v_i[t, 0, 43])
                model.addConstr((lv_ij_i[t, 0, 43] - PP_ij[t, 0, 43] - QQ_ij[t, 0, 43]) == 0)

                model.addConstr(lv_ij_i[t, 0, 44] == l_ij[t, 0, 44] * v_i[t, 0, 44])
                model.addConstr((lv_ij_i[t, 0, 44] - PP_ij[t, 0, 44] - QQ_ij[t, 0, 44]) == 0)

                model.addConstr(lv_ij_i[t, 0, 45] == l_ij[t, 0, 45] * v_i[t, 0, 45])
                model.addConstr((lv_ij_i[t, 0, 45] - PP_ij[t, 0, 45] - QQ_ij[t, 0, 45]) == 0)

                model.addConstr(lv_ij_i[t, 0, 46] == l_ij[t, 0, 46] * v_i[t, 0, 35])
                model.addConstr((lv_ij_i[t, 0, 46] - PP_ij[t, 0, 46] - QQ_ij[t, 0, 46]) == 0)

                model.addConstr(lv_ij_i[t, 0, 47] == l_ij[t, 0, 47] * v_i[t, 0, 47])
                model.addConstr((lv_ij_i[t, 0, 47] - PP_ij[t, 0, 47] - QQ_ij[t, 0, 47]) == 0)

                model.addConstr(lv_ij_i[t, 0, 48] == l_ij[t, 0, 48] * v_i[t, 0, 48])
                model.addConstr((lv_ij_i[t, 0, 48] - PP_ij[t, 0, 48] - QQ_ij[t, 0, 48]) == 0)

                model.addConstr(lv_ij_i[t, 0, 49] == l_ij[t, 0, 49] * v_i[t, 0, 49])
                model.addConstr((lv_ij_i[t, 0, 49] - PP_ij[t, 0, 49] - QQ_ij[t, 0, 49]) == 0)

                model.addConstr(lv_ij_i[t, 0, 50] == l_ij[t, 0, 50] * v_i[t, 0, 50])
                model.addConstr((lv_ij_i[t, 0, 50] - PP_ij[t, 0, 50] - QQ_ij[t, 0, 50]) == 0)

                model.addConstr(lv_ij_i[t, 0, 51] == l_ij[t, 0, 51] * v_i[t, 0, 51])
                model.addConstr((lv_ij_i[t, 0, 51] - PP_ij[t, 0, 51] - QQ_ij[t, 0, 51]) == 0)

                model.addConstr(lv_ij_i[t, 0, 52] == l_ij[t, 0, 52] * v_i[t, 0, 52])
                model.addConstr((lv_ij_i[t, 0, 52] - PP_ij[t, 0, 52] - QQ_ij[t, 0, 52]) == 0)

                model.addConstr(lv_ij_i[t, 0, 53] == l_ij[t, 0, 53] * v_i[t, 0, 53])
                model.addConstr((lv_ij_i[t, 0, 53] - PP_ij[t, 0, 53] - QQ_ij[t, 0, 53]) == 0)

                model.addConstr(lv_ij_i[t, 0, 54] == l_ij[t, 0, 54] * v_i[t, 0, 29])
                model.addConstr((lv_ij_i[t, 0, 54] - PP_ij[t, 0, 54] - QQ_ij[t, 0, 54]) == 0)

                model.addConstr(lv_ij_i[t, 0, 55] == l_ij[t, 0, 55] * v_i[t, 0, 55])
                model.addConstr((lv_ij_i[t, 0, 55] - PP_ij[t, 0, 55] - QQ_ij[t, 0, 55]) == 0)

                model.addConstr(lv_ij_i[t, 0, 56] == l_ij[t, 0, 56] * v_i[t, 0, 56])
                model.addConstr((lv_ij_i[t, 0, 56] - PP_ij[t, 0, 56] - QQ_ij[t, 0, 56]) == 0)

                model.addConstr(lv_ij_i[t, 0, 57] == l_ij[t, 0, 57] * v_i[t, 0, 57])
                model.addConstr((lv_ij_i[t, 0, 57] - PP_ij[t, 0, 57] - QQ_ij[t, 0, 57]) == 0)

                model.addConstr(lv_ij_i[t, 0, 58] == l_ij[t, 0, 58] * v_i[t, 0, 58])
                model.addConstr((lv_ij_i[t, 0, 58] - PP_ij[t, 0, 58] - QQ_ij[t, 0, 58]) == 0)

                model.addConstr(lv_ij_i[t, 0, 59] == l_ij[t, 0, 59] * v_i[t, 0, 59])
                model.addConstr((lv_ij_i[t, 0, 59] - PP_ij[t, 0, 59] - QQ_ij[t, 0, 59]) == 0)

                model.addConstr(lv_ij_i[t, 0, 60] == l_ij[t, 0, 60] * v_i[t, 0, 60])
                model.addConstr((lv_ij_i[t, 0, 60] - PP_ij[t, 0, 60] - QQ_ij[t, 0, 60]) == 0)

                model.addConstr(lv_ij_i[t, 0, 61] == l_ij[t, 0, 61] * v_i[t, 0, 61])
                model.addConstr((lv_ij_i[t, 0, 61] - PP_ij[t, 0, 61] - QQ_ij[t, 0, 61]) == 0)

                model.addConstr(lv_ij_i[t, 0, 62] == l_ij[t, 0, 62] * v_i[t, 0, 1])
                model.addConstr((lv_ij_i[t, 0, 62] - PP_ij[t, 0, 62] - QQ_ij[t, 0, 62]) == 0)

                model.addConstr(lv_ij_i[t, 0, 63] == l_ij[t, 0, 63] * v_i[t, 0, 63])
                model.addConstr((lv_ij_i[t, 0, 63] - PP_ij[t, 0, 63] - QQ_ij[t, 0, 63]) == 0)

                model.addConstr(lv_ij_i[t, 0, 64] == l_ij[t, 0, 64] * v_i[t, 0, 64])
                model.addConstr((lv_ij_i[t, 0, 64] - PP_ij[t, 0, 64] - QQ_ij[t, 0, 64]) == 0)

                model.addConstr(lv_ij_i[t, 0, 65] == l_ij[t, 0, 65] * v_i[t, 0, 65])
                model.addConstr((lv_ij_i[t, 0, 65] - PP_ij[t, 0, 65] - QQ_ij[t, 0, 65]) == 0)

                model.addConstr(lv_ij_i[t, 0, 66] == l_ij[t, 0, 66] * v_i[t, 0, 66])
                model.addConstr((lv_ij_i[t, 0, 66] - PP_ij[t, 0, 66] - QQ_ij[t, 0, 66]) == 0)

                model.addConstr(lv_ij_i[t, 0, 67] == l_ij[t, 0, 67] * v_i[t, 0, 67])
                model.addConstr((lv_ij_i[t, 0, 67] - PP_ij[t, 0, 67] - QQ_ij[t, 0, 67]) == 0)

                model.addConstr(lv_ij_i[t, 0, 68] == l_ij[t, 0, 68] * v_i[t, 0, 68])
                model.addConstr((lv_ij_i[t, 0, 68] - PP_ij[t, 0, 68] - QQ_ij[t, 0, 68]) == 0)

                model.addConstr(lv_ij_i[t, 0, 69] == l_ij[t, 0, 69] * v_i[t, 0, 69])
                model.addConstr((lv_ij_i[t, 0, 69] - PP_ij[t, 0, 69] - QQ_ij[t, 0, 69]) == 0)

                model.addConstr(lv_ij_i[t, 0, 70] == l_ij[t, 0, 70] * v_i[t, 0, 70])
                model.addConstr((lv_ij_i[t, 0, 70] - PP_ij[t, 0, 70] - QQ_ij[t, 0, 70]) == 0)

                model.addConstr(lv_ij_i[t, 0, 71] == l_ij[t, 0, 71] * v_i[t, 0, 71])
                model.addConstr((lv_ij_i[t, 0, 71] - PP_ij[t, 0, 71] - QQ_ij[t, 0, 71]) == 0)

                model.addConstr(lv_ij_i[t, 0, 72] == l_ij[t, 0, 72] * v_i[t, 0, 72])
                model.addConstr((lv_ij_i[t, 0, 72] - PP_ij[t, 0, 72] - QQ_ij[t, 0, 72]) == 0)

                model.addConstr(lv_ij_i[t, 0, 73] == l_ij[t, 0, 73] * v_i[t, 0, 73])
                model.addConstr((lv_ij_i[t, 0, 73] - PP_ij[t, 0, 73] - QQ_ij[t, 0, 73]) == 0)

                model.addConstr(lv_ij_i[t, 0, 74] == l_ij[t, 0, 74] * v_i[t, 0, 74])
                model.addConstr((lv_ij_i[t, 0, 74] - PP_ij[t, 0, 74] - QQ_ij[t, 0, 74]) == 0)

                model.addConstr(lv_ij_i[t, 0, 75] == l_ij[t, 0, 75] * v_i[t, 0, 75])
                model.addConstr((lv_ij_i[t, 0, 75] - PP_ij[t, 0, 75] - QQ_ij[t, 0, 75]) == 0)

                model.addConstr(lv_ij_i[t, 0, 76] == l_ij[t, 0, 76] * v_i[t, 0, 76])
                model.addConstr((lv_ij_i[t, 0, 76] - PP_ij[t, 0, 76] - QQ_ij[t, 0, 76]) == 0)

                model.addConstr(lv_ij_i[t, 0, 77] == l_ij[t, 0, 77] * v_i[t, 0, 64])
                model.addConstr((lv_ij_i[t, 0, 77] - PP_ij[t, 0, 77] - QQ_ij[t, 0, 77]) == 0)

                model.addConstr(lv_ij_i[t, 0, 78] == l_ij[t, 0, 78] * v_i[t, 0, 78])
                model.addConstr((lv_ij_i[t, 0, 78] - PP_ij[t, 0, 78] - QQ_ij[t, 0, 78]) == 0)

                model.addConstr(lv_ij_i[t, 0, 79] == l_ij[t, 0, 79] * v_i[t, 0, 79])
                model.addConstr((lv_ij_i[t, 0, 79] - PP_ij[t, 0, 79] - QQ_ij[t, 0, 79]) == 0)

                model.addConstr(lv_ij_i[t, 0, 80] == l_ij[t, 0, 80] * v_i[t, 0, 80])
                model.addConstr((lv_ij_i[t, 0, 80] - PP_ij[t, 0, 80] - QQ_ij[t, 0, 80]) == 0)

                model.addConstr(lv_ij_i[t, 0, 81] == l_ij[t, 0, 81] * v_i[t, 0, 81])
                model.addConstr((lv_ij_i[t, 0, 81] - PP_ij[t, 0, 81] - QQ_ij[t, 0, 81]) == 0)

                model.addConstr(lv_ij_i[t, 0, 82] == l_ij[t, 0, 82] * v_i[t, 0, 82])
                model.addConstr((lv_ij_i[t, 0, 82] - PP_ij[t, 0, 82] - QQ_ij[t, 0, 82]) == 0)

                model.addConstr(lv_ij_i[t, 0, 83] == l_ij[t, 0, 83] * v_i[t, 0, 83])
                model.addConstr((lv_ij_i[t, 0, 83] - PP_ij[t, 0, 83] - QQ_ij[t, 0, 83]) == 0)

                model.addConstr(lv_ij_i[t, 0, 84] == l_ij[t, 0, 84] * v_i[t, 0, 84])
                model.addConstr((lv_ij_i[t, 0, 84] - PP_ij[t, 0, 84] - QQ_ij[t, 0, 84]) == 0)

                model.addConstr(lv_ij_i[t, 0, 85] == l_ij[t, 0, 85] * v_i[t, 0, 79])
                model.addConstr((lv_ij_i[t, 0, 85] - PP_ij[t, 0, 85] - QQ_ij[t, 0, 85]) == 0)

                model.addConstr(lv_ij_i[t, 0, 86] == l_ij[t, 0, 86] * v_i[t, 0, 86])
                model.addConstr((lv_ij_i[t, 0, 86] - PP_ij[t, 0, 86] - QQ_ij[t, 0, 86]) == 0)

                model.addConstr(lv_ij_i[t, 0, 87] == l_ij[t, 0, 87] * v_i[t, 0, 87])
                model.addConstr((lv_ij_i[t, 0, 87] - PP_ij[t, 0, 87] - QQ_ij[t, 0, 87]) == 0)

                model.addConstr(lv_ij_i[t, 0, 88] == l_ij[t, 0, 88] * v_i[t, 0, 65])
                model.addConstr((lv_ij_i[t, 0, 88] - PP_ij[t, 0, 88] - QQ_ij[t, 0, 88]) == 0)

                model.addConstr(lv_ij_i[t, 0, 89] == l_ij[t, 0, 89] * v_i[t, 0, 89])
                model.addConstr((lv_ij_i[t, 0, 89] - PP_ij[t, 0, 89] - QQ_ij[t, 0, 89]) == 0)

                model.addConstr(lv_ij_i[t, 0, 90] == l_ij[t, 0, 90] * v_i[t, 0, 90])
                model.addConstr((lv_ij_i[t, 0, 90] - PP_ij[t, 0, 90] - QQ_ij[t, 0, 90]) == 0)

                model.addConstr(lv_ij_i[t, 0, 91] == l_ij[t, 0, 91] * v_i[t, 0, 91])
                model.addConstr((lv_ij_i[t, 0, 91] - PP_ij[t, 0, 91] - QQ_ij[t, 0, 91]) == 0)

                model.addConstr(lv_ij_i[t, 0, 92] == l_ij[t, 0, 92] * v_i[t, 0, 92])
                model.addConstr((lv_ij_i[t, 0, 92] - PP_ij[t, 0, 92] - QQ_ij[t, 0, 92]) == 0)

                model.addConstr(lv_ij_i[t, 0, 93] == l_ij[t, 0, 93] * v_i[t, 0, 93])
                model.addConstr((lv_ij_i[t, 0, 93] - PP_ij[t, 0, 93] - QQ_ij[t, 0, 93]) == 0)

                model.addConstr(lv_ij_i[t, 0, 94] == l_ij[t, 0, 94] * v_i[t, 0, 94])
                model.addConstr((lv_ij_i[t, 0, 94] - PP_ij[t, 0, 94] - QQ_ij[t, 0, 94]) == 0)

                model.addConstr(lv_ij_i[t, 0, 95] == l_ij[t, 0, 95] * v_i[t, 0, 91])
                model.addConstr((lv_ij_i[t, 0, 95] - PP_ij[t, 0, 95] - QQ_ij[t, 0, 95]) == 0)

                model.addConstr(lv_ij_i[t, 0, 96] == l_ij[t, 0, 96] * v_i[t, 0, 96])
                model.addConstr((lv_ij_i[t, 0, 96] - PP_ij[t, 0, 96] - QQ_ij[t, 0, 96]) == 0)

                model.addConstr(lv_ij_i[t, 0, 97] == l_ij[t, 0, 97] * v_i[t, 0, 97])
                model.addConstr((lv_ij_i[t, 0, 97] - PP_ij[t, 0, 97] - QQ_ij[t, 0, 97]) == 0)

                model.addConstr(lv_ij_i[t, 0, 98] == l_ij[t, 0, 98] * v_i[t, 0, 98])
                model.addConstr((lv_ij_i[t, 0, 98] - PP_ij[t, 0, 98] - QQ_ij[t, 0, 98]) == 0)

                model.addConstr(lv_ij_i[t, 0, 99] == l_ij[t, 0, 99] * v_i[t, 0, 1])
                model.addConstr((lv_ij_i[t, 0, 99] - PP_ij[t, 0, 99] - QQ_ij[t, 0, 99]) == 0)

                model.addConstr(lv_ij_i[t, 0, 100] == l_ij[t, 0, 100] * v_i[t, 0, 100])
                model.addConstr((lv_ij_i[t, 0, 100] - PP_ij[t, 0, 100] - QQ_ij[t, 0, 100]) == 0)

                model.addConstr(lv_ij_i[t, 0, 101] == l_ij[t, 0, 101] * v_i[t, 0, 101])
                model.addConstr((lv_ij_i[t, 0, 101] - PP_ij[t, 0, 101] - QQ_ij[t, 0, 101]) == 0)

                model.addConstr(lv_ij_i[t, 0, 102] == l_ij[t, 0, 102] * v_i[t, 0, 102])
                model.addConstr((lv_ij_i[t, 0, 102] - PP_ij[t, 0, 102] - QQ_ij[t, 0, 102]) == 0)

                model.addConstr(lv_ij_i[t, 0, 103] == l_ij[t, 0, 103] * v_i[t, 0, 103])
                model.addConstr((lv_ij_i[t, 0, 103] - PP_ij[t, 0, 103] - QQ_ij[t, 0, 103]) == 0)

                model.addConstr(lv_ij_i[t, 0, 104] == l_ij[t, 0, 104] * v_i[t, 0, 104])
                model.addConstr((lv_ij_i[t, 0, 104] - PP_ij[t, 0, 104] - QQ_ij[t, 0, 104]) == 0)

                model.addConstr(lv_ij_i[t, 0, 105] == l_ij[t, 0, 105] * v_i[t, 0, 105])
                model.addConstr((lv_ij_i[t, 0, 105] - PP_ij[t, 0, 105] - QQ_ij[t, 0, 105]) == 0)

                model.addConstr(lv_ij_i[t, 0, 106] == l_ij[t, 0, 106] * v_i[t, 0, 106])
                model.addConstr((lv_ij_i[t, 0, 106] - PP_ij[t, 0, 106] - QQ_ij[t, 0, 106]) == 0)

                model.addConstr(lv_ij_i[t, 0, 107] == l_ij[t, 0, 107] * v_i[t, 0, 107])
                model.addConstr((lv_ij_i[t, 0, 107] - PP_ij[t, 0, 107] - QQ_ij[t, 0, 107]) == 0)

                model.addConstr(lv_ij_i[t, 0, 108] == l_ij[t, 0, 108] * v_i[t, 0, 108])
                model.addConstr((lv_ij_i[t, 0, 108] - PP_ij[t, 0, 108] - QQ_ij[t, 0, 108]) == 0)

                model.addConstr(lv_ij_i[t, 0, 109] == l_ij[t, 0, 109] * v_i[t, 0, 109])
                model.addConstr((lv_ij_i[t, 0, 109] - PP_ij[t, 0, 109] - QQ_ij[t, 0, 109]) == 0)

                model.addConstr(lv_ij_i[t, 0, 110] == l_ij[t, 0, 110] * v_i[t, 0, 110])
                model.addConstr((lv_ij_i[t, 0, 110] - PP_ij[t, 0, 110] - QQ_ij[t, 0, 110]) == 0)

                model.addConstr(lv_ij_i[t, 0, 111] == l_ij[t, 0, 111] * v_i[t, 0, 110])
                model.addConstr((lv_ij_i[t, 0, 111] - PP_ij[t, 0, 111] - QQ_ij[t, 0, 111]) == 0)

                model.addConstr(lv_ij_i[t, 0, 112] == l_ij[t, 0, 112] * v_i[t, 0, 112])
                model.addConstr((lv_ij_i[t, 0, 112] - PP_ij[t, 0, 112] - QQ_ij[t, 0, 112]) == 0)

                model.addConstr(lv_ij_i[t, 0, 113] == l_ij[t, 0, 113] * v_i[t, 0, 100])
                model.addConstr((lv_ij_i[t, 0, 113] - PP_ij[t, 0, 113] - QQ_ij[t, 0, 113]) == 0)

                model.addConstr(lv_ij_i[t, 0, 114] == l_ij[t, 0, 114] * v_i[t, 0, 114])
                model.addConstr((lv_ij_i[t, 0, 114] - PP_ij[t, 0, 114] - QQ_ij[t, 0, 114]) == 0)

                model.addConstr(lv_ij_i[t, 0, 115] == l_ij[t, 0, 115] * v_i[t, 0, 115])
                model.addConstr((lv_ij_i[t, 0, 115] - PP_ij[t, 0, 115] - QQ_ij[t, 0, 115]) == 0)

                model.addConstr(lv_ij_i[t, 0, 116] == l_ij[t, 0, 116] * v_i[t, 0, 116])
                model.addConstr((lv_ij_i[t, 0, 116] - PP_ij[t, 0, 116] - QQ_ij[t, 0, 116]) == 0)

                model.addConstr(lv_ij_i[t, 0, 117] == l_ij[t, 0, 117] * v_i[t, 0, 117])
                model.addConstr((lv_ij_i[t, 0, 117] - PP_ij[t, 0, 117] - QQ_ij[t, 0, 117]) == 0)

            model.setParam('outPutFlag', 0)
            model.setParam(GRB.Param.TimeLimit, 300)
            model.Params.MIPGap = 0.001
            model.optimize()

            obj=model.objval
            dict_value=dict()
            for v in model.getVars():
                dict_value[v.varName]=v.x
        except GurobiError as e:
            print('Error code ' + str(e.errno) + ':' + str(e))
            dict_value=dict()
            obj = 'error'
            dict_value['P_DG_29[0]'] = 'error'
            dict_value['P_DG_64[0]'] = 'error'
            dict_value['SOC_BSS_12[0]'] = 'error'
            dict_value['SOC_BSS_34[0]'] = 'error'
            dict_value['SOC_BSS_68[0]'] = 'error'
            dict_value['SOC_BSS_103[0]'] = 'error'
            dict_value['cost_hour[0]'] = 'error'
        except AttributeError:
            print('Encountered an attribute error')
            dict_value=dict()
            obj = 'error'
            dict_value['P_DG_29[0]'] = 'error'
            dict_value['P_DG_64[0]'] = 'error'
            dict_value['SOC_BSS_12[0]'] = 'error'
            dict_value['SOC_BSS_34[0]'] = 'error'
            dict_value['SOC_BSS_68[0]'] = 'error'
            dict_value['SOC_BSS_103[0]'] = 'error'
            dict_value['cost_hour[0]'] = 'error'

        return (obj, dict_value['P_DG_29[0]'], dict_value['P_DG_64[0]'],
                dict_value['SOC_BSS_12[0]'], dict_value['SOC_BSS_34[0]'], dict_value['SOC_BSS_68[0]'], dict_value['SOC_BSS_103[0]'],
                dict_value['cost_hour[0]'], model.Runtime)


def _get_matrix(agent, model_path_1 = 'critic_1_model.pth', model_path_2 = 'critic_2_model.pth',
                model_path_3 = 'critic_shadow_1_model.pth', model_path_4 = 'critic_shadow_2_model.pth'):
    agent.critic_1.load_state_dict(torch.load(model_path_1))
    agent.critic_1.eval()
    agent.critic_2.load_state_dict(torch.load(model_path_2))
    agent.critic_2.eval()
    state_dict_1 = agent.critic_1.state_dict()
    state_dict_2 = agent.critic_2.state_dict()

    list_1 = ['As.0', 'Ws.0', 'bs.0', 'As.1', 'Ws.1', 'bs.1']
    list_2 = []
    for i in range(len(list_1)):
        list_2.append(state_dict_1[list_1[i]].cpu().detach().numpy())
    list_3 = []
    for i in range(len(list_1)):
        list_3.append(state_dict_2[list_1[i]].cpu().detach().numpy())

    return list_2, list_3


def _get_scenario():
    a0_1 = np.random.uniform(0.6, 0.8, (12, 118))
    a0_2 = np.random.uniform(0.8, 1, (12, 118))
    a0_3 = np.random.uniform(1, 1.2, (12, 118))
    a0_4 = np.random.uniform(0.8, 1, (12, 118))
    a1 = np.vstack([a0_1, a0_2, a0_3, a0_4])
    ori_load = [0.0, 0.1338, 0.0162, 0.0343, 0.073, 0.1442, 0.1045, 0.0285, 0.0876, 0.1982, 0.1468, 0.026, 0.0521, 0.1419, 0.0219,
     0.0334, 0.0324, 0.0202, 0.1569, 0.5463, 0.1803, 0.0932, 0.0852, 0.1681, 0.1251, 0.016, 0.026, 0.5946, 0.1206,
     0.1024, 0.5134, 0.4753, 0.1514, 0.2054, 0.1316, 0.4484, 0.4405, 0.1125, 0.054, 0.3931, 0.3267, 0.5363, 0.0762,
     0.0535, 0.0403, 0.0397, 0.0662, 0.0739, 0.1148, 0.9184, 0.2103, 0.0667, 0.0422, 0.4337, 0.0621, 0.0925, 0.0852,
     0.3453, 0.0225, 0.0806, 0.0959, 0.0629, 0.4788, 0.1209, 0.1391, 0.3918, 0.0277, 0.0528, 0.0669, 0.4675, 0.5948,
     0.1325, 0.0527, 0.8698, 0.0313, 0.1924, 0.0658, 0.2382, 0.2946, 0.4856, 0.2435, 0.2435, 0.1343, 0.0227, 0.0495,
     0.3838, 0.0496, 0.0225, 0.0629, 0.0307, 0.0625, 0.1146, 0.0813, 0.0317, 0.0333, 0.5313, 0.507, 0.0264, 0.046,
     0.1007, 0.4565, 0.5226, 0.4084, 0.1415, 0.1044, 0.0968, 0.4939, 0.2254, 0.5092, 0.1885, 0.918, 0.3051, 0.0544,
     0.2111, 0.067, 0.1621, 0.0488, 0.0339]
    for i in range(len(ori_load)):
        ori_load[i] = ori_load[i] * 0.5

    load_list = []
    for i in range(24 * 2):
        load_list.append(a1[i] * ori_load)
    load_list = np.array(load_list)
    load_min_ratio = np.array(0.6)
    load_max_ratio = np.array(1.2)
    load_min = load_min_ratio * ori_load
    load_max = load_max_ratio * ori_load

    a0_1_Q = np.random.uniform(0.6, 0.8, (12, 118))
    a0_2_Q = np.random.uniform(0.8, 1, (12, 118))
    a0_3_Q = np.random.uniform(1, 1.2, (12, 118))
    a0_4_Q = np.random.uniform(0.8, 1, (12, 118))
    a1_Q = np.vstack([a0_1_Q, a0_2_Q, a0_3_Q, a0_4_Q])
    ori_load_Q = [0.0, 0.1011, 0.0113, 0.0218, 0.0636, 0.0686, 0.0617, 0.0115, 0.0511, 0.1068, 0.076, 0.0187, 0.0232, 0.1175, 0.0288,
     0.0264, 0.0252, 0.0119, 0.0785, 0.3514, 0.1642, 0.0546, 0.0396, 0.0952, 0.1502, 0.0246, 0.0246, 0.5226, 0.0591,
     0.0996, 0.3185, 0.4561, 0.1368, 0.0833, 0.0931, 0.3698, 0.3216, 0.0551, 0.039, 0.3426, 0.2786, 0.2402, 0.0666,
     0.0398, 0.032, 0.0208, 0.0424, 0.0517, 0.058, 1.2051, 0.1467, 0.0566, 0.0402, 0.2834, 0.0269, 0.0884, 0.0554,
     0.3324, 0.0168, 0.0492, 0.0908, 0.0477, 0.4637, 0.052, 0.1003, 0.1935, 0.0267, 0.0253, 0.0387, 0.3951, 0.2397,
     0.0844, 0.0225, 0.6148, 0.0298, 0.1224, 0.0454, 0.2232, 0.1625, 0.4379, 0.183, 0.183, 0.1193, 0.028, 0.0265,
     0.2572, 0.0206, 0.0118, 0.043, 0.0349, 0.0668, 0.0817, 0.0665, 0.016, 0.0605, 0.2248, 0.3674, 0.0117, 0.0304,
     0.0476, 0.3503, 0.4493, 0.1685, 0.1343, 0.066, 0.0836, 0.4193, 0.1359, 0.3872, 0.1735, 0.8985, 0.2154, 0.041,
     0.1929, 0.0533, 0.0903, 0.0292, 0.019]
    for i in range(len(ori_load_Q)):
        ori_load_Q[i] = ori_load_Q[i] * 0.5

    load_list_Q = []
    for i in range(24 * 2):
        load_list_Q.append(a1_Q[i] * ori_load_Q)
    load_list_Q = np.array(load_list_Q)
    load_min_ratio_Q = np.array(0.6)
    load_max_ratio_Q = np.array(1.2)
    load_min_Q = load_min_ratio_Q * ori_load_Q
    load_max_Q = load_max_ratio_Q * ori_load_Q

    wind_11_avail_list = np.random.uniform(1.2, 1.8, 24 * 2)
    wind_32_avail_list = np.random.uniform(1.2, 1.8, 24 * 2)
    wind_66_avail_list = np.random.uniform(1.2, 1.8, 24 * 2)
    wind_101_avail_list = np.random.uniform(1.2, 1.8, 24 * 2)

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

    return (load_list, load_list_Q,
            wind_11_avail_list, wind_32_avail_list, wind_66_avail_list, wind_101_avail_list,
            pv_22_avail_list, pv_36_avail_list, pv_70_avail_list, pv_107_avail_list,
            load_min, load_max, load_min_Q, load_max_Q)


def _get_network():
    V_base = 12.66 * 1000
    S_base = 1 * 1000000

    list_r_x = [(0, 1, 0.001, 0.001), (1, 2, 0.036, 0.01296), (2, 3, 0.033, 0.01188), (2, 4, 0.045, 0.0162),
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
    list_r_x_pu = []
    for k in range(len(list_r_x)):
        list_r_x_pu.append([list_r_x[k][2] / (V_base * V_base / S_base), list_r_x[k][3] / (V_base * V_base / S_base)])

    list_p_q = [[0, 0], [133.84, 101.14], [16.214, 11.292], [34.315, 21.845], [73.016, 63.602], [144.2, 68.604],
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

    list_p_q_pu = []
    for k in range(len(list_p_q)):
        list_p_q_pu.append([list_p_q[k][0] / 1000 * 0.5, list_p_q[k][1] / 1000 * 0.5])

    return list_r_x_pu, list_p_q_pu


def _renew_scenario(t, old_load_list, old_load_list_Q,
                    old_wind_11_avail_list, old_wind_32_avail_list, old_wind_66_avail_list, old_wind_101_avail_list,
                    old_pv_22_avail_list, old_pv_36_avail_list, old_pv_70_avail_list, old_pv_107_avail_list):
    (fuzhu_load_list, fuzhu_load_list_Q,
     fuzhu_wind_11_avail_list, fuzhu_wind_32_avail_list, fuzhu_wind_66_avail_list, fuzhu_wind_101_avail_list,
     fuzhu_pv_22_avail_list, fuzhu_pv_36_avail_list, fuzhu_pv_70_avail_list, fuzhu_pv_107_avail_list,
     _, _, _, _) = _get_scenario()

    new_load_list = old_load_list.copy()
    new_load_list_Q = old_load_list_Q.copy()
    new_wind_11_avail_list = old_wind_11_avail_list.copy()
    new_wind_32_avail_list = old_wind_32_avail_list.copy()
    new_wind_66_avail_list = old_wind_66_avail_list.copy()
    new_wind_101_avail_list = old_wind_101_avail_list.copy()
    new_pv_22_avail_list = old_pv_22_avail_list.copy()
    new_pv_36_avail_list = old_pv_36_avail_list.copy()
    new_pv_70_avail_list = old_pv_70_avail_list.copy()
    new_pv_107_avail_list = old_pv_107_avail_list.copy()

    for i in range(t + 4, 24 * 2):
        new_load_list[i] = (fuzhu_load_list[i]).copy()
        new_load_list_Q[i] = (fuzhu_load_list_Q[i]).copy()
        new_wind_11_avail_list[i] = (fuzhu_wind_11_avail_list[i]).copy()
        new_wind_32_avail_list[i] = (fuzhu_wind_32_avail_list[i]).copy()
        new_wind_66_avail_list[i] = (fuzhu_wind_66_avail_list[i]).copy()
        new_wind_101_avail_list[i] = (fuzhu_wind_101_avail_list[i]).copy()
        new_pv_22_avail_list[i] = (fuzhu_pv_22_avail_list[i]).copy()
        new_pv_36_avail_list[i] = (fuzhu_pv_36_avail_list[i]).copy()
        new_pv_70_avail_list[i] = (fuzhu_pv_70_avail_list[i]).copy()
        new_pv_107_avail_list[i] = (fuzhu_pv_107_avail_list[i]).copy()

    new_load_list[t + 1] = old_load_list[t + 1] * 0.7 + fuzhu_load_list[t + 1] * 0.3
    new_load_list[t + 2] = old_load_list[t + 2] * 0.4 + fuzhu_load_list[t + 2] * 0.6
    new_load_list[t + 3] = old_load_list[t + 3] * 0.1 + fuzhu_load_list[t + 3] * 0.9
    new_load_list_Q[t + 1] = old_load_list_Q[t + 1] * 0.7 + fuzhu_load_list_Q[t + 1] * 0.3
    new_load_list_Q[t + 2] = old_load_list_Q[t + 2] * 0.4 + fuzhu_load_list_Q[t + 2] * 0.6
    new_load_list_Q[t + 3] = old_load_list_Q[t + 3] * 0.1 + fuzhu_load_list_Q[t + 3] * 0.9
    new_wind_11_avail_list[t + 1] = old_wind_11_avail_list[t + 1] * 0.7 + fuzhu_wind_11_avail_list[t + 1] * 0.3
    new_wind_11_avail_list[t + 2] = old_wind_11_avail_list[t + 2] * 0.4 + fuzhu_wind_11_avail_list[t + 2] * 0.6
    new_wind_11_avail_list[t + 3] = old_wind_11_avail_list[t + 3] * 0.1 + fuzhu_wind_11_avail_list[t + 3] * 0.9
    new_wind_32_avail_list[t + 1] = old_wind_32_avail_list[t + 1] * 0.7 + fuzhu_wind_32_avail_list[t + 1] * 0.3
    new_wind_32_avail_list[t + 2] = old_wind_32_avail_list[t + 2] * 0.4 + fuzhu_wind_32_avail_list[t + 2] * 0.6
    new_wind_32_avail_list[t + 3] = old_wind_32_avail_list[t + 3] * 0.1 + fuzhu_wind_32_avail_list[t + 3] * 0.9
    new_wind_66_avail_list[t + 1] = old_wind_66_avail_list[t + 1] * 0.7 + fuzhu_wind_66_avail_list[t + 1] * 0.3
    new_wind_66_avail_list[t + 2] = old_wind_66_avail_list[t + 2] * 0.4 + fuzhu_wind_66_avail_list[t + 2] * 0.6
    new_wind_66_avail_list[t + 3] = old_wind_66_avail_list[t + 3] * 0.1 + fuzhu_wind_66_avail_list[t + 3] * 0.9
    new_wind_101_avail_list[t + 1] = old_wind_101_avail_list[t + 1] * 0.7 + fuzhu_wind_101_avail_list[t + 1] * 0.3
    new_wind_101_avail_list[t + 2] = old_wind_101_avail_list[t + 2] * 0.4 + fuzhu_wind_101_avail_list[t + 2] * 0.6
    new_wind_101_avail_list[t + 3] = old_wind_101_avail_list[t + 3] * 0.1 + fuzhu_wind_101_avail_list[t + 3] * 0.9
    new_pv_22_avail_list[t + 1] = old_pv_22_avail_list[t + 1] * 0.7 + fuzhu_pv_22_avail_list[t + 1] * 0.3
    new_pv_22_avail_list[t + 2] = old_pv_22_avail_list[t + 2] * 0.4 + fuzhu_pv_22_avail_list[t + 2] * 0.6
    new_pv_22_avail_list[t + 3] = old_pv_22_avail_list[t + 3] * 0.1 + fuzhu_pv_22_avail_list[t + 3] * 0.9
    new_pv_36_avail_list[t + 1] = old_pv_36_avail_list[t + 1] * 0.7 + fuzhu_pv_36_avail_list[t + 1] * 0.3
    new_pv_36_avail_list[t + 2] = old_pv_36_avail_list[t + 2] * 0.4 + fuzhu_pv_36_avail_list[t + 2] * 0.6
    new_pv_36_avail_list[t + 3] = old_pv_36_avail_list[t + 3] * 0.1 + fuzhu_pv_36_avail_list[t + 3] * 0.9
    new_pv_70_avail_list[t + 1] = old_pv_70_avail_list[t + 1] * 0.7 + fuzhu_pv_70_avail_list[t + 1] * 0.3
    new_pv_70_avail_list[t + 2] = old_pv_70_avail_list[t + 2] * 0.4 + fuzhu_pv_70_avail_list[t + 2] * 0.6
    new_pv_70_avail_list[t + 3] = old_pv_70_avail_list[t + 3] * 0.1 + fuzhu_pv_70_avail_list[t + 3] * 0.9
    new_pv_107_avail_list[t + 1] = old_pv_107_avail_list[t + 1] * 0.7 + fuzhu_pv_107_avail_list[t + 1] * 0.3
    new_pv_107_avail_list[t + 2] = old_pv_107_avail_list[t + 2] * 0.4 + fuzhu_pv_107_avail_list[t + 2] * 0.6
    new_pv_107_avail_list[t + 3] = old_pv_107_avail_list[t + 3] * 0.1 + fuzhu_pv_107_avail_list[t + 3] * 0.9

    return (new_load_list, new_load_list_Q,
            new_wind_11_avail_list, new_wind_32_avail_list, new_wind_66_avail_list, new_wind_101_avail_list,
            new_pv_22_avail_list, new_pv_36_avail_list, new_pv_70_avail_list, new_pv_107_avail_list)


def _renew_scenario_remainder(t, old_load_list, old_load_list_Q,
                    old_wind_11_avail_list, old_wind_32_avail_list, old_wind_66_avail_list, old_wind_101_avail_list,
                    old_pv_22_avail_list, old_pv_36_avail_list, old_pv_70_avail_list, old_pv_107_avail_list):
    (fuzhu_load_list, fuzhu_load_list_Q,
     fuzhu_wind_11_avail_list, fuzhu_wind_32_avail_list, fuzhu_wind_66_avail_list, fuzhu_wind_101_avail_list,
     fuzhu_pv_22_avail_list, fuzhu_pv_36_avail_list, fuzhu_pv_70_avail_list, fuzhu_pv_107_avail_list,
     _, _, _, _) = _get_scenario()

    new_load_list = old_load_list.copy()
    new_load_list_Q = old_load_list_Q.copy()
    new_wind_11_avail_list = old_wind_11_avail_list.copy()
    new_wind_32_avail_list = old_wind_32_avail_list.copy()
    new_wind_66_avail_list = old_wind_66_avail_list.copy()
    new_wind_101_avail_list = old_wind_101_avail_list.copy()
    new_pv_22_avail_list = old_pv_22_avail_list.copy()
    new_pv_36_avail_list = old_pv_36_avail_list.copy()
    new_pv_70_avail_list = old_pv_70_avail_list.copy()
    new_pv_107_avail_list = old_pv_107_avail_list.copy()

    if t < (24 + 20):
        for i in range(t + 4, 24 * 2):
            new_load_list[i] = (fuzhu_load_list[i]).copy()
            new_load_list_Q[i] = (fuzhu_load_list_Q[i]).copy()
            new_wind_11_avail_list[i] = (fuzhu_wind_11_avail_list[i]).copy()
            new_wind_32_avail_list[i] = (fuzhu_wind_32_avail_list[i]).copy()
            new_wind_66_avail_list[i] = (fuzhu_wind_66_avail_list[i]).copy()
            new_wind_101_avail_list[i] = (fuzhu_wind_101_avail_list[i]).copy()
            new_pv_22_avail_list[i] = (fuzhu_pv_22_avail_list[i]).copy()
            new_pv_36_avail_list[i] = (fuzhu_pv_36_avail_list[i]).copy()
            new_pv_70_avail_list[i] = (fuzhu_pv_70_avail_list[i]).copy()
            new_pv_107_avail_list[i] = (fuzhu_pv_107_avail_list[i]).copy()
    if t < (24 + 23):
        new_load_list[t + 1] = old_load_list[t + 1] * 0.7 + fuzhu_load_list[t + 1] * 0.3
        new_load_list_Q[t + 1] = old_load_list_Q[t + 1] * 0.7 + fuzhu_load_list_Q[t + 1] * 0.3
        new_wind_11_avail_list[t + 1] = old_wind_11_avail_list[t + 1] * 0.7 + fuzhu_wind_11_avail_list[t + 1] * 0.3
        new_wind_32_avail_list[t + 1] = old_wind_32_avail_list[t + 1] * 0.7 + fuzhu_wind_32_avail_list[t + 1] * 0.3
        new_wind_66_avail_list[t + 1] = old_wind_66_avail_list[t + 1] * 0.7 + fuzhu_wind_66_avail_list[t + 1] * 0.3
        new_wind_101_avail_list[t + 1] = old_wind_101_avail_list[t + 1] * 0.7 + fuzhu_wind_101_avail_list[t + 1] * 0.3
        new_pv_22_avail_list[t + 1] = old_pv_22_avail_list[t + 1] * 0.7 + fuzhu_pv_22_avail_list[t + 1] * 0.3
        new_pv_36_avail_list[t + 1] = old_pv_36_avail_list[t + 1] * 0.7 + fuzhu_pv_36_avail_list[t + 1] * 0.3
        new_pv_70_avail_list[t + 1] = old_pv_70_avail_list[t + 1] * 0.7 + fuzhu_pv_70_avail_list[t + 1] * 0.3
        new_pv_107_avail_list[t + 1] = old_pv_107_avail_list[t + 1] * 0.7 + fuzhu_pv_107_avail_list[t + 1] * 0.3
    if t < (24 + 22):
        new_load_list[t + 2] = old_load_list[t + 2] * 0.4 + fuzhu_load_list[t + 2] * 0.6
        new_load_list_Q[t + 2] = old_load_list_Q[t + 2] * 0.4 + fuzhu_load_list_Q[t + 2] * 0.6
        new_wind_11_avail_list[t + 2] = old_wind_11_avail_list[t + 2] * 0.4 + fuzhu_wind_11_avail_list[t + 2] * 0.6
        new_wind_32_avail_list[t + 2] = old_wind_32_avail_list[t + 2] * 0.4 + fuzhu_wind_32_avail_list[t + 2] * 0.6
        new_wind_66_avail_list[t + 2] = old_wind_66_avail_list[t + 2] * 0.4 + fuzhu_wind_66_avail_list[t + 2] * 0.6
        new_wind_101_avail_list[t + 2] = old_wind_101_avail_list[t + 2] * 0.4 + fuzhu_wind_101_avail_list[t + 2] * 0.6
        new_pv_22_avail_list[t + 2] = old_pv_22_avail_list[t + 2] * 0.4 + fuzhu_pv_22_avail_list[t + 2] * 0.6
        new_pv_36_avail_list[t + 2] = old_pv_36_avail_list[t + 2] * 0.4 + fuzhu_pv_36_avail_list[t + 2] * 0.6
        new_pv_70_avail_list[t + 2] = old_pv_70_avail_list[t + 2] * 0.4 + fuzhu_pv_70_avail_list[t + 2] * 0.6
        new_pv_107_avail_list[t + 2] = old_pv_107_avail_list[t + 2] * 0.4 + fuzhu_pv_107_avail_list[t + 2] * 0.6
    if t < (24 + 21):
        new_load_list[t + 3] = old_load_list[t + 3] * 0.1 + fuzhu_load_list[t + 3] * 0.9
        new_load_list_Q[t + 3] = old_load_list_Q[t + 3] * 0.1 + fuzhu_load_list_Q[t + 3] * 0.9
        new_wind_11_avail_list[t + 3] = old_wind_11_avail_list[t + 3] * 0.1 + fuzhu_wind_11_avail_list[t + 3] * 0.9
        new_wind_32_avail_list[t + 3] = old_wind_32_avail_list[t + 3] * 0.1 + fuzhu_wind_32_avail_list[t + 3] * 0.9
        new_wind_66_avail_list[t + 3] = old_wind_66_avail_list[t + 3] * 0.1 + fuzhu_wind_66_avail_list[t + 3] * 0.9
        new_wind_101_avail_list[t + 3] = old_wind_101_avail_list[t + 3] * 0.1 + fuzhu_wind_101_avail_list[t + 3] * 0.9
        new_pv_22_avail_list[t + 3] = old_pv_22_avail_list[t + 3] * 0.1 + fuzhu_pv_22_avail_list[t + 3] * 0.9
        new_pv_36_avail_list[t + 3] = old_pv_36_avail_list[t + 3] * 0.1 + fuzhu_pv_36_avail_list[t + 3] * 0.9
        new_pv_70_avail_list[t + 3] = old_pv_70_avail_list[t + 3] * 0.1 + fuzhu_pv_70_avail_list[t + 3] * 0.9
        new_pv_107_avail_list[t + 3] = old_pv_107_avail_list[t + 3] * 0.1 + fuzhu_pv_107_avail_list[t + 3] * 0.9

    return (new_load_list, new_load_list_Q,
            new_wind_11_avail_list, new_wind_32_avail_list, new_wind_66_avail_list, new_wind_101_avail_list,
            new_pv_22_avail_list, new_pv_36_avail_list, new_pv_70_avail_list, new_pv_107_avail_list)

env = PowerSystemEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

seedlist = [89,91]
objlist = []
timelist = []

for seed in seedlist:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    actor_lr = 3e-4
    critic_lr = 3e-3
    alpha_lr = 3e-4
    num_episodes = 7000
    hidden_dim = 128
    gamma = 0.99
    tau = 0.005
    buffer_size = 100000
    minimal_size = 1000
    batch_size = 64
    target_entropy = -env.action_space.shape[0]

    device = torch.device("cpu")
    print(device)

    replay_buffer = rl_utils.ReplayBuffer(buffer_size)
    agent = SACContinuous(state_dim, hidden_dim, action_dim,
                          actor_lr, critic_lr, alpha_lr, target_entropy, tau,
                          gamma, device)

    Q_1_mat, Q_2_mat = _get_matrix(agent)

    (load_list, load_list_Q,
    wind_11_avail_list, wind_32_avail_list, wind_66_avail_list, wind_101_avail_list,
    pv_22_avail_list, pv_36_avail_list, pv_70_avail_list, pv_107_avail_list,
    load_min, load_max, load_min_Q, load_max_Q) = _get_scenario()

    list_r_x_pu, list_p_q_pu = _get_network()

    MPC = Solve_MPC(Q_1_mat, Q_2_mat, load_min, load_max, list_r_x_pu, list_p_q_pu, load_min_Q, load_max_Q)

    current_time = 0

    P_DG_29_init = 0
    P_DG_64_init = 0

    SOC_BSS_12_init = 1.25
    SOC_BSS_34_init = 1.25
    SOC_BSS_68_init = 1.25
    SOC_BSS_103_init = 1.25

    cost_all = 0
    window_time = 4

    solve_time_all = 0

    for t in range(24 * 2 - window_time + 1):
        (obj, new_P_DG_29_init, new_P_DG_64_init,
         new_SOC_BSS_12_init, new_SOC_BSS_34_init, new_SOC_BSS_68_init, new_SOC_BSS_103_init,
         cost_hour_now, solve_time) \
            = MPC.sol_pro(
                    current_time, window_time,
                    load_list, load_list_Q,
                    P_DG_29_init, P_DG_64_init,
                    SOC_BSS_12_init, SOC_BSS_34_init, SOC_BSS_68_init, SOC_BSS_103_init,
                    wind_11_avail_list, wind_32_avail_list, wind_66_avail_list, wind_101_avail_list,
                    pv_22_avail_list, pv_36_avail_list, pv_70_avail_list, pv_107_avail_list)

        solve_time_all += solve_time

        (load_list, load_list_Q, wind_11_avail_list, wind_32_avail_list, wind_66_avail_list, wind_101_avail_list,
         pv_22_avail_list, pv_36_avail_list, pv_70_avail_list, pv_107_avail_list) = _renew_scenario(
            t, load_list, load_list_Q,
            wind_11_avail_list, wind_32_avail_list, wind_66_avail_list, wind_101_avail_list,
            pv_22_avail_list, pv_36_avail_list, pv_70_avail_list, pv_107_avail_list)

        cost_all += cost_hour_now

        current_time += 1
        P_DG_29_init = new_P_DG_29_init
        P_DG_64_init = new_P_DG_64_init
        SOC_BSS_12_init = new_SOC_BSS_12_init
        SOC_BSS_34_init = new_SOC_BSS_34_init
        SOC_BSS_68_init = new_SOC_BSS_68_init
        SOC_BSS_103_init = new_SOC_BSS_103_init

    for t in range(24 * 2 - window_time + 1, 24 * 2):
        window_time -= 1

        (obj, new_P_DG_29_init, new_P_DG_64_init,
         new_SOC_BSS_12_init, new_SOC_BSS_34_init, new_SOC_BSS_68_init, new_SOC_BSS_103_init,
         cost_hour_now, solve_time) \
            = MPC.sol_pro_remainder(
                    current_time, window_time,
                    load_list, load_list_Q,
                    P_DG_29_init, P_DG_64_init,
                    SOC_BSS_12_init, SOC_BSS_34_init, SOC_BSS_68_init, SOC_BSS_103_init,
                    wind_11_avail_list, wind_32_avail_list, wind_66_avail_list, wind_101_avail_list,
                    pv_22_avail_list, pv_36_avail_list, pv_70_avail_list, pv_107_avail_list)

        solve_time_all += solve_time

        if t < (24 + 23):
            (load_list, load_list_Q, wind_11_avail_list, wind_32_avail_list, wind_66_avail_list, wind_101_avail_list,
             pv_22_avail_list, pv_36_avail_list, pv_70_avail_list, pv_107_avail_list) = _renew_scenario_remainder(
                t, load_list, load_list_Q,
                wind_11_avail_list, wind_32_avail_list, wind_66_avail_list, wind_101_avail_list,
                pv_22_avail_list, pv_36_avail_list, pv_70_avail_list, pv_107_avail_list)

        cost_all += cost_hour_now

        if t == (24 + 23):
            objlist.append(cost_all)
            timelist.append(solve_time_all)

            list_1 = list(load_list)
            for i in range(len(list_1)):
                list_1[i] = list(list_1[i])

            list_2 = list(load_list_Q)
            for i in range(len(list_2)):
                list_2[i] = list(list_2[i])

        current_time += 1
        P_DG_29_init = new_P_DG_29_init
        P_DG_64_init = new_P_DG_64_init
        SOC_BSS_12_init = new_SOC_BSS_12_init
        SOC_BSS_34_init = new_SOC_BSS_34_init
        SOC_BSS_68_init = new_SOC_BSS_68_init
        SOC_BSS_103_init = new_SOC_BSS_103_init

    print("seed =", seed)
    print("", cost_all)
    print("", solve_time_all)

print("", objlist)
print("", timelist)

