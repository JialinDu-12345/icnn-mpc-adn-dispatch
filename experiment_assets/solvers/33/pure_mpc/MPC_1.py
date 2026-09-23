from gurobipy import *
import gurobipy as grb
import matplotlib.pyplot as plt
import torch
import random
import numpy as np
from SAC_largesystem import PowerSystemEnv, SACContinuous
import rl_utils

import time
import pandas as pd

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
               
        self.pv_11_min = 0
        self.pv_11_max = 3
               
        self.wind_26_min = 0
        self.wind_26_max = 2
              
        self.svc_7_min = -0.3
        self.svc_7_max = 0.3
        self.svc_7_scale = 0.3

                        
        self.scale_Q = 0

                 
               
                           
                          
                        
        self.a_5_2 = 0.00240
        self.a_5_1 = 12.3299
        self.a_5_0 = 0

              
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
            for i in range(32):
                list_p_q_pu_t[t][i][0] = np.array(self.load_list)[t][i].copy()
                list_p_q_pu_t[t][i][1] = np.array(self.load_list_Q)[t][i].copy()

        return list_p_q_pu_t

    def sol_pro(self, current_time, P_DG_5_init, SOC_BSS_9_init, load_list, pv_11_avail_list, wind_26_avail_list,
                window_time, load_list_Q):

                 
        self.load_list = load_list
        self.load_list_Q = load_list_Q
        self.pv_11_avail_list = pv_11_avail_list
        self.wind_26_avail_list = wind_26_avail_list
        self.list_p_q_pu_t = self._get_pq_t()

                
        A0_1 = self.Q_1_mat[0]
        W0_1 = self.Q_1_mat[1]
        b0_1 = self.Q_1_mat[2]
        A1_1 = self.Q_1_mat[3]
        W1_1 = self.Q_1_mat[4]
        b1_1 = self.Q_1_mat[5]

                
        A0_2 = self.Q_2_mat[0]
        W0_2 = self.Q_2_mat[1]
        b0_2 = self.Q_2_mat[2]
        A1_2 = self.Q_2_mat[3]
        W1_2 = self.Q_2_mat[4]
        b1_2 = self.Q_2_mat[5]

        obj = "error"
        try:
                
            model = Model('Pure MPC')
                                            
                                          
                              
            model.setParam('MIPFocus',0)

                    
            z_0_1 = model.addMVar(shape=(1, 74), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z_0_1')
            z_1_1 = model.addMVar(shape=(1, 128), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z_1_1')
            z_1_relu_1 = model.addMVar(shape=(1, 128), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z_1_relu_1')
            z_2_1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z_2_1')

                    
            z_0_2 = model.addMVar(shape=(1, 74), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z_0_2')
            z_1_2 = model.addMVar(shape=(1, 128), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z_1_2')
            z_1_relu_2 = model.addMVar(shape=(1, 128), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z_1_relu_2')
            z_2_2 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z_2_2')

                  
            Q_f = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Q_f')
               
               
               

            z0 = model.addMVar(shape=(74,), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='z0')
            t_normal_st = (window_time - 1 + current_time) / 47
            load_normal_st = model.addMVar(shape=(32,), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='load_normal_st')

            load_normal_st_Q = model.addMVar(shape=(32,), lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS,
                                             name='load_normal_st_Q')

            P_PV_11_normal_st = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='P_PV_11_normal_st')
            P_WT_26_normal_st = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='P_WT_26_normal_st')
            P_DG_5_normal_st_1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='P_DG_5_normal_st_1')
            P_BSS_9_normal_st_1 = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='P_BSS_9_normal_st_1')
            Actor_5_st = model.addVar(lb=-1, ub=1, vtype=GRB.CONTINUOUS, name='Actor_5_st')
            Actor_9_st = model.addVar(lb=-1, ub=1, vtype=GRB.CONTINUOUS, name='Actor_9_st')
            Actor_7_st = model.addVar(lb=-1, ub=1, vtype=GRB.CONTINUOUS, name='Actor_7_st')
            Actor_11_st = model.addVar(lb=-1, ub=1, vtype=GRB.CONTINUOUS, name='Actor_11_st')
            Actor_26_st = model.addVar(lb=-1, ub=1, vtype=GRB.CONTINUOUS, name='Actor_26_st')
               
                
               

                  
            P_ij = model.addMVar(shape=(window_time - 1, 1, 32), lb=-100, ub=100, vtype=GRB.CONTINUOUS, name='P_ij')
            Q_ij = model.addMVar(shape=(window_time - 1, 1, 32), lb=-100, ub=100, vtype=GRB.CONTINUOUS, name='Q_ij')
            l_ij = model.addMVar(shape=(window_time - 1, 1, 32), lb=0, vtype=GRB.CONTINUOUS, name='l_ij')
                                                                                                                       
                                             
            v_i = model.addMVar(shape=(window_time - 1, 1, 33), lb=0.94 * 0.94, ub=1.06 * 1.06, vtype=GRB.CONTINUOUS, name='v_i')
            lv_ij_i = model.addMVar(shape=(window_time - 1, 1, 32), vtype=GRB.CONTINUOUS, name='lv_ij_i')
            PP_ij = model.addMVar(shape=(window_time - 1, 1, 32), vtype=GRB.CONTINUOUS, name='PP_ij')
            QQ_ij = model.addMVar(shape=(window_time - 1, 1, 32), vtype=GRB.CONTINUOUS, name='QQ_ij')

                  
            Loss = model.addMVar(shape=(window_time - 1,), lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Loss')

                     
            Q_add_7 = model.addMVar(shape=(window_time - 1,), lb=-0.3, ub=0.3, vtype=GRB.CONTINUOUS, name='Q_add_7')
            Q_add_11 = model.addMVar(shape=(window_time - 1,), vtype=GRB.CONTINUOUS, name='Q_add_11')
            Q_add_26 = model.addMVar(shape=(window_time - 1,), vtype=GRB.CONTINUOUS, name='Q_add_26')

                            
            P_DG_5 = model.addMVar(shape=(window_time - 1,), lb=self.gen_min, ub=self.gen_max, vtype=GRB.CONTINUOUS, name='P_DG_5')
            SOC_BSS_9 = model.addMVar(shape=(window_time - 1,), lb= self.SOC_min, ub=self.SOC_max, vtype=GRB.CONTINUOUS, name='SOC_BSS_9')
                   
                                                                                           
                                                                                                         
            P_BSS_ch_9 = model.addMVar(shape=(window_time - 1,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS, name='P_BSS_ch_9')
            P_BSS_dch_9 = model.addMVar(shape=(window_time - 1,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS, name='P_BSS_dch_9')

                
            cost_balance = model.addMVar(shape=(window_time - 1,), vtype=GRB.CONTINUOUS, name='cost_balance')
            P_balance = model.addMVar(shape=(window_time - 1,), vtype=GRB.CONTINUOUS, name='P_balance')
            cost_window = model.addVar(vtype=GRB.CONTINUOUS, name='cost_window')
            cost_day = model.addVar(vtype=GRB.CONTINUOUS, name='cost_day')
            cost_hour = model.addMVar(shape=(window_time - 1,), vtype=GRB.CONTINUOUS, name='cost_hour')
            cost_genera = model.addMVar(shape=(window_time - 1,), vtype=GRB.CONTINUOUS, name='cost_genera')

               
                
                      
                         
                            
               

                  
            model.setObjective(cost_day, GRB.MINIMIZE)
               
                
               

            model.addConstr(cost_day == cost_window - Q_f * 100 * self.scale_Q)
            model.addConstr(cost_window == cost_hour.sum())
            for t in range(window_time - 1):
                model.addConstr(cost_hour[t] == cost_genera[t] + cost_balance[t])
                model.addConstr(cost_balance[t] == (P_balance[t] * self.Prices[t + current_time]) / 2)
                model.addConstr(P_balance[t] == grb.max_(P_ij[t, 0, 0], 0))
                model.addConstr(cost_genera[t] == (self.a_5_2 * P_DG_5[t] * P_DG_5[t] + self.a_5_1 * P_DG_5[t] + self.a_5_0) / 2)
               
                
                                                      
               

            for t in range(window_time - 1):
                model.addConstr(self.pv_11_avail_list[t + current_time] * self.pv_11_avail_list[t + current_time]
                                + Q_add_11[t] * Q_add_11[t] <= self.pv_11_max * self.pv_11_max)
                model.addConstr(self.wind_26_avail_list[t + current_time] * self.wind_26_avail_list[t + current_time]
                                + Q_add_26[t] * Q_add_26[t] <= self.wind_26_max * self.wind_26_max)
               
                   
               

            model.addConstr(P_DG_5[0] <= P_DG_5_init + self.gen_ramp / 2)
            model.addConstr(P_DG_5[0] >= P_DG_5_init - self.gen_ramp / 2)
            for t in range(window_time - 2):
                model.addConstr(P_DG_5[t + 1] <= P_DG_5[t] + self.gen_ramp / 2)
                model.addConstr(P_DG_5[t + 1] >= P_DG_5[t] - self.gen_ramp / 2)
               
                   
                                    
               

                   
            model.addConstr(SOC_BSS_9[0] == SOC_BSS_9_init + (P_BSS_ch_9[0] - P_BSS_dch_9[0]) / 2)
            for t in range(window_time - 2):
                model.addConstr(SOC_BSS_9[t + 1] == SOC_BSS_9[t] + (P_BSS_ch_9[t + 1] - P_BSS_dch_9[t + 1]) / 2)
                                              
                                                                                 
                                                                                        
               
                  
                                  
               

                    
            for i in range(74):
                model.addConstr(z_0_1[0, i] == z0[i])
            model.addConstr(z_1_1 == z_0_1 @ (A0_1.T) + z_0_1 @ (W0_1.T) + b0_1)
            for i in range(128):
                model.addConstr(z_1_relu_1[0, i] == grb.max_(z_1_1[0, i], 0))
            model.addConstr(z_2_1 == z_0_1 @ (A1_1.T) + z_1_relu_1 @ (W1_1.T) + b1_1)

                    
            for i in range(74):
                model.addConstr(z_0_2[0, i] == z0[i])
            model.addConstr(z_1_2 == z_0_2 @ (A0_2.T) + z_0_2 @ (W0_2.T) + b0_2)
            for i in range(128):
                model.addConstr(z_1_relu_2[0, i] == grb.max_(z_1_2[0, i],0))
            model.addConstr(z_2_2 == z_0_2 @ (A1_2.T) + z_1_relu_2 @ (W1_2.T) + b1_2)

                    
            model.addConstr(Q_f == grb.min_(z_2_1, z_2_2))
               
                 
               

            for i in range(32):
                model.addConstr(load_normal_st[i] ==
                                (self.list_p_q_pu_t[window_time - 1 + current_time][i][0] - self.load_min[i])
                                / (self.load_max[i] - self.load_min[i] + 1e-6) )
                model.addConstr(load_normal_st_Q[i] ==
                                (self.list_p_q_pu_t[window_time - 1 + current_time][i][1] - self.load_min_Q[i])
                                / (self.load_max_Q[i] - self.load_min_Q[i] + 1e-6))
            model.addConstr(P_PV_11_normal_st ==
                            (self.pv_11_avail_list[window_time - 1 + current_time] - self.pv_11_min)
                            / (self.pv_11_max - self.pv_11_min + 1e-6))
            model.addConstr(P_WT_26_normal_st ==
                            (self.wind_26_avail_list[window_time - 1 + current_time] - self.wind_26_min)
                            / (self.wind_26_max - self.wind_26_min + 1e-6))
            model.addConstr(P_DG_5_normal_st_1 == (P_DG_5[window_time - 2] - self.gen_min)
                            / (self.gen_max - self.gen_min + 1e-6))
            model.addConstr(P_BSS_9_normal_st_1 == (SOC_BSS_9[window_time - 2] - self.SOC_min)
                            / (self.SOC_max - self.SOC_min + 1e-6))

            model.addConstr(z0[0] == t_normal_st)
            for i in range(32):
                model.addConstr(z0[i+1] == load_normal_st[i])
            for i in range(32):
                model.addConstr(z0[i+33] == load_normal_st_Q[i])
            model.addConstr(z0[65] == P_PV_11_normal_st)
            model.addConstr(z0[66] == P_WT_26_normal_st)
            model.addConstr(z0[67] == P_DG_5_normal_st_1)
            model.addConstr(z0[68] == P_BSS_9_normal_st_1)
                           
            model.addConstr(z0[69] == Actor_5_st)
            model.addConstr(z0[70] == Actor_9_st)
            model.addConstr(z0[71] == Actor_7_st)
            model.addConstr(z0[72] == Actor_11_st)
            model.addConstr(z0[73] == Actor_26_st)
               
                
               

            for t in range(window_time - 1):
                      
                model.addConstr(v_i[t, 0, 0] == 1)

                      
                model.addConstr(Loss[t] == self.list_r_x_pu[0][0] * l_ij[t, 0, 0]
                                + self.list_r_x_pu[1][0] * l_ij[t, 0, 1] + self.list_r_x_pu[2][0] * l_ij[t, 0, 2]
                                + self.list_r_x_pu[3][0] * l_ij[t, 0, 3] + self.list_r_x_pu[4][0] * l_ij[t, 0, 4]
                                + self.list_r_x_pu[5][0] * l_ij[t, 0, 5] + self.list_r_x_pu[6][0] * l_ij[t, 0, 6]
                                + self.list_r_x_pu[7][0] * l_ij[t, 0, 7] + self.list_r_x_pu[8][0] * l_ij[t, 0, 8]
                                + self.list_r_x_pu[9][0] * l_ij[t, 0, 9] + self.list_r_x_pu[10][0] * l_ij[t, 0, 10]
                                + self.list_r_x_pu[11][0] * l_ij[t, 0, 11] + self.list_r_x_pu[12][0] * l_ij[t, 0, 12]
                                + self.list_r_x_pu[13][0] * l_ij[t, 0, 13] + self.list_r_x_pu[14][0] * l_ij[t, 0, 14]
                                + self.list_r_x_pu[15][0] * l_ij[t, 0, 15] + self.list_r_x_pu[16][0] * l_ij[t, 0, 16]
                                + self.list_r_x_pu[17][0] * l_ij[t, 0, 17] + self.list_r_x_pu[18][0] * l_ij[t, 0, 18]
                                + self.list_r_x_pu[19][0] * l_ij[t, 0, 19] + self.list_r_x_pu[20][0] * l_ij[t, 0, 20]
                                + self.list_r_x_pu[21][0] * l_ij[t, 0, 21] + self.list_r_x_pu[22][0] * l_ij[t, 0, 22]
                                + self.list_r_x_pu[23][0] * l_ij[t, 0, 23] + self.list_r_x_pu[24][0] * l_ij[t, 0, 24]
                                + self.list_r_x_pu[25][0] * l_ij[t, 0, 25] + self.list_r_x_pu[26][0] * l_ij[t, 0, 26]
                                + self.list_r_x_pu[27][0] * l_ij[t, 0, 27] + self.list_r_x_pu[28][0] * l_ij[t, 0, 28]
                                + self.list_r_x_pu[29][0] * l_ij[t, 0, 29] + self.list_r_x_pu[30][0] * l_ij[t, 0, 30]
                                + self.list_r_x_pu[31][0] * l_ij[t, 0, 31])
                      
                for i in range(32):
                    model.addConstr(PP_ij[t, 0, i] == P_ij[t, 0, i] * P_ij[t, 0, i])
                    model.addConstr(QQ_ij[t, 0, i] == Q_ij[t, 0, i] * Q_ij[t, 0, i])

                        
                   
                model.addConstr(-self.list_p_q_pu_t[t + current_time][0][0] ==
                                (P_ij[t, 0, 1] + P_ij[t, 0, 17])
                                - (P_ij[t, 0, 0] - self.list_r_x_pu[0][0] * l_ij[t, 0, 0]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][0][1] ==
                                (Q_ij[t, 0, 1] + Q_ij[t, 0, 17])
                                - (Q_ij[t, 0, 0] - self.list_r_x_pu[0][1] * l_ij[t, 0, 0]))
                   
                model.addConstr(-self.list_p_q_pu_t[t + current_time][1][0] ==
                                (P_ij[t, 0, 2] + P_ij[t, 0, 21])
                                - (P_ij[t, 0, 1] - self.list_r_x_pu[1][0] * l_ij[t, 0, 1]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][1][1] ==
                                (Q_ij[t, 0, 2] + Q_ij[t, 0, 21])
                                - (Q_ij[t, 0, 1] - self.list_r_x_pu[1][1] * l_ij[t, 0, 1]))
                   
                model.addConstr(-self.list_p_q_pu_t[t + current_time][2][0] ==
                                (P_ij[t, 0, 3])
                                - (P_ij[t, 0, 2] - self.list_r_x_pu[2][0] * l_ij[t, 0, 2]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][2][1] ==
                                (Q_ij[t, 0, 3])
                                - (Q_ij[t, 0, 2] - self.list_r_x_pu[2][1] * l_ij[t, 0, 2]))
                   
                model.addConstr(-self.list_p_q_pu_t[t + current_time][3][0] + P_DG_5[t] ==
                                (P_ij[t, 0, 4])
                                - (P_ij[t, 0, 3] - self.list_r_x_pu[3][0] * l_ij[t, 0, 3]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][3][1] ==
                                (Q_ij[t, 0, 4])
                                - (Q_ij[t, 0, 3] - self.list_r_x_pu[3][1] * l_ij[t, 0, 3]))
                   
                model.addConstr(-self.list_p_q_pu_t[t + current_time][4][0] ==
                                (P_ij[t, 0, 5] + P_ij[t, 0, 24])
                                - (P_ij[t, 0, 4] - self.list_r_x_pu[4][0] * l_ij[t, 0, 4]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][4][1] ==
                                (Q_ij[t, 0, 5] + Q_ij[t, 0, 24])
                                - (Q_ij[t, 0, 4] - self.list_r_x_pu[4][1] * l_ij[t, 0, 4]))
                   
                model.addConstr(-self.list_p_q_pu_t[t + current_time][5][0] ==
                                (P_ij[t, 0, 6])
                                - (P_ij[t, 0, 5] - self.list_r_x_pu[5][0] * l_ij[t, 0, 5]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][5][1] + Q_add_7[t] ==
                                (Q_ij[t, 0, 6])
                                - (Q_ij[t, 0, 5] - self.list_r_x_pu[5][1] * l_ij[t, 0, 5]))
                   
                model.addConstr(-self.list_p_q_pu_t[t + current_time][6][0] ==
                                (P_ij[t, 0, 7])
                                - (P_ij[t, 0, 6] - self.list_r_x_pu[6][0] * l_ij[t, 0, 6]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][6][1] ==
                                (Q_ij[t, 0, 7])
                                - (Q_ij[t, 0, 6] - self.list_r_x_pu[6][1] * l_ij[t, 0, 6]))
                   
                model.addConstr(-self.list_p_q_pu_t[t + current_time][7][0] - P_BSS_ch_9[t] * 1.02 + P_BSS_dch_9[t] * 0.98 ==
                                (P_ij[t, 0, 8])
                                - (P_ij[t, 0, 7] - self.list_r_x_pu[7][0] * l_ij[t, 0, 7]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][7][1] ==
                                (Q_ij[t, 0, 8])
                                - (Q_ij[t, 0, 7] - self.list_r_x_pu[7][1] * l_ij[t, 0, 7]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][8][0] ==
                                (P_ij[t, 0, 9])
                                - (P_ij[t, 0, 8] - self.list_r_x_pu[8][0] * l_ij[t, 0, 8]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][8][1] ==
                                (Q_ij[t, 0, 9])
                                - (Q_ij[t, 0, 8] - self.list_r_x_pu[8][1] * l_ij[t, 0, 8]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][9][0] + self.pv_11_avail_list[t + current_time] ==
                                (P_ij[t, 0, 10])
                                - (P_ij[t, 0, 9] - self.list_r_x_pu[9][0] * l_ij[t, 0, 9]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][9][1] + Q_add_11[t] ==
                                (Q_ij[t, 0, 10])
                                - (Q_ij[t, 0, 9] - self.list_r_x_pu[9][1] * l_ij[t, 0, 9]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][10][0] ==
                                (P_ij[t, 0, 11])
                                - (P_ij[t, 0, 10] - self.list_r_x_pu[10][0] * l_ij[t, 0, 10]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][10][1] ==
                                (Q_ij[t, 0, 11])
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
                                - (P_ij[t, 0, 16] - self.list_r_x_pu[16][0] * l_ij[t, 0, 16]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][16][1] ==
                                - (Q_ij[t, 0, 16] - self.list_r_x_pu[16][1] * l_ij[t, 0, 16]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][17][0] ==
                                (P_ij[t, 0, 18])
                                - (P_ij[t, 0, 17] - self.list_r_x_pu[17][0] * l_ij[t, 0, 17]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][17][1] ==
                                (Q_ij[t, 0, 18])
                                - (Q_ij[t, 0, 17] - self.list_r_x_pu[17][1] * l_ij[t, 0, 17]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][18][0] ==
                                (P_ij[t, 0, 19])
                                - (P_ij[t, 0, 18] - self.list_r_x_pu[18][0] * l_ij[t, 0, 18]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][18][1] ==
                                (Q_ij[t, 0, 19])
                                - (Q_ij[t, 0, 18] - self.list_r_x_pu[18][1] * l_ij[t, 0, 18]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][19][0] ==
                                (P_ij[t, 0, 20])
                                - (P_ij[t, 0, 19] - self.list_r_x_pu[19][0] * l_ij[t, 0, 19]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][19][1] ==
                                (Q_ij[t, 0, 20])
                                - (Q_ij[t, 0, 19] - self.list_r_x_pu[19][1] * l_ij[t, 0, 19]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][20][0] ==
                                - (P_ij[t, 0, 20] - self.list_r_x_pu[20][0] * l_ij[t, 0, 20]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][20][1] ==
                                - (Q_ij[t, 0, 20] - self.list_r_x_pu[20][1] * l_ij[t, 0, 20]))
                    
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
                                - (P_ij[t, 0, 23] - self.list_r_x_pu[23][0] * l_ij[t, 0, 23]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][23][1] ==
                                - (Q_ij[t, 0, 23] - self.list_r_x_pu[23][1] * l_ij[t, 0, 23]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][24][0] + self.wind_26_avail_list[t + current_time] ==
                                (P_ij[t, 0, 25])
                                - (P_ij[t, 0, 24] - self.list_r_x_pu[24][0] * l_ij[t, 0, 24]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][24][1] + Q_add_26[t] ==
                                (Q_ij[t, 0, 25])
                                - (Q_ij[t, 0, 24] - self.list_r_x_pu[24][1] * l_ij[t, 0, 24]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][25][0] ==
                                (P_ij[t, 0, 26])
                                - (P_ij[t, 0, 25] - self.list_r_x_pu[25][0] * l_ij[t, 0, 25]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][25][1] ==
                                (Q_ij[t, 0, 26])
                                - (Q_ij[t, 0, 25] - self.list_r_x_pu[25][1] * l_ij[t, 0, 25]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][26][0] ==
                                (P_ij[t, 0, 27])
                                - (P_ij[t, 0, 26] - self.list_r_x_pu[26][0] * l_ij[t, 0, 26]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][26][1] ==
                                (Q_ij[t, 0, 27])
                                - (Q_ij[t, 0, 26] - self.list_r_x_pu[26][1] * l_ij[t, 0, 26]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][27][0] ==
                                (P_ij[t, 0, 28])
                                - (P_ij[t, 0, 27] - self.list_r_x_pu[27][0] * l_ij[t, 0, 27]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][27][1] ==
                                (Q_ij[t, 0, 28])
                                - (Q_ij[t, 0, 27] - self.list_r_x_pu[27][1] * l_ij[t, 0, 27]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][28][0] ==
                                (P_ij[t, 0, 29])
                                - (P_ij[t, 0, 28] - self.list_r_x_pu[28][0] * l_ij[t, 0, 28]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][28][1] ==
                                (Q_ij[t, 0, 29])
                                - (Q_ij[t, 0, 28] - self.list_r_x_pu[28][1] * l_ij[t, 0, 28]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][29][0] ==
                                (P_ij[t, 0, 30])
                                - (P_ij[t, 0, 29] - self.list_r_x_pu[29][0] * l_ij[t, 0, 29]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][29][1] ==
                                (Q_ij[t, 0, 30])
                                - (Q_ij[t, 0, 29] - self.list_r_x_pu[29][1] * l_ij[t, 0, 29]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][30][0] ==
                                (P_ij[t, 0, 31])
                                - (P_ij[t, 0, 30] - self.list_r_x_pu[30][0] * l_ij[t, 0, 30]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][30][1] ==
                                (Q_ij[t, 0, 31])
                                - (Q_ij[t, 0, 30] - self.list_r_x_pu[30][1] * l_ij[t, 0, 30]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][31][0] ==
                                - (P_ij[t, 0, 31] - self.list_r_x_pu[31][0] * l_ij[t, 0, 31]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][31][1] ==
                                - (Q_ij[t, 0, 31] - self.list_r_x_pu[31][1] * l_ij[t, 0, 31]))

                        
                    
                model.addConstr(v_i[t, 0, 0] - v_i[t, 0, 1] - 2 * (
                            self.list_r_x_pu[0][0] * P_ij[t, 0, 0] + self.list_r_x_pu[0][1] * Q_ij[t, 0, 0]) + (
                                        self.list_r_x_pu[0][0] * self.list_r_x_pu[0][0] + self.list_r_x_pu[0][1] *
                                        self.list_r_x_pu[0][1]) * l_ij[t, 0, 0] == 0)
                    
                model.addConstr(v_i[t, 0, 1] - v_i[t, 0, 2] - 2 * (
                            self.list_r_x_pu[1][0] * P_ij[t, 0, 1] + self.list_r_x_pu[1][1] * Q_ij[t, 0, 1]) + (
                                        self.list_r_x_pu[1][0] * self.list_r_x_pu[1][0] + self.list_r_x_pu[1][1] *
                                        self.list_r_x_pu[1][1]) * l_ij[t, 0, 1] == 0)
                    
                model.addConstr(v_i[t, 0, 2] - v_i[t, 0, 3] - 2 * (
                            self.list_r_x_pu[2][0] * P_ij[t, 0, 2] + self.list_r_x_pu[2][1] * Q_ij[t, 0, 2]) + (
                                        self.list_r_x_pu[2][0] * self.list_r_x_pu[2][0] + self.list_r_x_pu[2][1] *
                                        self.list_r_x_pu[2][1]) * l_ij[t, 0, 2] == 0)
                    
                model.addConstr(v_i[t, 0, 3] - v_i[t, 0, 4] - 2 * (
                            self.list_r_x_pu[3][0] * P_ij[t, 0, 3] + self.list_r_x_pu[3][1] * Q_ij[t, 0, 3]) + (
                                        self.list_r_x_pu[3][0] * self.list_r_x_pu[3][0] + self.list_r_x_pu[3][1] *
                                        self.list_r_x_pu[3][1]) * l_ij[t, 0, 3] == 0)
                    
                model.addConstr(v_i[t, 0, 4] - v_i[t, 0, 5] - 2 * (
                            self.list_r_x_pu[4][0] * P_ij[t, 0, 4] + self.list_r_x_pu[4][1] * Q_ij[t, 0, 4]) + (
                                        self.list_r_x_pu[4][0] * self.list_r_x_pu[4][0] + self.list_r_x_pu[4][1] *
                                        self.list_r_x_pu[4][1]) * l_ij[t, 0, 4] == 0)
                    
                model.addConstr(v_i[t, 0, 5] - v_i[t, 0, 6] - 2 * (
                            self.list_r_x_pu[5][0] * P_ij[t, 0, 5] + self.list_r_x_pu[5][1] * Q_ij[t, 0, 5]) + (
                                        self.list_r_x_pu[5][0] * self.list_r_x_pu[5][0] + self.list_r_x_pu[5][1] *
                                        self.list_r_x_pu[5][1]) * l_ij[t, 0, 5] == 0)
                    
                model.addConstr(v_i[t, 0, 6] - v_i[t, 0, 7] - 2 * (
                            self.list_r_x_pu[6][0] * P_ij[t, 0, 6] + self.list_r_x_pu[6][1] * Q_ij[t, 0, 6]) + (
                                        self.list_r_x_pu[6][0] * self.list_r_x_pu[6][0] + self.list_r_x_pu[6][1] *
                                        self.list_r_x_pu[6][1]) * l_ij[t, 0, 6] == 0)
                    
                model.addConstr(v_i[t, 0, 7] - v_i[t, 0, 8] - 2 * (
                            self.list_r_x_pu[7][0] * P_ij[t, 0, 7] + self.list_r_x_pu[7][1] * Q_ij[t, 0, 7]) + (
                                        self.list_r_x_pu[7][0] * self.list_r_x_pu[7][0] + self.list_r_x_pu[7][1] *
                                        self.list_r_x_pu[7][1]) * l_ij[t, 0, 7] == 0)
                    
                model.addConstr(v_i[t, 0, 8] - v_i[t, 0, 9] - 2 * (
                            self.list_r_x_pu[8][0] * P_ij[t, 0, 8] + self.list_r_x_pu[8][1] * Q_ij[t, 0, 8]) + (
                                        self.list_r_x_pu[8][0] * self.list_r_x_pu[8][0] + self.list_r_x_pu[8][1] *
                                        self.list_r_x_pu[8][1]) * l_ij[t, 0, 8] == 0)
                     
                model.addConstr(v_i[t, 0, 9] - v_i[t, 0, 10] - 2 * (
                            self.list_r_x_pu[9][0] * P_ij[t, 0, 9] + self.list_r_x_pu[9][1] * Q_ij[t, 0, 9]) + (
                                        self.list_r_x_pu[9][0] * self.list_r_x_pu[9][0] + self.list_r_x_pu[9][1] *
                                        self.list_r_x_pu[9][1]) * l_ij[t, 0, 9] == 0)
                     
                model.addConstr(v_i[t, 0, 10] - v_i[t, 0, 11] - 2 * (
                        self.list_r_x_pu[10][0] * P_ij[t, 0, 10] + self.list_r_x_pu[10][1] * Q_ij[t, 0, 10]) + (
                                        self.list_r_x_pu[10][0] * self.list_r_x_pu[10][0] + self.list_r_x_pu[10][1] *
                                        self.list_r_x_pu[10][1]) * l_ij[t, 0, 10] == 0)
                     
                model.addConstr(v_i[t, 0, 11] - v_i[t, 0, 12] - 2 * (
                        self.list_r_x_pu[11][0] * P_ij[t, 0, 11] + self.list_r_x_pu[11][1] * Q_ij[t, 0, 11]) + (
                                        self.list_r_x_pu[11][0] * self.list_r_x_pu[11][0] + self.list_r_x_pu[11][1] *
                                        self.list_r_x_pu[11][1]) * l_ij[t, 0, 11] == 0)
                     
                model.addConstr(v_i[t, 0, 12] - v_i[t, 0, 13] - 2 * (
                        self.list_r_x_pu[12][0] * P_ij[t, 0, 12] + self.list_r_x_pu[12][1] * Q_ij[t, 0, 12]) + (
                                        self.list_r_x_pu[12][0] * self.list_r_x_pu[12][0] + self.list_r_x_pu[12][1] *
                                        self.list_r_x_pu[12][1]) * l_ij[t, 0, 12] == 0)
                     
                model.addConstr(v_i[t, 0, 13] - v_i[t, 0, 14] - 2 * (
                        self.list_r_x_pu[13][0] * P_ij[t, 0, 13] + self.list_r_x_pu[13][1] * Q_ij[t, 0, 13]) + (
                                        self.list_r_x_pu[13][0] * self.list_r_x_pu[13][0] + self.list_r_x_pu[13][1] *
                                        self.list_r_x_pu[13][1]) * l_ij[t, 0, 13] == 0)
                     
                model.addConstr(v_i[t, 0, 14] - v_i[t, 0, 15] - 2 * (
                        self.list_r_x_pu[14][0] * P_ij[t, 0, 14] + self.list_r_x_pu[14][1] * Q_ij[t, 0, 14]) + (
                                        self.list_r_x_pu[14][0] * self.list_r_x_pu[14][0] + self.list_r_x_pu[14][1] *
                                        self.list_r_x_pu[14][1]) * l_ij[t, 0, 14] == 0)
                     
                model.addConstr(v_i[t, 0, 15] - v_i[t, 0, 16] - 2 * (
                        self.list_r_x_pu[15][0] * P_ij[t, 0, 15] + self.list_r_x_pu[15][1] * Q_ij[t, 0, 15]) + (
                                        self.list_r_x_pu[15][0] * self.list_r_x_pu[15][0] + self.list_r_x_pu[15][1] *
                                        self.list_r_x_pu[15][1]) * l_ij[t, 0, 15] == 0)
                     
                model.addConstr(v_i[t, 0, 16] - v_i[t, 0, 17] - 2 * (
                        self.list_r_x_pu[16][0] * P_ij[t, 0, 16] + self.list_r_x_pu[16][1] * Q_ij[t, 0, 16]) + (
                                        self.list_r_x_pu[16][0] * self.list_r_x_pu[16][0] + self.list_r_x_pu[16][1] *
                                        self.list_r_x_pu[16][1]) * l_ij[t, 0, 16] == 0)
                     
                model.addConstr(v_i[t, 0, 1] - v_i[t, 0, 18] - 2 * (
                        self.list_r_x_pu[17][0] * P_ij[t, 0, 17] + self.list_r_x_pu[17][1] * Q_ij[t, 0, 17]) + (
                                        self.list_r_x_pu[17][0] * self.list_r_x_pu[17][0] + self.list_r_x_pu[17][1] *
                                        self.list_r_x_pu[17][1]) * l_ij[t, 0, 17] == 0)
                     
                model.addConstr(v_i[t, 0, 18] - v_i[t, 0, 19] - 2 * (
                        self.list_r_x_pu[18][0] * P_ij[t, 0, 18] + self.list_r_x_pu[18][1] * Q_ij[t, 0, 18]) + (
                                        self.list_r_x_pu[18][0] * self.list_r_x_pu[18][0] + self.list_r_x_pu[18][1] *
                                        self.list_r_x_pu[18][1]) * l_ij[t, 0, 18] == 0)
                     
                model.addConstr(v_i[t, 0, 19] - v_i[t, 0, 20] - 2 * (
                        self.list_r_x_pu[19][0] * P_ij[t, 0, 19] + self.list_r_x_pu[19][1] * Q_ij[t, 0, 19]) + (
                                        self.list_r_x_pu[19][0] * self.list_r_x_pu[19][0] + self.list_r_x_pu[19][1] *
                                        self.list_r_x_pu[19][1]) * l_ij[t, 0, 19] == 0)
                     
                model.addConstr(v_i[t, 0, 20] - v_i[t, 0, 21] - 2 * (
                        self.list_r_x_pu[20][0] * P_ij[t, 0, 20] + self.list_r_x_pu[20][1] * Q_ij[t, 0, 20]) + (
                                        self.list_r_x_pu[20][0] * self.list_r_x_pu[20][0] + self.list_r_x_pu[20][1] *
                                        self.list_r_x_pu[20][1]) * l_ij[t, 0, 20] == 0)
                     
                model.addConstr(v_i[t, 0, 2] - v_i[t, 0, 22] - 2 * (
                        self.list_r_x_pu[21][0] * P_ij[t, 0, 21] + self.list_r_x_pu[21][1] * Q_ij[t, 0, 21]) + (
                                        self.list_r_x_pu[21][0] * self.list_r_x_pu[21][0] + self.list_r_x_pu[21][1] *
                                        self.list_r_x_pu[21][1]) * l_ij[t, 0, 21] == 0)
                     
                model.addConstr(v_i[t, 0, 22] - v_i[t, 0, 23] - 2 * (
                        self.list_r_x_pu[22][0] * P_ij[t, 0, 22] + self.list_r_x_pu[22][1] * Q_ij[t, 0, 22]) + (
                                        self.list_r_x_pu[22][0] * self.list_r_x_pu[22][0] + self.list_r_x_pu[22][1] *
                                        self.list_r_x_pu[22][1]) * l_ij[t, 0, 22] == 0)
                     
                model.addConstr(v_i[t, 0, 23] - v_i[t, 0, 24] - 2 * (
                        self.list_r_x_pu[23][0] * P_ij[t, 0, 23] + self.list_r_x_pu[23][1] * Q_ij[t, 0, 23]) + (
                                        self.list_r_x_pu[23][0] * self.list_r_x_pu[23][0] + self.list_r_x_pu[23][1] *
                                        self.list_r_x_pu[23][1]) * l_ij[t, 0, 23] == 0)
                     
                model.addConstr(v_i[t, 0, 5] - v_i[t, 0, 25] - 2 * (
                        self.list_r_x_pu[24][0] * P_ij[t, 0, 24] + self.list_r_x_pu[24][1] * Q_ij[t, 0, 24]) + (
                                        self.list_r_x_pu[24][0] * self.list_r_x_pu[24][0] + self.list_r_x_pu[24][1] *
                                        self.list_r_x_pu[24][1]) * l_ij[t, 0, 24] == 0)
                     
                model.addConstr(v_i[t, 0, 25] - v_i[t, 0, 26] - 2 * (
                        self.list_r_x_pu[25][0] * P_ij[t, 0, 25] + self.list_r_x_pu[25][1] * Q_ij[t, 0, 25]) + (
                                        self.list_r_x_pu[25][0] * self.list_r_x_pu[25][0] + self.list_r_x_pu[25][1] *
                                        self.list_r_x_pu[25][1]) * l_ij[t, 0, 25] == 0)
                     
                model.addConstr(v_i[t, 0, 26] - v_i[t, 0, 27] - 2 * (
                        self.list_r_x_pu[26][0] * P_ij[t, 0, 26] + self.list_r_x_pu[26][1] * Q_ij[t, 0, 26]) + (
                                        self.list_r_x_pu[26][0] * self.list_r_x_pu[26][0] + self.list_r_x_pu[26][1] *
                                        self.list_r_x_pu[26][1]) * l_ij[t, 0, 26] == 0)
                     
                model.addConstr(v_i[t, 0, 27] - v_i[t, 0, 28] - 2 * (
                        self.list_r_x_pu[27][0] * P_ij[t, 0, 27] + self.list_r_x_pu[27][1] * Q_ij[t, 0, 27]) + (
                                        self.list_r_x_pu[27][0] * self.list_r_x_pu[27][0] + self.list_r_x_pu[27][1] *
                                        self.list_r_x_pu[27][1]) * l_ij[t, 0, 27] == 0)
                     
                model.addConstr(v_i[t, 0, 28] - v_i[t, 0, 29] - 2 * (
                        self.list_r_x_pu[28][0] * P_ij[t, 0, 28] + self.list_r_x_pu[28][1] * Q_ij[t, 0, 28]) + (
                                        self.list_r_x_pu[28][0] * self.list_r_x_pu[28][0] + self.list_r_x_pu[28][1] *
                                        self.list_r_x_pu[28][1]) * l_ij[t, 0, 28] == 0)
                     
                model.addConstr(v_i[t, 0, 29] - v_i[t, 0, 30] - 2 * (
                        self.list_r_x_pu[29][0] * P_ij[t, 0, 29] + self.list_r_x_pu[29][1] * Q_ij[t, 0, 29]) + (
                                        self.list_r_x_pu[29][0] * self.list_r_x_pu[29][0] + self.list_r_x_pu[29][1] *
                                        self.list_r_x_pu[29][1]) * l_ij[t, 0, 29] == 0)
                     
                model.addConstr(v_i[t, 0, 30] - v_i[t, 0, 31] - 2 * (
                        self.list_r_x_pu[30][0] * P_ij[t, 0, 30] + self.list_r_x_pu[30][1] * Q_ij[t, 0, 30]) + (
                                        self.list_r_x_pu[30][0] * self.list_r_x_pu[30][0] + self.list_r_x_pu[30][1] *
                                        self.list_r_x_pu[30][1]) * l_ij[t, 0, 30] == 0)
                     
                model.addConstr(v_i[t, 0, 31] - v_i[t, 0, 32] - 2 * (
                        self.list_r_x_pu[31][0] * P_ij[t, 0, 31] + self.list_r_x_pu[31][1] * Q_ij[t, 0, 31]) + (
                                        self.list_r_x_pu[31][0] * self.list_r_x_pu[31][0] + self.list_r_x_pu[31][1] *
                                        self.list_r_x_pu[31][1]) * l_ij[t, 0, 31] == 0)

                         
                    
                model.addConstr(lv_ij_i[t, 0, 0] == l_ij[t, 0, 0] * v_i[t, 0, 0])
                model.addConstr(lv_ij_i[t, 0, 0] - PP_ij[t, 0, 0] - QQ_ij[t, 0, 0] == 0)
                    
                model.addConstr(lv_ij_i[t, 0, 1] == l_ij[t, 0, 1] * v_i[t, 0, 1])
                model.addConstr(lv_ij_i[t, 0, 1] - PP_ij[t, 0, 1] - QQ_ij[t, 0, 1] == 0)
                    
                model.addConstr(lv_ij_i[t, 0, 2] == l_ij[t, 0, 2] * v_i[t, 0, 2])
                model.addConstr(lv_ij_i[t, 0, 2] - PP_ij[t, 0, 2] - QQ_ij[t, 0, 2] == 0)
                    
                model.addConstr(lv_ij_i[t, 0, 3] == l_ij[t, 0, 3] * v_i[t, 0, 3])
                model.addConstr(lv_ij_i[t, 0, 3] - PP_ij[t, 0, 3] - QQ_ij[t, 0, 3] == 0)
                    
                model.addConstr(lv_ij_i[t, 0, 4] == l_ij[t, 0, 4] * v_i[t, 0, 4])
                model.addConstr(lv_ij_i[t, 0, 4] - PP_ij[t, 0, 4] - QQ_ij[t, 0, 4] == 0)
                    
                model.addConstr(lv_ij_i[t, 0, 5] == l_ij[t, 0, 5] * v_i[t, 0, 5])
                model.addConstr(lv_ij_i[t, 0, 5] - PP_ij[t, 0, 5] - QQ_ij[t, 0, 5] == 0)
                    
                model.addConstr(lv_ij_i[t, 0, 6] == l_ij[t, 0, 6] * v_i[t, 0, 6])
                model.addConstr(lv_ij_i[t, 0, 6] - PP_ij[t, 0, 6] - QQ_ij[t, 0, 6] == 0)
                    
                model.addConstr(lv_ij_i[t, 0, 7] == l_ij[t, 0, 7] * v_i[t, 0, 7])
                model.addConstr(lv_ij_i[t, 0, 7] - PP_ij[t, 0, 7] - QQ_ij[t, 0, 7] == 0)
                    
                model.addConstr(lv_ij_i[t, 0, 8] == l_ij[t, 0, 8] * v_i[t, 0, 8])
                model.addConstr(lv_ij_i[t, 0, 8] - PP_ij[t, 0, 8] - QQ_ij[t, 0, 8] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 9] == l_ij[t, 0, 9] * v_i[t, 0, 9])
                model.addConstr(lv_ij_i[t, 0, 9] - PP_ij[t, 0, 9] - QQ_ij[t, 0, 9] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 10] == l_ij[t, 0, 10] * v_i[t, 0, 10])
                model.addConstr(lv_ij_i[t, 0, 10] - PP_ij[t, 0, 10] - QQ_ij[t, 0, 10] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 11] == l_ij[t, 0, 11] * v_i[t, 0, 11])
                model.addConstr(lv_ij_i[t, 0, 11] - PP_ij[t, 0, 11] - QQ_ij[t, 0, 11] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 12] == l_ij[t, 0, 12] * v_i[t, 0, 12])
                model.addConstr(lv_ij_i[t, 0, 12] - PP_ij[t, 0, 12] - QQ_ij[t, 0, 12] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 13] == l_ij[t, 0, 13] * v_i[t, 0, 13])
                model.addConstr(lv_ij_i[t, 0, 13] - PP_ij[t, 0, 13] - QQ_ij[t, 0, 13] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 14] == l_ij[t, 0, 14] * v_i[t, 0, 14])
                model.addConstr(lv_ij_i[t, 0, 14] - PP_ij[t, 0, 14] - QQ_ij[t, 0, 14] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 15] == l_ij[t, 0, 15] * v_i[t, 0, 15])
                model.addConstr(lv_ij_i[t, 0, 15] - PP_ij[t, 0, 15] - QQ_ij[t, 0, 15] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 16] == l_ij[t, 0, 16] * v_i[t, 0, 16])
                model.addConstr(lv_ij_i[t, 0, 16] - PP_ij[t, 0, 16] - QQ_ij[t, 0, 16] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 17] == l_ij[t, 0, 17] * v_i[t, 0, 1])
                model.addConstr(lv_ij_i[t, 0, 17] - PP_ij[t, 0, 17] - QQ_ij[t, 0, 17] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 18] == l_ij[t, 0, 18] * v_i[t, 0, 18])
                model.addConstr(lv_ij_i[t, 0, 18] - PP_ij[t, 0, 18] - QQ_ij[t, 0, 18] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 19] == l_ij[t, 0, 19] * v_i[t, 0, 19])
                model.addConstr(lv_ij_i[t, 0, 19] - PP_ij[t, 0, 19] - QQ_ij[t, 0, 19] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 20] == l_ij[t, 0, 20] * v_i[t, 0, 20])
                model.addConstr(lv_ij_i[t, 0, 20] - PP_ij[t, 0, 20] - QQ_ij[t, 0, 20] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 21] == l_ij[t, 0, 21] * v_i[t, 0, 2])
                model.addConstr(lv_ij_i[t, 0, 21] - PP_ij[t, 0, 21] - QQ_ij[t, 0, 21] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 22] == l_ij[t, 0, 22] * v_i[t, 0, 22])
                model.addConstr(lv_ij_i[t, 0, 22] - PP_ij[t, 0, 22] - QQ_ij[t, 0, 22] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 23] == l_ij[t, 0, 23] * v_i[t, 0, 23])
                model.addConstr(lv_ij_i[t, 0, 23] - PP_ij[t, 0, 23] - QQ_ij[t, 0, 23] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 24] == l_ij[t, 0, 24] * v_i[t, 0, 5])
                model.addConstr(lv_ij_i[t, 0, 24] - PP_ij[t, 0, 24] - QQ_ij[t, 0, 24] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 25] == l_ij[t, 0, 25] * v_i[t, 0, 25])
                model.addConstr(lv_ij_i[t, 0, 25] - PP_ij[t, 0, 25] - QQ_ij[t, 0, 25] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 26] == l_ij[t, 0, 26] * v_i[t, 0, 26])
                model.addConstr(lv_ij_i[t, 0, 26] - PP_ij[t, 0, 26] - QQ_ij[t, 0, 26] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 27] == l_ij[t, 0, 27] * v_i[t, 0, 27])
                model.addConstr(lv_ij_i[t, 0, 27] - PP_ij[t, 0, 27] - QQ_ij[t, 0, 27] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 28] == l_ij[t, 0, 28] * v_i[t, 0, 28])
                model.addConstr(lv_ij_i[t, 0, 28] - PP_ij[t, 0, 28] - QQ_ij[t, 0, 28] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 29] == l_ij[t, 0, 29] * v_i[t, 0, 29])
                model.addConstr(lv_ij_i[t, 0, 29] - PP_ij[t, 0, 29] - QQ_ij[t, 0, 29] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 30] == l_ij[t, 0, 30] * v_i[t, 0, 30])
                model.addConstr(lv_ij_i[t, 0, 30] - PP_ij[t, 0, 30] - QQ_ij[t, 0, 30] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 31] == l_ij[t, 0, 31] * v_i[t, 0, 31])
                model.addConstr(lv_ij_i[t, 0, 31] - PP_ij[t, 0, 31] - QQ_ij[t, 0, 31] == 0)
               
                
               

                
                                                     
            model.setParam('outPutFlag', 0)           
            model.setParam(GRB.Param.TimeLimit, 300)
            model.Params.MIPGap = 0.001
            model.optimize()

                     
                                          

                
            print('obj=', model.objVal)
            obj=model.objval
            dict_value=dict()
            for v in model.getVars():
                                            
                dict_value[v.varName]=v.x
            dict_plot['P_DG_5'].append(dict_value['P_DG_5[0]'])
            dict_plot['SOC_BSS_9'].append(dict_value['SOC_BSS_9[0]'])
            dict_plot['Loss'].append(dict_value['Loss[0]'])
            dict_plot['P_BSS'].append(- dict_value['P_BSS_ch_9[0]'] * 1.02 + dict_value['P_BSS_dch_9[0]'] * 0.98)
            dict_plot['P_G'].append(dict_value['P_ij[0,0,0]'])

        except GurobiError as e:
            print('Error code ' + str(e.errno) + ':' + str(e))

        except AttributeError:
            print('Encountered an attribute error')

        return obj, dict_value['P_DG_5[0]'], dict_value['SOC_BSS_9[0]'], dict_value['cost_hour[0]'], model.Runtime

    def sol_pro_remainder(self, current_time, P_DG_5_init, SOC_BSS_9_init, load_list, pv_11_avail_list,
                          wind_26_avail_list, window_time, load_list_Q):

                 
        self.load_list = load_list
        self.load_list_Q = load_list_Q
        self.pv_11_avail_list = pv_11_avail_list
        self.wind_26_avail_list = wind_26_avail_list
        self.list_p_q_pu_t = self._get_pq_t()

        obj = "error"
        try:
                
            model = Model('Pure MPC')
                                            
                                          
                              
            model.setParam('MIPFocus',0)

                  
                                  
            P_ij = model.addMVar(shape=(window_time, 1, 32), lb=-100, ub=100, vtype=GRB.CONTINUOUS, name='P_ij')
            Q_ij = model.addMVar(shape=(window_time, 1, 32), lb=-100, ub=100, vtype=GRB.CONTINUOUS, name='Q_ij')
            l_ij = model.addMVar(shape=(window_time, 1, 32), lb=0, vtype=GRB.CONTINUOUS, name='l_ij')
            v_i = model.addMVar(shape=(window_time, 1, 33), lb=0.94 * 0.94, ub=1.06 * 1.06, vtype=GRB.CONTINUOUS, name='v_i')
            lv_ij_i = model.addMVar(shape=(window_time, 1, 32), vtype=GRB.CONTINUOUS, name='lv_ij_i')
            PP_ij = model.addMVar(shape=(window_time, 1, 32), vtype=GRB.CONTINUOUS, name='PP_ij')
            QQ_ij = model.addMVar(shape=(window_time, 1, 32), vtype=GRB.CONTINUOUS, name='QQ_ij')

                  
            Loss = model.addMVar(shape=(window_time,), lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='Loss')

                     
            Q_add_7 = model.addMVar(shape=(window_time,), lb=-0.3, ub=0.3, vtype=GRB.CONTINUOUS, name='Q_add_7')
            Q_add_11 = model.addMVar(shape=(window_time,), vtype=GRB.CONTINUOUS, name='Q_add_11')
            Q_add_26 = model.addMVar(shape=(window_time,), vtype=GRB.CONTINUOUS, name='Q_add_26')

                            
            P_DG_5 = model.addMVar(shape=(window_time,), lb=self.gen_min, ub=self.gen_max, vtype=GRB.CONTINUOUS, name='P_DG_5')
            SOC_BSS_9 = model.addMVar(shape=(window_time,), lb= self.SOC_min, ub=self.SOC_max, vtype=GRB.CONTINUOUS, name='SOC_BSS_9')
                   
                                                                                           
                                                                                                     
            P_BSS_ch_9 = model.addMVar(shape=(window_time,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS,
                                       name='P_BSS_ch_9')
            P_BSS_dch_9 = model.addMVar(shape=(window_time,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS,
                                        name='P_BSS_dch_9')

                
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
                model.addConstr(cost_genera[t] == (self.a_5_2 * P_DG_5[t] * P_DG_5[t] + self.a_5_1 * P_DG_5[t] + self.a_5_0) / 2)
               
                
               

            for t in range(window_time):
                model.addConstr(self.pv_11_avail_list[t + current_time] * self.pv_11_avail_list[t + current_time]
                                + Q_add_11[t] * Q_add_11[t] <= self.pv_11_max * self.pv_11_max)
                model.addConstr(self.wind_26_avail_list[t + current_time] * self.wind_26_avail_list[t + current_time]
                                + Q_add_26[t] * Q_add_26[t] <= self.wind_26_max * self.wind_26_max)
               
                   
               

            model.addConstr(P_DG_5[0] <= P_DG_5_init + self.gen_ramp / 2)
            model.addConstr(P_DG_5[0] >= P_DG_5_init - self.gen_ramp / 2)
            if window_time > 1:
                for t in range(window_time - 1):
                    model.addConstr(P_DG_5[t + 1] <= P_DG_5[t] + self.gen_ramp / 2)
                    model.addConstr(P_DG_5[t + 1] >= P_DG_5[t] - self.gen_ramp / 2)
               
                   
                                    
               

                   
            model.addConstr(SOC_BSS_9[0] == SOC_BSS_9_init + (P_BSS_ch_9[0] - P_BSS_dch_9[0]) / 2)
            if window_time > 1:
                for t in range(window_time - 1):
                    model.addConstr(SOC_BSS_9[t + 1] == SOC_BSS_9[t] + (P_BSS_ch_9[t + 1] - P_BSS_dch_9[t + 1]) / 2)
                                          
                                                                                 
                                                                                        

               
                  
                                  
               

            for t in range(window_time):
                      
                model.addConstr(v_i[t, 0, 0] == 1)

                      
                model.addConstr(Loss[t] == self.list_r_x_pu[0][0] * l_ij[t, 0, 0]
                    + self.list_r_x_pu[1][0] * l_ij[t, 0, 1] + self.list_r_x_pu[2][0] * l_ij[t, 0, 2]
                    + self.list_r_x_pu[3][0] * l_ij[t, 0, 3] + self.list_r_x_pu[4][0] * l_ij[t, 0, 4]
                    + self.list_r_x_pu[5][0] * l_ij[t, 0, 5] + self.list_r_x_pu[6][0] * l_ij[t, 0, 6]
                    + self.list_r_x_pu[7][0] * l_ij[t, 0, 7]  + self.list_r_x_pu[8][0] * l_ij[t, 0, 8]
                    + self.list_r_x_pu[9][0] * l_ij[t, 0, 9]  + self.list_r_x_pu[10][0] * l_ij[t, 0, 10]
                    + self.list_r_x_pu[11][0] * l_ij[t, 0, 11]  + self.list_r_x_pu[12][0] * l_ij[t, 0, 12]
                    + self.list_r_x_pu[13][0] * l_ij[t, 0, 13]  + self.list_r_x_pu[14][0] * l_ij[t, 0, 14]
                    + self.list_r_x_pu[15][0] * l_ij[t, 0, 15]  + self.list_r_x_pu[16][0] * l_ij[t, 0, 16]
                    + self.list_r_x_pu[17][0] * l_ij[t, 0, 17]  + self.list_r_x_pu[18][0] * l_ij[t, 0, 18]
                    + self.list_r_x_pu[19][0] * l_ij[t, 0, 19]  + self.list_r_x_pu[20][0] * l_ij[t, 0, 20]
                    + self.list_r_x_pu[21][0] * l_ij[t, 0, 21]  + self.list_r_x_pu[22][0] * l_ij[t, 0, 22]
                    + self.list_r_x_pu[23][0] * l_ij[t, 0, 23]  + self.list_r_x_pu[24][0] * l_ij[t, 0, 24]
                    + self.list_r_x_pu[25][0] * l_ij[t, 0, 25] + self.list_r_x_pu[26][0] * l_ij[t, 0, 26]
                    + self.list_r_x_pu[27][0] * l_ij[t, 0, 27] + self.list_r_x_pu[28][0] * l_ij[t, 0, 28]
                    + self.list_r_x_pu[29][0] * l_ij[t, 0, 29] + self.list_r_x_pu[30][0] * l_ij[t, 0, 30]
                    + self.list_r_x_pu[31][0] * l_ij[t, 0, 31])

                      
                for i in range(32):
                    model.addConstr(PP_ij[t, 0, i] == P_ij[t, 0, i] * P_ij[t, 0, i])
                    model.addConstr(QQ_ij[t, 0, i] == Q_ij[t, 0, i] * Q_ij[t, 0, i])

                        
                   
                model.addConstr(-self.list_p_q_pu_t[t + current_time][0][0] ==
                                (P_ij[t, 0, 1] + P_ij[t, 0, 17])
                                - (P_ij[t, 0, 0] - self.list_r_x_pu[0][0] * l_ij[t, 0, 0]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][0][1] ==
                                (Q_ij[t, 0, 1] + Q_ij[t, 0, 17])
                            - (Q_ij[t, 0, 0] - self.list_r_x_pu[0][1] * l_ij[t, 0, 0]))
                   
                model.addConstr(-self.list_p_q_pu_t[t + current_time][1][0] ==
                                (P_ij[t, 0, 2] + P_ij[t, 0, 21])
                                - (P_ij[t, 0, 1] - self.list_r_x_pu[1][0] * l_ij[t, 0, 1]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][1][1] ==
                                (Q_ij[t, 0, 2] + Q_ij[t, 0, 21])
                                - (Q_ij[t, 0, 1] - self.list_r_x_pu[1][1] * l_ij[t, 0, 1]))
                   
                model.addConstr(-self.list_p_q_pu_t[t + current_time][2][0] ==
                                (P_ij[t, 0, 3])
                                - (P_ij[t, 0, 2] - self.list_r_x_pu[2][0] * l_ij[t, 0, 2]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][2][1] ==
                                (Q_ij[t, 0, 3])
                                - (Q_ij[t, 0, 2] - self.list_r_x_pu[2][1] * l_ij[t, 0, 2]))
                   
                model.addConstr(-self.list_p_q_pu_t[t + current_time][3][0] + P_DG_5[t] ==
                                (P_ij[t, 0, 4])
                                - (P_ij[t, 0, 3] - self.list_r_x_pu[3][0] * l_ij[t, 0, 3]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][3][1] ==
                                (Q_ij[t, 0, 4])
                                - (Q_ij[t, 0, 3] - self.list_r_x_pu[3][1] * l_ij[t, 0, 3]))
                   
                model.addConstr(-self.list_p_q_pu_t[t + current_time][4][0] ==
                                (P_ij[t, 0, 5] + P_ij[t, 0, 24])
                                - (P_ij[t, 0, 4] - self.list_r_x_pu[4][0] * l_ij[t, 0, 4]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][4][1] ==
                                (Q_ij[t, 0, 5] + Q_ij[t, 0, 24])
                                - (Q_ij[t, 0, 4] - self.list_r_x_pu[4][1] * l_ij[t, 0, 4]))
                   
                model.addConstr(-self.list_p_q_pu_t[t + current_time][5][0] ==
                                (P_ij[t, 0, 6])
                                - (P_ij[t, 0, 5] - self.list_r_x_pu[5][0] * l_ij[t, 0, 5]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][5][1] + Q_add_7[t] ==
                                (Q_ij[t, 0, 6])
                                - (Q_ij[t, 0, 5] - self.list_r_x_pu[5][1] * l_ij[t, 0, 5]))
                   
                model.addConstr(-self.list_p_q_pu_t[t + current_time][6][0] ==
                                (P_ij[t, 0, 7])
                                - (P_ij[t, 0, 6] - self.list_r_x_pu[6][0] * l_ij[t, 0, 6]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][6][1] ==
                                (Q_ij[t, 0, 7])
                                - (Q_ij[t, 0, 6] - self.list_r_x_pu[6][1] * l_ij[t, 0, 6]))
                   
                model.addConstr(-self.list_p_q_pu_t[t + current_time][7][0] - P_BSS_ch_9[t] * 1.02 + P_BSS_dch_9[t] * 0.98 ==
                                (P_ij[t, 0, 8])
                                - (P_ij[t, 0, 7] - self.list_r_x_pu[7][0] * l_ij[t, 0, 7]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][7][1] ==
                                (Q_ij[t, 0, 8])
                                - (Q_ij[t, 0, 7] - self.list_r_x_pu[7][1] * l_ij[t, 0, 7]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][8][0] ==
                                (P_ij[t, 0, 9])
                                - (P_ij[t, 0, 8] - self.list_r_x_pu[8][0] * l_ij[t, 0, 8]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][8][1] ==
                                (Q_ij[t, 0, 9])
                                - (Q_ij[t, 0, 8] - self.list_r_x_pu[8][1] * l_ij[t, 0, 8]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][9][0] + self.pv_11_avail_list[t + current_time] ==
                                (P_ij[t, 0, 10])
                                - (P_ij[t, 0, 9] - self.list_r_x_pu[9][0] * l_ij[t, 0, 9]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][9][1] + Q_add_11[t] ==
                                (Q_ij[t, 0, 10])
                                - (Q_ij[t, 0, 9] - self.list_r_x_pu[9][1] * l_ij[t, 0, 9]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][10][0] ==
                                (P_ij[t, 0, 11])
                                - (P_ij[t, 0, 10] - self.list_r_x_pu[10][0] * l_ij[t, 0, 10]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][10][1] ==
                                (Q_ij[t, 0, 11])
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
                                - (P_ij[t, 0, 16] - self.list_r_x_pu[16][0] * l_ij[t, 0, 16]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][16][1] ==
                                - (Q_ij[t, 0, 16] - self.list_r_x_pu[16][1] * l_ij[t, 0, 16]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][17][0] ==
                                (P_ij[t, 0, 18])
                                - (P_ij[t, 0, 17] - self.list_r_x_pu[17][0] * l_ij[t, 0, 17]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][17][1] ==
                                (Q_ij[t, 0, 18])
                                - (Q_ij[t, 0, 17] - self.list_r_x_pu[17][1] * l_ij[t, 0, 17]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][18][0] ==
                                (P_ij[t, 0, 19])
                                - (P_ij[t, 0, 18] - self.list_r_x_pu[18][0] * l_ij[t, 0, 18]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][18][1] ==
                                (Q_ij[t, 0, 19])
                                - (Q_ij[t, 0, 18] - self.list_r_x_pu[18][1] * l_ij[t, 0, 18]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][19][0] ==
                                (P_ij[t, 0, 20])
                                - (P_ij[t, 0, 19] - self.list_r_x_pu[19][0] * l_ij[t, 0, 19]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][19][1] ==
                                (Q_ij[t, 0, 20])
                                - (Q_ij[t, 0, 19] - self.list_r_x_pu[19][1] * l_ij[t, 0, 19]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][20][0] ==
                                - (P_ij[t, 0, 20] - self.list_r_x_pu[20][0] * l_ij[t, 0, 20]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][20][1] ==
                                - (Q_ij[t, 0, 20] - self.list_r_x_pu[20][1] * l_ij[t, 0, 20]))
                    
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
                                - (P_ij[t, 0, 23] - self.list_r_x_pu[23][0] * l_ij[t, 0, 23]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][23][1] ==
                                - (Q_ij[t, 0, 23] - self.list_r_x_pu[23][1] * l_ij[t, 0, 23]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][24][0] + self.wind_26_avail_list[t + current_time]  ==
                                (P_ij[t, 0, 25])
                                - (P_ij[t, 0, 24] - self.list_r_x_pu[24][0] * l_ij[t, 0, 24]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][24][1] + Q_add_26[t] ==
                                (Q_ij[t, 0, 25])
                                - (Q_ij[t, 0, 24] - self.list_r_x_pu[24][1] * l_ij[t, 0, 24]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][25][0] ==
                                (P_ij[t, 0, 26])
                                - (P_ij[t, 0, 25] - self.list_r_x_pu[25][0] * l_ij[t, 0, 25]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][25][1] ==
                                (Q_ij[t, 0, 26])
                                - (Q_ij[t, 0, 25] - self.list_r_x_pu[25][1] * l_ij[t, 0, 25]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][26][0] ==
                                (P_ij[t, 0, 27])
                                - (P_ij[t, 0, 26] - self.list_r_x_pu[26][0] * l_ij[t, 0, 26]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][26][1] ==
                                (Q_ij[t, 0, 27])
                                - (Q_ij[t, 0, 26] - self.list_r_x_pu[26][1] * l_ij[t, 0, 26]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][27][0] ==
                                (P_ij[t, 0, 28])
                                - (P_ij[t, 0, 27] - self.list_r_x_pu[27][0] * l_ij[t, 0, 27]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][27][1] ==
                                (Q_ij[t, 0, 28])
                                - (Q_ij[t, 0, 27] - self.list_r_x_pu[27][1] * l_ij[t, 0, 27]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][28][0] ==
                                (P_ij[t, 0, 29])
                                - (P_ij[t, 0, 28] - self.list_r_x_pu[28][0] * l_ij[t, 0, 28]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][28][1] ==
                                (Q_ij[t, 0, 29])
                                - (Q_ij[t, 0, 28] - self.list_r_x_pu[28][1] * l_ij[t, 0, 28]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][29][0] ==
                                (P_ij[t, 0, 30])
                                - (P_ij[t, 0, 29] - self.list_r_x_pu[29][0] * l_ij[t, 0, 29]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][29][1] ==
                                (Q_ij[t, 0, 30])
                                - (Q_ij[t, 0, 29] - self.list_r_x_pu[29][1] * l_ij[t, 0, 29]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][30][0] ==
                                (P_ij[t, 0, 31])
                                - (P_ij[t, 0, 30] - self.list_r_x_pu[30][0] * l_ij[t, 0, 30]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][30][1] ==
                                (Q_ij[t, 0, 31])
                                - (Q_ij[t, 0, 30] - self.list_r_x_pu[30][1] * l_ij[t, 0, 30]))
                    
                model.addConstr(-self.list_p_q_pu_t[t + current_time][31][0] ==
                                - (P_ij[t, 0, 31] - self.list_r_x_pu[31][0] * l_ij[t, 0, 31]))
                model.addConstr(-self.list_p_q_pu_t[t + current_time][31][1] ==
                                - (Q_ij[t, 0, 31] - self.list_r_x_pu[31][1] * l_ij[t, 0, 31]))

                        
                    
                model.addConstr(v_i[t, 0, 0] - v_i[t, 0, 1] - 2 * (self.list_r_x_pu[0][0] * P_ij[t, 0, 0] + self.list_r_x_pu[0][1] * Q_ij[t, 0, 0]) + (
                                self.list_r_x_pu[0][0] * self.list_r_x_pu[0][0] + self.list_r_x_pu[0][1] * self.list_r_x_pu[0][1]) * l_ij[t, 0, 0] == 0)
                    
                model.addConstr(v_i[t, 0, 1] - v_i[t, 0, 2] - 2 * (self.list_r_x_pu[1][0] * P_ij[t, 0, 1] + self.list_r_x_pu[1][1] * Q_ij[t, 0, 1]) + (
                                self.list_r_x_pu[1][0] * self.list_r_x_pu[1][0] + self.list_r_x_pu[1][1] * self.list_r_x_pu[1][1]) * l_ij[t, 0, 1] == 0)
                    
                model.addConstr(v_i[t, 0, 2] - v_i[t, 0, 3] - 2 * (self.list_r_x_pu[2][0] * P_ij[t, 0, 2] + self.list_r_x_pu[2][1] * Q_ij[t, 0, 2]) + (
                                self.list_r_x_pu[2][0] * self.list_r_x_pu[2][0] + self.list_r_x_pu[2][1] * self.list_r_x_pu[2][1]) * l_ij[t, 0, 2] == 0)
                    
                model.addConstr(v_i[t, 0, 3] - v_i[t, 0, 4] - 2 * (self.list_r_x_pu[3][0] * P_ij[t, 0, 3] + self.list_r_x_pu[3][1] * Q_ij[t, 0, 3]) + (
                                self.list_r_x_pu[3][0] * self.list_r_x_pu[3][0] + self.list_r_x_pu[3][1] * self.list_r_x_pu[3][1]) * l_ij[t, 0, 3] == 0)
                    
                model.addConstr(v_i[t, 0, 4] - v_i[t, 0, 5] - 2 * (self.list_r_x_pu[4][0] * P_ij[t, 0, 4] + self.list_r_x_pu[4][1] * Q_ij[t, 0, 4]) + (
                                self.list_r_x_pu[4][0] * self.list_r_x_pu[4][0] + self.list_r_x_pu[4][1] * self.list_r_x_pu[4][1]) * l_ij[t, 0, 4] == 0)
                    
                model.addConstr(v_i[t, 0, 5] - v_i[t, 0, 6] - 2 * (self.list_r_x_pu[5][0] * P_ij[t, 0, 5] + self.list_r_x_pu[5][1] * Q_ij[t, 0, 5]) + (
                                self.list_r_x_pu[5][0] * self.list_r_x_pu[5][0] + self.list_r_x_pu[5][1] * self.list_r_x_pu[5][1]) * l_ij[t, 0, 5] == 0)
                    
                model.addConstr(v_i[t, 0, 6] - v_i[t, 0, 7] - 2 * (self.list_r_x_pu[6][0] * P_ij[t, 0, 6] + self.list_r_x_pu[6][1] * Q_ij[t, 0, 6]) + (
                                self.list_r_x_pu[6][0] * self.list_r_x_pu[6][0] + self.list_r_x_pu[6][1] * self.list_r_x_pu[6][1]) * l_ij[t, 0, 6] == 0)
                    
                model.addConstr(v_i[t, 0, 7] - v_i[t, 0, 8] - 2 * (self.list_r_x_pu[7][0] * P_ij[t, 0, 7] + self.list_r_x_pu[7][1] * Q_ij[t, 0, 7]) + (
                                self.list_r_x_pu[7][0] * self.list_r_x_pu[7][0] + self.list_r_x_pu[7][1] * self.list_r_x_pu[7][1]) * l_ij[t, 0, 7] == 0)
                    
                model.addConstr(v_i[t, 0, 8] - v_i[t, 0, 9] - 2 * (self.list_r_x_pu[8][0] * P_ij[t, 0, 8] + self.list_r_x_pu[8][1] * Q_ij[t, 0, 8]) + (
                                self.list_r_x_pu[8][0] * self.list_r_x_pu[8][0] + self.list_r_x_pu[8][1] * self.list_r_x_pu[8][1]) * l_ij[t, 0, 8] == 0)
                     
                model.addConstr(v_i[t, 0, 9] - v_i[t, 0, 10] - 2 * (self.list_r_x_pu[9][0] * P_ij[t, 0, 9] + self.list_r_x_pu[9][1] * Q_ij[t, 0, 9]) + (
                                self.list_r_x_pu[9][0] * self.list_r_x_pu[9][0] + self.list_r_x_pu[9][1] * self.list_r_x_pu[9][1]) * l_ij[t, 0, 9] == 0)
                     
                model.addConstr(v_i[t, 0, 10] - v_i[t, 0, 11] - 2 * (
                            self.list_r_x_pu[10][0] * P_ij[t, 0, 10] + self.list_r_x_pu[10][1] * Q_ij[t, 0, 10]) + (
                                                              self.list_r_x_pu[10][0] * self.list_r_x_pu[10][0] + self.list_r_x_pu[10][1] *
                                                              self.list_r_x_pu[10][1]) * l_ij[t, 0, 10] == 0)
                     
                model.addConstr(v_i[t, 0, 11] - v_i[t, 0, 12] - 2 * (
                            self.list_r_x_pu[11][0] * P_ij[t, 0, 11] + self.list_r_x_pu[11][1] * Q_ij[t, 0, 11]) + (
                                                              self.list_r_x_pu[11][0] * self.list_r_x_pu[11][0] + self.list_r_x_pu[11][1] *
                                                              self.list_r_x_pu[11][1]) * l_ij[t, 0, 11] == 0)
                     
                model.addConstr(v_i[t, 0, 12] - v_i[t, 0, 13] - 2 * (
                            self.list_r_x_pu[12][0] * P_ij[t, 0, 12] + self.list_r_x_pu[12][1] * Q_ij[t, 0, 12]) + (
                                                              self.list_r_x_pu[12][0] * self.list_r_x_pu[12][0] + self.list_r_x_pu[12][1] *
                                                              self.list_r_x_pu[12][1]) * l_ij[t, 0, 12] == 0)
                     
                model.addConstr(v_i[t, 0, 13] - v_i[t, 0, 14] - 2 * (
                            self.list_r_x_pu[13][0] * P_ij[t, 0, 13] + self.list_r_x_pu[13][1] * Q_ij[t, 0, 13]) + (
                                                              self.list_r_x_pu[13][0] * self.list_r_x_pu[13][0] + self.list_r_x_pu[13][1] *
                                                              self.list_r_x_pu[13][1]) * l_ij[t, 0, 13] == 0)
                     
                model.addConstr(v_i[t, 0, 14] - v_i[t, 0, 15] - 2 * (
                            self.list_r_x_pu[14][0] * P_ij[t, 0, 14] + self.list_r_x_pu[14][1] * Q_ij[t, 0, 14]) + (
                                                              self.list_r_x_pu[14][0] * self.list_r_x_pu[14][0] + self.list_r_x_pu[14][1] *
                                                              self.list_r_x_pu[14][1]) * l_ij[t, 0, 14] == 0)
                     
                model.addConstr(v_i[t, 0, 15] - v_i[t, 0, 16] - 2 * (
                            self.list_r_x_pu[15][0] * P_ij[t, 0, 15] + self.list_r_x_pu[15][1] * Q_ij[t, 0, 15]) + (
                                                              self.list_r_x_pu[15][0] * self.list_r_x_pu[15][0] + self.list_r_x_pu[15][1] *
                                                              self.list_r_x_pu[15][1]) * l_ij[t, 0, 15] == 0)
                     
                model.addConstr(v_i[t, 0, 16] - v_i[t, 0, 17] - 2 * (
                            self.list_r_x_pu[16][0] * P_ij[t, 0, 16] + self.list_r_x_pu[16][1] * Q_ij[t, 0, 16]) + (
                                                              self.list_r_x_pu[16][0] * self.list_r_x_pu[16][0] + self.list_r_x_pu[16][1] *
                                                              self.list_r_x_pu[16][1]) * l_ij[t, 0, 16] == 0)
                     
                model.addConstr(v_i[t, 0, 1] - v_i[t, 0, 18] - 2 * (
                            self.list_r_x_pu[17][0] * P_ij[t, 0, 17] + self.list_r_x_pu[17][1] * Q_ij[t, 0, 17]) + (
                                                              self.list_r_x_pu[17][0] * self.list_r_x_pu[17][0] + self.list_r_x_pu[17][1] *
                                                              self.list_r_x_pu[17][1]) * l_ij[t, 0, 17] == 0)
                     
                model.addConstr(v_i[t, 0, 18] - v_i[t, 0, 19] - 2 * (
                            self.list_r_x_pu[18][0] * P_ij[t, 0, 18] + self.list_r_x_pu[18][1] * Q_ij[t, 0, 18]) + (
                                                              self.list_r_x_pu[18][0] * self.list_r_x_pu[18][0] + self.list_r_x_pu[18][1] *
                                                              self.list_r_x_pu[18][1]) * l_ij[t, 0, 18] == 0)
                     
                model.addConstr(v_i[t, 0, 19] - v_i[t, 0, 20] - 2 * (
                            self.list_r_x_pu[19][0] * P_ij[t, 0, 19] + self.list_r_x_pu[19][1] * Q_ij[t, 0, 19]) + (
                                                              self.list_r_x_pu[19][0] * self.list_r_x_pu[19][0] + self.list_r_x_pu[19][1] *
                                                              self.list_r_x_pu[19][1]) * l_ij[t, 0, 19] == 0)
                     
                model.addConstr(v_i[t, 0, 20] - v_i[t, 0, 21] - 2 * (
                            self.list_r_x_pu[20][0] * P_ij[t, 0, 20] + self.list_r_x_pu[20][1] * Q_ij[t, 0, 20]) + (
                                                              self.list_r_x_pu[20][0] * self.list_r_x_pu[20][0] + self.list_r_x_pu[20][1] *
                                                              self.list_r_x_pu[20][1]) * l_ij[t, 0, 20] == 0)
                     
                model.addConstr(v_i[t, 0, 2] - v_i[t, 0, 22] - 2 * (
                            self.list_r_x_pu[21][0] * P_ij[t, 0, 21] + self.list_r_x_pu[21][1] * Q_ij[t, 0, 21]) + (
                                                              self.list_r_x_pu[21][0] * self.list_r_x_pu[21][0] + self.list_r_x_pu[21][1] *
                                                              self.list_r_x_pu[21][1]) * l_ij[t, 0, 21] == 0)
                     
                model.addConstr(v_i[t, 0, 22] - v_i[t, 0, 23] - 2 * (
                            self.list_r_x_pu[22][0] * P_ij[t, 0, 22] + self.list_r_x_pu[22][1] * Q_ij[t, 0, 22]) + (
                                                              self.list_r_x_pu[22][0] * self.list_r_x_pu[22][0] + self.list_r_x_pu[22][1] *
                                                              self.list_r_x_pu[22][1]) * l_ij[t, 0, 22] == 0)
                     
                model.addConstr(v_i[t, 0, 23] - v_i[t, 0, 24] - 2 * (
                            self.list_r_x_pu[23][0] * P_ij[t, 0, 23] + self.list_r_x_pu[23][1] * Q_ij[t, 0, 23]) + (
                                                              self.list_r_x_pu[23][0] * self.list_r_x_pu[23][0] + self.list_r_x_pu[23][1] *
                                                              self.list_r_x_pu[23][1]) * l_ij[t, 0, 23] == 0)
                     
                model.addConstr(v_i[t, 0, 5] - v_i[t, 0, 25] - 2 * (
                            self.list_r_x_pu[24][0] * P_ij[t, 0, 24] + self.list_r_x_pu[24][1] * Q_ij[t, 0, 24]) + (
                                                              self.list_r_x_pu[24][0] * self.list_r_x_pu[24][0] + self.list_r_x_pu[24][1] *
                                                              self.list_r_x_pu[24][1]) * l_ij[t, 0, 24] == 0)
                     
                model.addConstr(v_i[t, 0, 25] - v_i[t, 0, 26] - 2 * (
                            self.list_r_x_pu[25][0] * P_ij[t, 0, 25] + self.list_r_x_pu[25][1] * Q_ij[t, 0, 25]) + (
                                                              self.list_r_x_pu[25][0] * self.list_r_x_pu[25][0] + self.list_r_x_pu[25][1] *
                                                              self.list_r_x_pu[25][1]) * l_ij[t, 0, 25] == 0)
                     
                model.addConstr(v_i[t, 0, 26] - v_i[t, 0, 27] - 2 * (
                            self.list_r_x_pu[26][0] * P_ij[t, 0, 26] + self.list_r_x_pu[26][1] * Q_ij[t, 0, 26]) + (
                                                              self.list_r_x_pu[26][0] * self.list_r_x_pu[26][0] + self.list_r_x_pu[26][1] *
                                                              self.list_r_x_pu[26][1]) * l_ij[t, 0, 26] == 0)
                     
                model.addConstr(v_i[t, 0, 27] - v_i[t, 0, 28] - 2 * (
                            self.list_r_x_pu[27][0] * P_ij[t, 0, 27] + self.list_r_x_pu[27][1] * Q_ij[t, 0, 27]) + (
                                                              self.list_r_x_pu[27][0] * self.list_r_x_pu[27][0] + self.list_r_x_pu[27][1] *
                                                              self.list_r_x_pu[27][1]) * l_ij[t, 0, 27] == 0)
                     
                model.addConstr(v_i[t, 0, 28] - v_i[t, 0, 29] - 2 * (
                            self.list_r_x_pu[28][0] * P_ij[t, 0, 28] + self.list_r_x_pu[28][1] * Q_ij[t, 0, 28]) + (
                                                              self.list_r_x_pu[28][0] * self.list_r_x_pu[28][0] + self.list_r_x_pu[28][1] *
                                                              self.list_r_x_pu[28][1]) * l_ij[t, 0, 28] == 0)
                     
                model.addConstr(v_i[t, 0, 29] - v_i[t, 0, 30] - 2 * (
                            self.list_r_x_pu[29][0] * P_ij[t, 0, 29] + self.list_r_x_pu[29][1] * Q_ij[t, 0, 29]) + (
                                                              self.list_r_x_pu[29][0] * self.list_r_x_pu[29][0] + self.list_r_x_pu[29][1] *
                                                              self.list_r_x_pu[29][1]) * l_ij[t, 0, 29] == 0)
                     
                model.addConstr(v_i[t, 0, 30] - v_i[t, 0, 31] - 2 * (
                            self.list_r_x_pu[30][0] * P_ij[t, 0, 30] + self.list_r_x_pu[30][1] * Q_ij[t, 0, 30]) + (
                                                              self.list_r_x_pu[30][0] * self.list_r_x_pu[30][0] + self.list_r_x_pu[30][1] *
                                                              self.list_r_x_pu[30][1]) * l_ij[t, 0, 30] == 0)
                     
                model.addConstr(v_i[t, 0, 31] - v_i[t, 0, 32] - 2 * (
                            self.list_r_x_pu[31][0] * P_ij[t, 0, 31] + self.list_r_x_pu[31][1] * Q_ij[t, 0, 31]) + (
                                                              self.list_r_x_pu[31][0] * self.list_r_x_pu[31][0] + self.list_r_x_pu[31][1] *
                                                              self.list_r_x_pu[31][1]) * l_ij[t, 0, 31] == 0)

                         
                    
                model.addConstr(lv_ij_i[t, 0, 0] == l_ij[t, 0, 0] * v_i[t, 0, 0])
                model.addConstr(lv_ij_i[t, 0, 0] - PP_ij[t, 0, 0] - QQ_ij[t, 0, 0] == 0)
                    
                model.addConstr(lv_ij_i[t, 0, 1] == l_ij[t, 0, 1] * v_i[t, 0, 1])
                model.addConstr(lv_ij_i[t, 0, 1] - PP_ij[t, 0, 1] - QQ_ij[t, 0, 1] == 0)
                    
                model.addConstr(lv_ij_i[t, 0, 2] == l_ij[t, 0, 2] * v_i[t, 0, 2])
                model.addConstr(lv_ij_i[t, 0, 2] - PP_ij[t, 0, 2] - QQ_ij[t, 0, 2] == 0)
                    
                model.addConstr(lv_ij_i[t, 0, 3] == l_ij[t, 0, 3] * v_i[t, 0, 3])
                model.addConstr(lv_ij_i[t, 0, 3] - PP_ij[t, 0, 3] - QQ_ij[t, 0, 3] == 0)
                    
                model.addConstr(lv_ij_i[t, 0, 4] == l_ij[t, 0, 4] * v_i[t, 0, 4])
                model.addConstr(lv_ij_i[t, 0, 4] - PP_ij[t, 0, 4] - QQ_ij[t, 0, 4] == 0)
                    
                model.addConstr(lv_ij_i[t, 0, 5] == l_ij[t, 0, 5] * v_i[t, 0, 5])
                model.addConstr(lv_ij_i[t, 0, 5] - PP_ij[t, 0, 5] - QQ_ij[t, 0, 5] == 0)
                    
                model.addConstr(lv_ij_i[t, 0, 6] == l_ij[t, 0, 6] * v_i[t, 0, 6])
                model.addConstr(lv_ij_i[t, 0, 6] - PP_ij[t, 0, 6] - QQ_ij[t, 0, 6] == 0)
                    
                model.addConstr(lv_ij_i[t, 0, 7] == l_ij[t, 0, 7] * v_i[t, 0, 7])
                model.addConstr(lv_ij_i[t, 0, 7] - PP_ij[t, 0, 7] - QQ_ij[t, 0, 7] == 0)
                    
                model.addConstr(lv_ij_i[t, 0, 8] == l_ij[t, 0, 8] * v_i[t, 0, 8])
                model.addConstr(lv_ij_i[t, 0, 8] - PP_ij[t, 0, 8] - QQ_ij[t, 0, 8] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 9] == l_ij[t, 0, 9] * v_i[t, 0, 9])
                model.addConstr(lv_ij_i[t, 0, 9] - PP_ij[t, 0, 9] - QQ_ij[t, 0, 9] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 10] == l_ij[t, 0, 10] * v_i[t, 0, 10])
                model.addConstr(lv_ij_i[t, 0, 10] - PP_ij[t, 0, 10] - QQ_ij[t, 0, 10] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 11] == l_ij[t, 0, 11] * v_i[t, 0, 11])
                model.addConstr(lv_ij_i[t, 0, 11] - PP_ij[t, 0, 11] - QQ_ij[t, 0, 11] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 12] == l_ij[t, 0, 12] * v_i[t, 0, 12])
                model.addConstr(lv_ij_i[t, 0, 12] - PP_ij[t, 0, 12] - QQ_ij[t, 0, 12] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 13] == l_ij[t, 0, 13] * v_i[t, 0, 13])
                model.addConstr(lv_ij_i[t, 0, 13] - PP_ij[t, 0, 13] - QQ_ij[t, 0, 13] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 14] == l_ij[t, 0, 14] * v_i[t, 0, 14])
                model.addConstr(lv_ij_i[t, 0, 14] - PP_ij[t, 0, 14] - QQ_ij[t, 0, 14] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 15] == l_ij[t, 0, 15] * v_i[t, 0, 15])
                model.addConstr(lv_ij_i[t, 0, 15] - PP_ij[t, 0, 15] - QQ_ij[t, 0, 15] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 16] == l_ij[t, 0, 16] * v_i[t, 0, 16])
                model.addConstr(lv_ij_i[t, 0, 16] - PP_ij[t, 0, 16] - QQ_ij[t, 0, 16] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 17] == l_ij[t, 0, 17] * v_i[t, 0, 1])
                model.addConstr(lv_ij_i[t, 0, 17] - PP_ij[t, 0, 17] - QQ_ij[t, 0, 17] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 18] == l_ij[t, 0, 18] * v_i[t, 0, 18])
                model.addConstr(lv_ij_i[t, 0, 18] - PP_ij[t, 0, 18] - QQ_ij[t, 0, 18] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 19] == l_ij[t, 0, 19] * v_i[t, 0, 19])
                model.addConstr(lv_ij_i[t, 0, 19] - PP_ij[t, 0, 19] - QQ_ij[t, 0, 19] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 20] == l_ij[t, 0, 20] * v_i[t, 0, 20])
                model.addConstr(lv_ij_i[t, 0, 20] - PP_ij[t, 0, 20] - QQ_ij[t, 0, 20] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 21] == l_ij[t, 0, 21] * v_i[t, 0, 2])
                model.addConstr(lv_ij_i[t, 0, 21] - PP_ij[t, 0, 21] - QQ_ij[t, 0, 21] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 22] == l_ij[t, 0, 22] * v_i[t, 0, 22])
                model.addConstr(lv_ij_i[t, 0, 22] - PP_ij[t, 0, 22] - QQ_ij[t, 0, 22] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 23] == l_ij[t, 0, 23] * v_i[t, 0, 23])
                model.addConstr(lv_ij_i[t, 0, 23] - PP_ij[t, 0, 23] - QQ_ij[t, 0, 23] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 24] == l_ij[t, 0, 24] * v_i[t, 0, 5])
                model.addConstr(lv_ij_i[t, 0, 24] - PP_ij[t, 0, 24] - QQ_ij[t, 0, 24] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 25] == l_ij[t, 0, 25] * v_i[t, 0, 25])
                model.addConstr(lv_ij_i[t, 0, 25] - PP_ij[t, 0, 25] - QQ_ij[t, 0, 25] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 26] == l_ij[t, 0, 26] * v_i[t, 0, 26])
                model.addConstr(lv_ij_i[t, 0, 26] - PP_ij[t, 0, 26] - QQ_ij[t, 0, 26] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 27] == l_ij[t, 0, 27] * v_i[t, 0, 27])
                model.addConstr(lv_ij_i[t, 0, 27] - PP_ij[t, 0, 27] - QQ_ij[t, 0, 27] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 28] == l_ij[t, 0, 28] * v_i[t, 0, 28])
                model.addConstr(lv_ij_i[t, 0, 28] - PP_ij[t, 0, 28] - QQ_ij[t, 0, 28] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 29] == l_ij[t, 0, 29] * v_i[t, 0, 29])
                model.addConstr(lv_ij_i[t, 0, 29] - PP_ij[t, 0, 29] - QQ_ij[t, 0, 29] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 30] == l_ij[t, 0, 30] * v_i[t, 0, 30])
                model.addConstr(lv_ij_i[t, 0, 30] - PP_ij[t, 0, 30] - QQ_ij[t, 0, 30] == 0)
                     
                model.addConstr(lv_ij_i[t, 0, 31] == l_ij[t, 0, 31] * v_i[t, 0, 31])
                model.addConstr(lv_ij_i[t, 0, 31] - PP_ij[t, 0, 31] - QQ_ij[t, 0, 31] == 0)
               
                
               

                
                                                     
            model.setParam('outPutFlag', 0)           
            model.setParam(GRB.Param.TimeLimit, 300)
            model.Params.MIPGap = 0.001
            model.optimize()

                
            print('obj=', model.objVal)
            obj=model.objval
            dict_value=dict()
            for v in model.getVars():
                                            
                dict_value[v.varName]=v.x
            dict_plot['P_DG_5'].append(dict_value['P_DG_5[0]'])
            dict_plot['SOC_BSS_9'].append(dict_value['SOC_BSS_9[0]'])
            dict_plot['Loss'].append(dict_value['Loss[0]'])
            dict_plot['P_BSS'].append(- dict_value['P_BSS_ch_9[0]'] * 1.02 + dict_value['P_BSS_dch_9[0]'] * 0.98)
            dict_plot['P_G'].append(dict_value['P_ij[0,0,0]'])

        except GurobiError as e:
            print('Error code ' + str(e.errno) + ':' + str(e))

        except AttributeError:
            print('Encountered an attribute error')

        return obj, dict_value['P_DG_5[0]'], dict_value['SOC_BSS_9[0]'], dict_value['cost_hour[0]'], model.Runtime


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

        
          
    a0_1 = np.random.uniform(0.6, 0.8, (12, 32))
    a0_2 = np.random.uniform(0.8, 1, (12, 32))
    a0_3 = np.random.uniform(1, 1.2, (12, 32))
    a0_4 = np.random.uniform(0.8, 1, (12, 32))
    a1 = np.vstack([a0_1, a0_2, a0_3, a0_4])
    ori_load = [0.1, 0.09, 0.12, 0.06, 0.06, 0.2, 0.2, 0.06, 0.06, 0.045, 0.06, 0.06, 0.12, 0.06,
                0.06, 0.06, 0.09, 0.09, 0.09, 0.09, 0.09, 0.09, 0.42, 0.42, 0.06, 0.06, 0.06, 0.12,
                0.2, 0.15, 0.21, 0.06, ]
                 
    load_list = []
    for i in range(24 * 2):
        load_list.append(a1[i] * ori_load)
    load_list = np.array(load_list)
    load_min_ratio = np.array(0.6)
    load_max_ratio = np.array(1.2)
    load_min = load_min_ratio * ori_load
    load_max = load_max_ratio * ori_load

        
          
    a0_1_Q = np.random.uniform(0.6, 0.8, (12, 32))
    a0_2_Q = np.random.uniform(0.8, 1, (12, 32))
    a0_3_Q = np.random.uniform(1, 1.2, (12, 32))
    a0_4_Q = np.random.uniform(0.8, 1, (12, 32))
    a1_Q = np.vstack([a0_1_Q, a0_2_Q, a0_3_Q, a0_4_Q])
    ori_load_Q = [0.06, 0.04, 0.08, 0.03, 0.02, 0.1, 0.1, 0.02, 0.02, 0.03, 0.035, 0.035, 0.08, 0.01, 0.02, 0.02, 0.04,
                  0.04, 0.04, 0.04, 0.04, 0.05, 0.2, 0.2, 0.025, 0.025, 0.02, 0.07, 0.6, 0.07, 0.1, 0.04]
                 
    load_list_Q = []
    for i in range(24 * 2):
        load_list_Q.append(a1_Q[i] * ori_load_Q)
    load_list_Q = np.array(load_list_Q)
    load_min_ratio_Q = np.array(0.6)
    load_max_ratio_Q = np.array(1.2)
    load_min_Q = load_min_ratio_Q * ori_load_Q
    load_max_Q = load_max_ratio_Q * ori_load_Q

           
    wind_26_avail_list = np.random.uniform(1.2, 1.8, 24 * 2)

           
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
    ori_pv_11 = [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0,
        0.08555, 0.2146, 0.28005, 0.34255, 0.4083,
        0.4965, 0.57495, 0.65735, 0.7352, 0.77635,
        0.7934, 0.76625, 0.7384, 0.66305, 0.5944,
        0.5039, 0.4242, 0.34665, 0.27375, 0.21285,
        0.15455, 0.112, 0.0765, 0.04975, 0.03195,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0]
    pv_11_avail_list = []
    for i in range(24 * 2):
        pv_11_avail_list.append(ori_pv_11[i] * np.random.uniform(0.8, 1.2) * 3)
    pv_11_avail_list = np.array(pv_11_avail_list)


    return load_list, pv_11_avail_list, wind_26_avail_list, load_min, load_max, load_list_Q, load_min_Q, load_max_Q


def _get_network():

                                         
    V_base = 12.66 * 1000
    S_base = 1 * 1000000

                
    list_r_x = [[0.0922, 0.0470], [0.4930, 0.2511], [0.3660, 0.1864], [0.3811, 0.1941], [0.8190, 0.7070],
                [0.1872, 0.6188], [1.7114, 1.2351], [1.0300, 0.7400],
                [1.0440, 0.7400], [0.1966, 0.0650], [0.3744, 0.1238], [1.4680, 1.1550], [0.5416, 0.7129],
                [0.5910, 0.5260], [0.7463, 0.5450], [1.2890, 1.7210],
                [0.7320, 0.5740], [0.1640, 0.1565], [1.5042, 1.3554], [0.4095, 0.4784], [0.7089, 0.9373],
                [0.4512, 0.3083], [0.8980, 0.7091], [0.8960, 0.7011],
                [0.2030, 0.1034], [0.2842, 0.1447], [1.0590, 0.9337], [0.8042, 0.7006], [0.5075, 0.2585],
                [0.9744, 0.9630], [0.3105, 0.3619], [0.3410, 0.5302],
                [2.0000, 2.0000], [2.0000, 2.0000], [2.0000, 2.0000], [0.5000, 0.5000], [0.5000, 0.5000]]
    list_r_x_pu = []
    for k in range(len(list_r_x)):
        list_r_x_pu.append([list_r_x[k][0] / (V_base * V_base / S_base), list_r_x[k][1] / (V_base * V_base / S_base)])

                       
    list_p_q_pu = [[0.1,0.06],[0.09,0.04],[0.12,0.08],[0.06,0.03],[0.06,0.02],[0.2,0.1],[0.2,0.1],[0.06,0.02],[0.06,0.02],[0.045,0.03],[0.06,0.035],[0.06,0.035],
                  [0.12,0.08],[0.06,0.01],[0.06,0.02],[0.06,0.02],[0.09,0.04],[0.09,0.04],[0.09,0.04],[0.09,0.04],[0.09,0.04],
                  [0.09,0.05],[0.42,0.2],[0.42,0.2],[0.06,0.025],[0.06,0.025],[0.06,0.02],[0.12,0.07],[0.2,0.6],[0.15,0.07],
                  [0.21,0.1],[0.06,0.04]]

    return list_r_x_pu, list_p_q_pu


def _renew_scenario(t, old_load_list, old_pv_11_avail_list, old_wind_26_avail_list, old_load_list_Q):

    fuzhu_load_list, fuzhu_pv_11_avail_list, fuzhu_wind_26_avail_list, _, _, fuzhu_load_list_Q, _, _ = _get_scenario()

    new_load_list = old_load_list.copy()
    new_load_list_Q = old_load_list_Q.copy()
    new_pv_11_avail_list = old_pv_11_avail_list.copy()
    new_wind_26_avail_list = old_wind_26_avail_list.copy()

    for i in range(t + 4, 24 * 2):
        new_load_list[i] = (fuzhu_load_list[i]).copy()
        new_load_list_Q[i] = (fuzhu_load_list_Q[i]).copy()
        new_pv_11_avail_list[i] = (fuzhu_pv_11_avail_list[i]).copy()
        new_wind_26_avail_list[i] = (fuzhu_wind_26_avail_list[i]).copy()
    new_load_list[t + 1] = old_load_list[t + 1] * 0.7 + fuzhu_load_list[t + 1] * 0.3
    new_load_list[t + 2] = old_load_list[t + 2] * 0.4 + fuzhu_load_list[t + 2] * 0.6
    new_load_list[t + 3] = old_load_list[t + 3] * 0.1 + fuzhu_load_list[t + 3] * 0.9
    new_load_list_Q[t + 1] = old_load_list_Q[t + 1] * 0.7 + fuzhu_load_list_Q[t + 1] * 0.3
    new_load_list_Q[t + 2] = old_load_list_Q[t + 2] * 0.4 + fuzhu_load_list_Q[t + 2] * 0.6
    new_load_list_Q[t + 3] = old_load_list_Q[t + 3] * 0.1 + fuzhu_load_list_Q[t + 3] * 0.9
    new_pv_11_avail_list[t + 1] = old_pv_11_avail_list[t + 1] * 0.7 + fuzhu_pv_11_avail_list[t + 1] * 0.3
    new_pv_11_avail_list[t + 2] = old_pv_11_avail_list[t + 2] * 0.4 + fuzhu_pv_11_avail_list[t + 2] * 0.6
    new_pv_11_avail_list[t + 3] = old_pv_11_avail_list[t + 3] * 0.1 + fuzhu_pv_11_avail_list[t + 3] * 0.9
    new_wind_26_avail_list[t + 1] = old_wind_26_avail_list[t + 1] * 0.7 + fuzhu_wind_26_avail_list[t + 1] * 0.3
    new_wind_26_avail_list[t + 2] = old_wind_26_avail_list[t + 2] * 0.4 + fuzhu_wind_26_avail_list[t + 2] * 0.6
    new_wind_26_avail_list[t + 3] = old_wind_26_avail_list[t + 3] * 0.1 + fuzhu_wind_26_avail_list[t + 3] * 0.9

    return new_load_list, new_pv_11_avail_list, new_wind_26_avail_list, new_load_list_Q


def _renew_scenario_remainder(t, old_load_list, old_pv_11_avail_list, old_wind_26_avail_list, old_load_list_Q):

    fuzhu_load_list, fuzhu_pv_11_avail_list, fuzhu_wind_26_avail_list, _, _, fuzhu_load_list_Q, _, _ = _get_scenario()

    new_load_list = old_load_list.copy()
    new_load_list_Q = old_load_list_Q.copy()
    new_pv_11_avail_list = old_pv_11_avail_list.copy()
    new_wind_26_avail_list = old_wind_26_avail_list.copy()

    if t < (24  + 20):
        for i in range(t + 4, 24 * 2):
            new_load_list[i] = (fuzhu_load_list[i]).copy()
            new_load_list_Q[i] = (fuzhu_load_list_Q[i]).copy()
            new_pv_11_avail_list[i] = (fuzhu_pv_11_avail_list[i]).copy()
            new_wind_26_avail_list[i] = (fuzhu_wind_26_avail_list[i]).copy()
    if t < (24  + 23):
        new_load_list[t + 1] = old_load_list[t + 1] * 0.7 + fuzhu_load_list[t + 1] * 0.3
        new_load_list_Q[t + 1] = old_load_list_Q[t + 1] * 0.7 + fuzhu_load_list_Q[t + 1] * 0.3
        new_pv_11_avail_list[t + 1] = old_pv_11_avail_list[t + 1] * 0.7 + fuzhu_pv_11_avail_list[t + 1] * 0.3
        new_wind_26_avail_list[t + 1] = old_wind_26_avail_list[t + 1] * 0.7 + fuzhu_wind_26_avail_list[t + 1] * 0.3
    if t < (24  + 22):
        new_load_list[t + 2] = old_load_list[t + 2] * 0.4 + fuzhu_load_list[t + 2] * 0.6
        new_load_list_Q[t + 2] = old_load_list_Q[t + 2] * 0.4 + fuzhu_load_list_Q[t + 2] * 0.6
        new_pv_11_avail_list[t + 2] = old_pv_11_avail_list[t + 2] * 0.4 + fuzhu_pv_11_avail_list[t + 2] * 0.6
        new_wind_26_avail_list[t + 2] = old_wind_26_avail_list[t + 2] * 0.4 + fuzhu_wind_26_avail_list[t + 2] * 0.6
    if t < (24  + 21):
        new_load_list[t + 3] = old_load_list[t + 3] * 0.1 + fuzhu_load_list[t + 3] * 0.9
        new_load_list_Q[t + 3] = old_load_list_Q[t + 3] * 0.1 + fuzhu_load_list_Q[t + 3] * 0.9
        new_pv_11_avail_list[t + 3] = old_pv_11_avail_list[t + 3] * 0.1 + fuzhu_pv_11_avail_list[t + 3] * 0.9
        new_wind_26_avail_list[t + 3] = old_wind_26_avail_list[t + 3] * 0.1 + fuzhu_wind_26_avail_list[t + 3] * 0.9

    return new_load_list, new_pv_11_avail_list, new_wind_26_avail_list, new_load_list_Q


env = PowerSystemEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

    
objlist = []
timelist = []

for seed in range(0,1):
    print("seed",seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

          
    window_time = 4
                     
                     

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

                
    load_list, pv_11_avail_list, wind_26_avail_list, load_min, load_max, load_list_Q, load_min_Q, load_max_Q = _get_scenario()

                
    list_r_x_pu, list_p_q_pu = _get_network()

           
    MPC = Solve_MPC(Q_1_mat, Q_2_mat, load_min, load_max, list_r_x_pu, list_p_q_pu, load_min_Q, load_max_Q)

             
    current_time = 0
    P_DG_5_init = 0
    SOC_BSS_9_init = 1.25
    cost_all = 0

                              

    solve_time_all = 0
    dict_plot={}
    dict_plot['P_DG_5']=[]
    dict_plot['SOC_BSS_9']=[]
    dict_plot['Loss']=[]
    dict_plot['P_BSS']=[]
    dict_plot['P_G']=[]

    for t in range(24 * 2 - window_time + 1):
        obj, new_P_DG_5_init, new_SOC_BSS_9_init, cost_hour_now, solve_time = MPC.sol_pro(current_time, P_DG_5_init, SOC_BSS_9_init,
                                                                        load_list, pv_11_avail_list, wind_26_avail_list,
                                                                              window_time, load_list_Q)

        solve_time_all += solve_time

                   
        load_list, pv_11_avail_list, wind_26_avail_list, load_list_Q = _renew_scenario(t, load_list, pv_11_avail_list,
                                                                                         wind_26_avail_list, load_list_Q)
        cost_all += cost_hour_now

                 
                        
                               
                                        
                                            
                                        
                                 
                 
                                     
                                 
        current_time += 1
        P_DG_5_init = new_P_DG_5_init
        SOC_BSS_9_init = new_SOC_BSS_9_init
       
                  
                                        
                          
                                             
       

    for t in range(24 * 2 - window_time + 1, 24 * 2):
                                   
        window_time -= 1

        obj, new_P_DG_5_init, new_SOC_BSS_9_init, cost_hour_now, solve_time = MPC.sol_pro_remainder(current_time, P_DG_5_init,
                                                                                        SOC_BSS_9_init, load_list,
                                                                                        pv_11_avail_list, wind_26_avail_list,
                                                                                        window_time, load_list_Q)

        solve_time_all += solve_time

                   
        if t < (24 + 23):
            load_list, pv_11_avail_list, wind_26_avail_list, load_list_Q = _renew_scenario_remainder(t, load_list, pv_11_avail_list,
                                                                                wind_26_avail_list, load_list_Q)
        cost_all += cost_hour_now

                 
                        
                               
                                        
                                            
                                        
                                 
                           
                     
                                         
                                     

        if t == (24  + 22):
                     
            print("Actual trajectories with perfect information")
            print("PV at bus 11", list(pv_11_avail_list))
            print("WT at bus 26", list(wind_26_avail_list))
            list_1 = list(load_list)
            for i in range(len(list_1)):
                list_1[i] = list(list_1[i])
                                
            list_2 = list(load_list_Q)
            for i in range(len(list_2)):
                list_2[i] = list(list_2[i])
                                 
            new_load_list = []
            for m in range(len(load_list)):
                new_load_list.append(sum(load_list[m]))
            print("Load", new_load_list)

        if t == (24 + 23):
            print("seed", seed)
            objlist.append(cost_all)
            timelist.append(solve_time_all)
            print("cost_all", cost_all)
            print("solve_time_all", solve_time_all)

        current_time += 1
        P_DG_5_init = new_P_DG_5_init
        SOC_BSS_9_init = new_SOC_BSS_9_init
       
                         
       

                                    

                            
                                                            
df = pd.DataFrame(objlist, columns=["objlist"])
df.to_excel("objlist.xlsx", index=False)
df = pd.DataFrame(timelist, columns=["timelist"])
df.to_excel("timelist.xlsx", index=False)

print('P_DG_5',dict_plot['P_DG_5'])
print('SOC_BSS_9',dict_plot['SOC_BSS_9'])
print('Loss',dict_plot['Loss'])
print('P_BSS',dict_plot['P_BSS'])
print('P_G',dict_plot['P_G'])




