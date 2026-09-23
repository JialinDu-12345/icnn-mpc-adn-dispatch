from gurobipy import *
import gurobipy as grb
import numpy as np


def _model_attr(model, attr_name, default=np.nan):
    try:
        return getattr(model, attr_name)
    except Exception:
        return default


def _collect_model_stats(model):
    external_collector = globals().get("collect_gurobi_model_stats")
    if callable(external_collector):
        return external_collector(model)

    sol_count = _model_attr(model, "SolCount", 0)
    try:
        sol_count = int(sol_count)
    except (TypeError, ValueError):
        sol_count = 0

    try:
        num_bin_vars = sum(1 for var in model.getVars() if getattr(var, "VType", "") == GRB.BINARY)
    except Exception:
        num_bin_vars = np.nan

    mip_gap = np.nan
    try:
        if bool(_model_attr(model, "IsMIP", False)) and sol_count > 0:
            mip_gap = float(_model_attr(model, "MIPGap", np.nan))
    except Exception:
        mip_gap = np.nan

    return {
        "num_vars": _model_attr(model, "NumVars", np.nan),
        "num_bin_vars": num_bin_vars,
        "num_constrs": _model_attr(model, "NumConstrs", np.nan),
        "num_qconstrs": _model_attr(model, "NumQConstrs", np.nan),
        "num_genconstrs": _model_attr(model, "NumGenConstrs", np.nan),
        "runtime": _model_attr(model, "Runtime", np.nan),
        "mip_gap": mip_gap,
        "status": _model_attr(model, "Status", np.nan),
        "sol_count": sol_count,
    }


class Solve_MPC:
    def __init__(self, load_min, load_max, list_r_x_pu, list_p_q_pu, load_min_Q, load_max_Q):
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

        self.time_limit_s = None
        self.mip_gap = 1e-3
        self.output_flag = 0
        self.last_model_stats = {}

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

    def sol_pro_all_day(self, P_DG_5_init, SOC_BSS_9_init, load_list, pv_11_avail_list, wind_26_avail_list,
                window_time, load_list_Q):
        current_time = 0

        self.load_list = load_list
        self.load_list_Q = load_list_Q
        self.pv_11_avail_list = pv_11_avail_list
        self.wind_26_avail_list = wind_26_avail_list
        self.list_p_q_pu_t = self._get_pq_t()

        obj = "error"
        try:
            model = Model('PerfectGlobal')
            model.setParam('OutputFlag', int(self.output_flag))
            model.setParam('NonConvex', 2)
            model.setParam('MIPFocus', 0)
            if self.time_limit_s is not None:
                model.setParam(GRB.Param.TimeLimit, float(self.time_limit_s))
            if self.mip_gap is not None:
                model.Params.MIPGap = float(self.mip_gap)

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

            P_BSS_ch_9 = model.addMVar(shape=(window_time,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS, name='P_BSS_ch_9')
            P_BSS_dch_9 = model.addMVar(shape=(window_time,), lb=0, ub=self.SOC_scale, vtype=GRB.CONTINUOUS, name='P_BSS_dch_9')

            cost_balance = model.addMVar(shape=(window_time,), vtype=GRB.CONTINUOUS, name='cost_balance')
            P_balance = model.addMVar(shape=(window_time,), vtype=GRB.CONTINUOUS, name='P_balance')
            cost_window = model.addVar(vtype=GRB.CONTINUOUS, name='cost_window')
            cost_day = model.addVar(vtype=GRB.CONTINUOUS, name='cost_day')
            cost_hour = model.addMVar(shape=(window_time,), vtype=GRB.CONTINUOUS, name='cost_hour')
            cost_genera = model.addMVar(shape=(window_time,), vtype=GRB.CONTINUOUS, name='cost_genera')

            model.setObjective(cost_day, GRB.MINIMIZE)

            model.addConstr(cost_day == cost_window)
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
            for t in range(window_time - 1):
                model.addConstr(P_DG_5[t + 1] <= P_DG_5[t] + self.gen_ramp / 2)
                model.addConstr(P_DG_5[t + 1] >= P_DG_5[t] - self.gen_ramp / 2)

            model.addConstr(SOC_BSS_9[0] == SOC_BSS_9_init + (P_BSS_ch_9[0] - P_BSS_dch_9[0]) / 2)
            for t in range(window_time - 1):
                model.addConstr(SOC_BSS_9[t + 1] == SOC_BSS_9[t] + (P_BSS_ch_9[t + 1] - P_BSS_dch_9[t + 1]) / 2)

            for t in range(window_time):
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

            model.optimize()
            self.last_model_stats = _collect_model_stats(model)

            sol_count = int(getattr(model, "SolCount", 0) or 0)
            if sol_count == 0:
                raise RuntimeError(
                    f"Gurobi produced no solution: status={model.Status}, sol_count={sol_count}"
                )

            dict_value = {var.VarName: var.X for var in model.getVars()}
            cost_hour_values = [float(dict_value[f"cost_hour[{t}]"]) for t in range(window_time)]
            P_DG_5_values = [float(dict_value[f"P_DG_5[{t}]"]) for t in range(window_time)]
            SOC_BSS_9_values = [float(dict_value[f"SOC_BSS_9[{t}]"]) for t in range(window_time)]

            return {
                "obj": float(model.ObjVal),
                "cost_hour": cost_hour_values,
                "P_DG_5": P_DG_5_values,
                "SOC_BSS_9": SOC_BSS_9_values,
                "runtime": float(model.Runtime),
                "model_stats": self.last_model_stats,
            }
        except GurobiError as e:
            raise RuntimeError(f"Gurobi failed with code {e.errno}: {e}") from e


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

