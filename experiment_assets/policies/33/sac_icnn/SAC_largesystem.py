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

NUM_CON = 5
NUM_GEN = 1
NUM_BSS = 1
NUM_RENEW = 2
NUM_BUS = 33

np.random.seed(2)


class PowerSystemEnv(gym.Env):
    def __init__(self):
        super(PowerSystemEnv, self).__init__()

        self.net = pp.networks.case33bw()

        self.net.sn_mva = 1

        self.net.line.loc[6, 'r_ohm_per_km'] = 1.7114
        self.net.line.loc[6, 'x_ohm_per_km'] = 1.2351

        self.net.bus.loc[0, 'max_vm_pu'] = 1.06
        self.net.bus.loc[0, 'min_vm_pu'] = 0.94

        self.net.ext_grid['vm_pu'] = 1.0

        pp.runpp(self.net)

        total_gen = sum(self.net.sgen.p_mw)
        total_load = sum(self.net.load.p_mw)
        total_loss = self.net.res_line.pl_mw.sum()
        balance_error = abs(total_gen - total_load - total_loss)
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

        self.pv_11_min = 0
        self.pv_11_max = 3

        self.wind_26_min = 0
        self.wind_26_max = 2

        self.svc_7_min = -0.3
        self.svc_7_max = 0.3
        self.svc_7_scale = 0.3

        self.action_space = spaces.Box(
            low=np.array([-1] * NUM_CON),
            high=np.array([1] * NUM_CON),
            dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(1 + self.num_load_buses + self.num_load_buses + NUM_RENEW + NUM_GEN + NUM_BSS,),
            dtype=np.float32)

    def _adjust_network(self):
        pp.create_sgen(self.net, bus=4, p_mw=0, q_mvar=0, name='DG5')
        pp.create_sgen(self.net, bus=6, p_mw=0, q_mvar=0, name='SVC7')
        pp.create_sgen(self.net, bus=8, p_mw=0, q_mvar=0, name='BSS9')
        pp.create_sgen(self.net, bus=10, p_mw=0, q_mvar=0, name='PV11')
        pp.create_sgen(self.net, bus=25, p_mw=0, q_mvar=0, name='WT26')

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

        wind_26 = np.random.uniform(1.2, 1.8, 24 * 2)

        return {
            'pv11': pv_11_avail_list,
            'wind26': wind_26
        }

    def reset(self):
        self.hour = 0
        self.SOC = 1.25
        gen_init = self.gen_min
        self.net.sgen.loc[0, 'p_mw'] = gen_init

        self.load_profile, self.load_profile_Q = self._create_load_profile()

        self.renew_profile = self._create_renew_profile()

        state = self._get_state()

        return state

    def new_reset(self, list_pv_11, list_feng_26, list_fu_he, list_fu_he_Q):
        self.hour = 0
        self.SOC = 1.25
        gen_init = self.gen_min
        self.net.sgen.loc[0, 'p_mw'] = gen_init

        self.load_profile = list_fu_he

        self.load_profile_Q = list_fu_he_Q

        self.renew_profile = {
            'pv11': list_pv_11,
            'wind26': list_feng_26
        }

        state = self._get_state()

        return state

    def _get_state(self):
        load = self.load_profile[self.hour].copy()

        load_Q = self.load_profile_Q[self.hour].copy()

        renew_pred = [self.renew_profile['pv11'][self.hour],
                      self.renew_profile['wind26'][self.hour]].copy()

        gen_p = self.net.sgen.p_mw.values[0].copy()

        BSS_p = self.SOC

        t_normalized = self.hour / (24 + 23)

        BSS_p_normalized = (BSS_p - self.SOC_min) / (self.SOC_max - self.SOC_min + 1e-6)
        gen_p_normalized = (gen_p - self.gen_min) / (self.gen_max - self.gen_min + 1e-6)
        load_normalized = ((load - self.ori_load * self.load_min) /
                           (self.ori_load * self.load_max - self.ori_load * self.load_min + 1e-6))
        load_Q_normalized = ((load_Q - self.ori_load_Q * self.load_min_Q) /
                           (self.ori_load_Q * self.load_max_Q - self.ori_load_Q * self.load_min_Q + 1e-6))
        self.pv_11_normalized = (renew_pred[0] - self.pv_11_min) / (self.pv_11_max - self.pv_11_min + 1e-6)
        self.wind_26_normalized = (renew_pred[1] - self.wind_26_min) / (self.wind_26_max - self.wind_26_min + 1e-6)

        state = np.concatenate([
            [t_normalized],
            load_normalized,
            load_Q_normalized,
            [self.pv_11_normalized, self.wind_26_normalized],
            [gen_p_normalized, BSS_p_normalized]
        ])

        return state.astype(np.float32)

    def step(self, action):
        get_action = action[:NUM_CON]

        gen_delta = get_action[0] * self.gen_ramp / 2
        new_gen_p = self.net.sgen.p_mw[0] + gen_delta
        new_gen_p = np.clip(new_gen_p, self.gen_min, self.gen_max)

        SOC_delta = get_action[1] * self.SOC_scale / 2
        new_SOC = self.SOC + SOC_delta
        new_SOC = np.clip(new_SOC, self.SOC_min, self.SOC_max)

        if self.SOC - new_SOC >= 0:
            new_BSS_p = (self.SOC - new_SOC) * 0.98 * 2
        else:
            new_BSS_p = (self.SOC - new_SOC) * 1.02 * 2

        new_SVC = get_action[2] * self.svc_7_scale

        pv_11_pu = np.clip(self.pv_11_normalized.copy(), 0.0, 1.0)
        new_PVQ11 = get_action[3] * np.sqrt(np.maximum(0.0, 1.0 - pv_11_pu ** 2)) * self.pv_11_max

        wind_26_pu = np.clip(self.wind_26_normalized.copy(), 0.0, 1.0)
        new_WTQ26 = get_action[4] * np.sqrt(np.maximum(0.0, 1.0 - wind_26_pu ** 2)) * self.wind_26_max

        pv_11_avail = self.renew_profile['pv11'][self.hour].copy()
        wind_26_avail = self.renew_profile['wind26'][self.hour].copy()
        load = self.load_profile[self.hour].copy()
        load_Q = self.load_profile_Q[self.hour].copy()

        self.net.sgen.p_mw = np.hstack((new_gen_p, [0], new_BSS_p, [pv_11_avail, wind_26_avail]))
        self.net.sgen.q_mvar = np.hstack(([0], new_SVC, [0], new_PVQ11, new_WTQ26))

        self.SOC = new_SOC.copy()

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

        p = self.net.sgen.p_mw[0]

        cost = (0.00240 * p ** 2 + 12.3299 * p + 0) / 2

        reward = - (cost + balance_penalty) / 100

        done = self.hour >= (24 + 23)

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

        gen_delta = get_action[0] * self.gen_ramp / 2
        new_gen_p = self.net.sgen.p_mw[0] + gen_delta
        new_gen_p = np.clip(new_gen_p, self.gen_min, self.gen_max)

        SOC_delta = get_action[1] * self.SOC_scale / 2
        new_SOC = self.SOC + SOC_delta
        new_SOC = np.clip(new_SOC, self.SOC_min, self.SOC_max)

        if self.SOC - new_SOC >= 0:
            new_BSS_p = (self.SOC - new_SOC) * 0.98 * 2
        else:
            new_BSS_p = (self.SOC - new_SOC) * 1.02 * 2

        new_SVC = get_action[2] * self.svc_7_scale

        pv_11_pu = np.clip(self.pv_11_normalized.copy(), 0.0, 1.0)
        new_PVQ11 = get_action[3] * np.sqrt(np.maximum(0.0, 1.0 - pv_11_pu ** 2)) * self.pv_11_max

        wind_26_pu = np.clip(self.wind_26_normalized.copy(), 0.0, 1.0)
        new_WTQ26 = get_action[4] * np.sqrt(np.maximum(0.0, 1.0 - wind_26_pu ** 2)) * self.wind_26_max

        pv_11_avail = self.renew_profile['pv11'][self.hour].copy()
        wind_26_avail = self.renew_profile['wind26'][self.hour].copy()
        load = self.load_profile[self.hour].copy()
        load_Q = self.load_profile_Q[self.hour].copy()

        self.net.sgen.p_mw = np.hstack((new_gen_p, [0], new_BSS_p, [pv_11_avail, wind_26_avail]))
        self.net.sgen.q_mvar = np.hstack(([0], new_SVC, [0], new_PVQ11, new_WTQ26))

        self.SOC = new_SOC.copy()

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

        p = self.net.sgen.p_mw[0]

        cost = (0.00240 * p ** 2 + 12.3299 * p + 0) / 2

        reward = - (cost + balance_penalty) / 100

        done = self.hour >= (24 + 23)

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


def _test_agent(env, agent, list_pv_11, list_feng_26, list_fu_he, list_fu_he_Q, model_path='model.pth'):
    agent.actor.load_state_dict(torch.load(model_path))

    agent.actor.eval()

    total_cost = 0
    t1 = 0

    state = env.new_reset(list_pv_11, list_feng_26, list_fu_he, list_fu_he_Q)
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


def _test_agent_1(env, agent, list_pv_11, list_feng_26, list_fu_he, list_fu_he_Q, model_path='model.pth'):
    agent.actor.load_state_dict(torch.load(model_path))

    agent.actor.eval()

    total_cost = 0
    t1 = 0

    total_cost_c = 0

    state = env.new_reset(list_pv_11, list_feng_26, list_fu_he, list_fu_he_Q)
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

    return total_cost * 100, total_cost_c, end_time - start_time


def hourly_to_15min_linear(load_hourly):
    load_hourly = np.asarray(load_hourly)

    if load_hourly.ndim == 1:
        load_hourly = load_hourly.reshape(-1, 1)

    T, N = load_hourly.shape
    load_15min = np.zeros((T * 4, N))

    for t in range(T - 1):
        for k in range(4):
            alpha = k / 4.0
            load_15min[t * 4 + k] = (
                (1 - alpha) * load_hourly[t]
                + alpha * load_hourly[t + 1]
            )

    load_15min[-4:] = load_hourly[-1]

    return load_15min

env = PowerSystemEnv()
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
random.seed(9)
np.random.seed(9)
torch.manual_seed(9)

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

list_cost = []
list_cost_c = []
list_time_ = []

for seed in range(0,1):
    print("seed:", seed)
    load_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "33"))
    load_path = os.path.join(load_dir, "results_{}.npz".format(seed))

    data = np.load(load_path)

    list_pv_11 = data["pv_11"]
    list_feng_26 = data["wind_26"]
    list_fu_he = data["load"]
    list_fu_he_Q = data["load_Q"]

    cost, cost_c, time_ = _test_agent_1(env, agent, list_pv_11, list_feng_26, list_fu_he, list_fu_he_Q)

    list_cost.append(cost)
    list_cost_c.append(cost_c)
    list_time_.append(time_)

df = pd.DataFrame(list_cost, columns=["list_cost"])
df.to_excel("list_cost.xlsx", index=False)
df = pd.DataFrame(list_cost_c, columns=["list_cost_c"])
df.to_excel("list_cost_c.xlsx", index=False)
df = pd.DataFrame(list_time_, columns=["list_time_"])
df.to_excel("list_time_.xlsx", index=False)

kk = -1

if kk == 0:
    list_feng_11 = np.array([1.3437797415145727, 1.292945062348112, 1.524743516610152, 1.4773883331355693, 1.5734208720907374, 1.4118242959909402, 1.3792212116994422, 1.4844764900247378, 1.3549611193886077, 1.643171452355527, 1.5726344174136757, 1.5351310870357904, 1.4508267977812275, 1.49239587674583, 1.4615488198356419, 1.5737790624824708, 1.4935166829838216, 1.628818003984308, 1.7086194333713054, 1.5502770576118388, 1.5700559269228749, 1.6822864013102907, 1.726227046463857, 1.5458475839002621])
    list_feng_26 = np.array([0.9765280883444509, 1.0083793281981168, 0.9742238828076892, 1.0347570817638676, 1.085712817770854, 0.9688295429275269, 0.9818933271814859, 0.9486318501954952, 0.9888984846644906, 0.94680464005921, 1.1377966342967814, 1.1307352381742457, 1.0138413964454793, 0.9733447387293048, 1.0215732622850386, 1.0103182877625525, 0.9342904142841135, 1.1217460293007255, 1.0208955567741036, 0.9826160906001553, 1.064130357855312, 1.0878096283120298, 1.0082272247557198, 1.003493547039016])
    list_fu_he = np.array([
        [0.06480999, 0.0679198 , 0.07918089, 0.        , 0.04472685,
       0.15547229, 0.12707018, 0.        , 0.03745086, 0.        ,
       0.03863254, 0.04110512, 0.08813318, 0.03727963, 0.03619659,
       0.04784368, 0.06210045, 0.05920007, 0.05773652, 0.06066802,
       0.06535178, 0.06705692, 0.2766806 , 0.29140865, 0.        ,
       0.04769252, 0.0387411 , 0.08463208, 0.14911306, 0.10824229,
       0.1613367 , 0.04790134], [0.06629389, 0.0578081 , 0.07955152, 0.        , 0.04205316,
       0.1484208 , 0.15436681, 0.        , 0.04675085, 0.        ,
       0.04491807, 0.04144108, 0.09490167, 0.03899371, 0.03819326,
       0.04276059, 0.06147612, 0.05650477, 0.06540393, 0.05871064,
       0.06955603, 0.05497291, 0.31333269, 0.29801405, 0.        ,
       0.04342382, 0.04580021, 0.08966316, 0.14033243, 0.11230786,
       0.1524662 , 0.03852723], [0.0697529 , 0.06308799, 0.08118079, 0.        , 0.04142999,
       0.14229636, 0.14739753, 0.        , 0.04593267, 0.        ,
       0.04225416, 0.04152572, 0.08441813, 0.04026238, 0.04109313,
       0.0455484 , 0.0685298 , 0.06352304, 0.05883789, 0.05807805,
       0.06033094, 0.06859868, 0.30585833, 0.27938352, 0.        ,
       0.04397492, 0.04329227, 0.08002531, 0.1337386 , 0.10514354,
       0.14741226, 0.03907337], [0.07564717, 0.06890924, 0.07916989, 0.        , 0.04261102,
       0.14293192, 0.12325315, 0.        , 0.04088455, 0.        ,
       0.04005966, 0.03785407, 0.08252296, 0.03927043, 0.04450506,
       0.04318204, 0.06230453, 0.05787267, 0.06446369, 0.06210509,
       0.06637854, 0.06422522, 0.31865325, 0.29561857, 0.        ,
       0.04028211, 0.04102745, 0.09068255, 0.13370851, 0.09742178,
       0.13755184, 0.0452391 ], [0.07149034, 0.05832295, 0.09145852, 0.        , 0.04196466,
       0.14311949, 0.13586186, 0.        , 0.04458517, 0.        ,
       0.03896889, 0.04364547, 0.08646246, 0.04174398, 0.04252201,
       0.04214402, 0.05790852, 0.06254306, 0.06373986, 0.06217039,
       0.06427869, 0.06158962, 0.30450417, 0.29761784, 0.        ,
       0.04118433, 0.04018211, 0.08844456, 0.13511502, 0.10742513,
       0.1483857 , 0.04399563], [0.07297526, 0.06169455, 0.08633461, 0.        , 0.04440359,
       0.14070574, 0.13163564, 0.        , 0.04317482, 0.        ,
       0.04094979, 0.04428968, 0.08500252, 0.03848693, 0.04073812,
       0.04457145, 0.06583581, 0.06216964, 0.06648257, 0.06405705,
       0.06401   , 0.06090123, 0.28338869, 0.2812066 , 0.        ,
       0.03853639, 0.04011012, 0.07991361, 0.14508792, 0.10193006,
       0.1497817 , 0.03975959], [0.08852223, 0.0815601 , 0.10864403, 0.        , 0.05193833,
       0.17998049, 0.18419457, 0.        , 0.05142893, 0.        ,
       0.05369122, 0.05492921, 0.11042121, 0.05556016, 0.05377907,
       0.05266803, 0.08091833, 0.07864777, 0.08442819, 0.08155412,
       0.08329038, 0.07952101, 0.37906774, 0.35519158, 0.        ,
       0.05176722, 0.05282771, 0.10521366, 0.18231988, 0.12667164,
       0.18297445, 0.05274873], [0.09321771, 0.08357704, 0.10943285, 0.        , 0.05428207,
       0.1959406 , 0.18053823, 0.        , 0.05536355, 0.        ,
       0.0530096 , 0.05700372, 0.10637595, 0.05734128, 0.05271976,
       0.05434693, 0.08165535, 0.08034723, 0.08410469, 0.08399466,
       0.07751898, 0.08425216, 0.38111744, 0.38875375, 0.        ,
       0.05293101, 0.05064654, 0.10953331, 0.16996028, 0.13269447,
       0.18522485, 0.05687939], [0.09104365, 0.07592093, 0.11177149, 0.        , 0.05389789,
       0.18318186, 0.17380733, 0.        , 0.05532168, 0.        ,
       0.05412785, 0.05357485, 0.109831  , 0.05449896, 0.05234467,
       0.05383953, 0.081575  , 0.07740418, 0.0846797 , 0.07935894,
       0.08174223, 0.0831779 , 0.3950913 , 0.37581622, 0.        ,
       0.05305371, 0.0556773 , 0.1072237 , 0.18468424, 0.13260075,
       0.18480337, 0.05260266], [0.09102984, 0.08533338, 0.10560619, 0.        , 0.05118756,
       0.1782088 , 0.17403049, 0.        , 0.055096  , 0.        ,
       0.05514999, 0.0526585 , 0.10721723, 0.05483527, 0.05445584,
       0.05443043, 0.07671395, 0.08488312, 0.08157146, 0.08180707,
       0.08210111, 0.08262984, 0.37605093, 0.392203  , 0.        ,
       0.05227077, 0.05185092, 0.10047626, 0.18816067, 0.13283704,
       0.19874786, 0.05488505], [0.09296294, 0.07728042, 0.1034734 , 0.        , 0.05441347,
       0.18471919, 0.18027959, 0.        , 0.05411478, 0.        ,
       0.05169018, 0.05156296, 0.10782315, 0.05221504, 0.05334534,
       0.05373219, 0.08506208, 0.0801283 , 0.07904425, 0.07910713,
       0.0833392 , 0.07761486, 0.37780051, 0.39453641, 0.        ,
       0.05344871, 0.05696272, 0.1002583 , 0.18740516, 0.14151223,
       0.18661614, 0.05403439], [0.08952343, 0.07735302, 0.10055547, 0.        , 0.05266192,
       0.17738588, 0.17378295, 0.        , 0.05711639, 0.        ,
       0.04994854, 0.05447293, 0.10059525, 0.05439026, 0.05220717,
       0.05481912, 0.08084513, 0.07476521, 0.08098532, 0.08041075,
       0.07729624, 0.08300613, 0.35497744, 0.37980321, 0.        ,
       0.05400291, 0.0547462 , 0.10899671, 0.16710332, 0.13309914,
       0.19535709, 0.05125143], [0.1129059 , 0.0986411 , 0.133646  , 0.        , 0.06659313,
       0.21507508, 0.2223721 , 0.        , 0.06896895, 0.        ,
       0.0661906 , 0.06581049, 0.13202586, 0.07034742, 0.0654009 ,
       0.06528674, 0.09841545, 0.10369947, 0.10072196, 0.10185976,
       0.10047557, 0.09666976, 0.46223193, 0.46064826, 0.        ,
       0.06239163, 0.06346217, 0.13581964, 0.22058712, 0.16724955,
       0.23488471, 0.06475897], [0.10871974, 0.1053427 , 0.13075614, 0.        , 0.06649234,
       0.22685614, 0.21307325, 0.        , 0.06766976, 0.        ,
       0.06723229, 0.06590978, 0.1300121 , 0.06601869, 0.06320702,
       0.06708506, 0.10106955, 0.09767443, 0.09512366, 0.09546311,
       0.09728785, 0.09604437, 0.45172106, 0.42847845, 0.        ,
       0.06592876, 0.06963297, 0.1264874 , 0.22251138, 0.16512617,
       0.24995758, 0.06978856], [0.10422349, 0.09858765, 0.12877262, 0.        , 0.06465017,
       0.22438229, 0.23778536, 0.        , 0.06196241, 0.        ,
       0.06511828, 0.06342299, 0.12897981, 0.06521062, 0.0655593 ,
       0.06728148, 0.09771705, 0.09808659, 0.0932845 , 0.09595356,
       0.1002861 , 0.10355286, 0.45192522, 0.47720164, 0.        ,
       0.06600058, 0.06828942, 0.12984338, 0.21100207, 0.16267509,
       0.2240493 , 0.0661631 ], [0.10728617, 0.10456296, 0.129158  , 0.        , 0.06490638,
       0.21991177, 0.22084647, 0.        , 0.06847907, 0.        ,
       0.06595606, 0.06682815, 0.1280873 , 0.06685389, 0.0660902 ,
       0.06744071, 0.09890765, 0.09766266, 0.10001988, 0.09812978,
       0.09872626, 0.10677393, 0.46577979, 0.46851878, 0.        ,
       0.0622939 , 0.06871029, 0.13474844, 0.22001626, 0.1689024 ,
       0.24012494, 0.06279723], [0.1071026 , 0.09972483, 0.12688358, 0.        , 0.06815706,
       0.21403637, 0.22548731, 0.        , 0.0649945 , 0.        ,
       0.06240668, 0.06710227, 0.13592041, 0.0626964 , 0.06379351,
       0.06756096, 0.09747282, 0.10100878, 0.10318965, 0.09473267,
       0.102422  , 0.09449979, 0.48304967, 0.45337064, 0.        ,
       0.06711157, 0.06262749, 0.13448561, 0.21780458, 0.16276724,
       0.24199196, 0.06767193], [0.10456062, 0.0946596 , 0.13345567, 0.        , 0.06637615,
       0.21823283, 0.23243737, 0.        , 0.06746092, 0.        ,
       0.06640458, 0.063922  , 0.13637724, 0.06635967, 0.06237784,
       0.06737655, 0.09967772, 0.0966065 , 0.10054883, 0.09618811,
       0.10149813, 0.09483121, 0.43877573, 0.495735  , 0.        ,
       0.06507983, 0.06598394, 0.13156529, 0.22363205, 0.1587566 ,
       0.22744222, 0.06674621], [0.08399442, 0.08571399, 0.11089293, 0.        , 0.05140238,
       0.18830936, 0.18730503, 0.        , 0.05547165, 0.        ,
       0.0537058 , 0.05525665, 0.10944768, 0.05300212, 0.05224158,
       0.05658839, 0.0802607 , 0.07832209, 0.07711869, 0.08212924,
       0.07540592, 0.07657995, 0.36780681, 0.36488541, 0.        ,
       0.05156093, 0.05217012, 0.10992891, 0.18985121, 0.1263864 ,
       0.19275974, 0.05334423], [0.08960522, 0.07932396, 0.10761521, 0.        , 0.0549422 ,
       0.17408273, 0.18018765, 0.        , 0.05532423, 0.        ,
       0.05480973, 0.05070677, 0.10749485, 0.05360826, 0.05547096,
       0.05666209, 0.07804988, 0.07889978, 0.08149755, 0.07502802,
       0.07674157, 0.08581862, 0.37282301, 0.37202302, 0.        ,
       0.05586117, 0.05298086, 0.10643736, 0.18723579, 0.13438234,
       0.19997543, 0.0573372 ], [0.09211239, 0.07636842, 0.10896569, 0.        , 0.051078  ,
       0.19775178, 0.18026289, 0.        , 0.05554147, 0.        ,
       0.0523781 , 0.05649783, 0.10970269, 0.05007866, 0.05548124,
       0.05533328, 0.0796716 , 0.0833563 , 0.0842999 , 0.08279862,
       0.0858189 , 0.08195785, 0.40593971, 0.35764173, 0.        ,
       0.05474692, 0.05572486, 0.10662256, 0.18870652, 0.13027869,
       0.19286947, 0.05594187], [0.09231524, 0.08121958, 0.10554787, 0.        , 0.05550107,
       0.17573619, 0.18411474, 0.        , 0.05557764, 0.        ,
       0.05308427, 0.0560296 , 0.1104331 , 0.05736719, 0.05490962,
       0.05613231, 0.08130572, 0.07706651, 0.07938731, 0.08487791,
       0.07887701, 0.07620271, 0.39962343, 0.35772367, 0.        ,
       0.05551591, 0.0542057 , 0.10507861, 0.18287695, 0.13976272,
       0.18240809, 0.05681916], [0.08500442, 0.08107649, 0.10975221, 0.        , 0.05657368,
       0.16869024, 0.18825564, 0.        , 0.05036665, 0.        ,
       0.05338753, 0.05349161, 0.10159127, 0.05707103, 0.05315521,
       0.05659857, 0.08202506, 0.07853636, 0.08155825, 0.08318733,
       0.08244102, 0.08181969, 0.39827637, 0.37531611, 0.        ,
       0.04976504, 0.05509782, 0.10874772, 0.18859332, 0.14171029,
       0.19130218, 0.05343852], [0.08348136, 0.08234842, 0.11521648, 0.        , 0.05399208,
       0.17973339, 0.18888102, 0.        , 0.0535305 , 0.        ,
       0.05325934, 0.05566352, 0.1057623 , 0.05416172, 0.05392494,
       0.0566184 , 0.08117782, 0.08343131, 0.08595126, 0.08001783,
       0.08489269, 0.07748897, 0.37596628, 0.40366487, 0.        ,
       0.05777595, 0.05449995, 0.1117252 , 0.17051288, 0.13720735,
       0.1903799 , 0.0545284 ]])
    list_fu_he_Q = np.array([
       [0.0591369 , 0.03819582, 0.07779735, 0.        , 0.01983545,
       0.09719877, 0.09991243, 0.        , 0.01964479, 0.        ,
       0.03474596, 0.03477369, 0.07970464, 0.00992371, 0.01946422,
       0.01980561, 0.03847758, 0.03929668, 0.0394059 , 0.03869793,
       0.03960134, 0.04966347, 0.19434887, 0.19184582, 0.        ,
       0.0247272 , 0.01925597, 0.0697307 , 0.58846907, 0.06851903,
       0.09862814, 0.03897763],[0.05753714, 0.03847914, 0.07675036, 0.        , 0.01985647,
       0.09777338, 0.09803876, 0.        , 0.01969664, 0.        ,
       0.03428656, 0.03436578, 0.07873274, 0.00979665, 0.01952985,
       0.019316  , 0.03926781, 0.03905487, 0.03893245, 0.03896336,
       0.03855481, 0.04892187, 0.1951993 , 0.19465778, 0.        ,
       0.02387325, 0.01911312, 0.06756783, 0.58014192, 0.06874142,
       0.09828615, 0.03930153], [0.05923796, 0.0391612 , 0.07698133, 0.        , 0.01950789,
       0.09780935, 0.09676667, 0.        , 0.0196225 , 0.        ,
       0.03424432, 0.03478405, 0.07803584, 0.00983612, 0.01956324,
       0.01964055, 0.03865824, 0.03906628, 0.03913836, 0.03941376,
       0.03928169, 0.0489939 , 0.19550061, 0.19449559, 0.        ,
       0.02460473, 0.01924925, 0.06762778, 0.58429081, 0.06834248,
       0.09659798, 0.03935915], [0.0574636 , 0.03861237, 0.07862655, 0.        , 0.01954042,
       0.09677628, 0.0972575 , 0.        , 0.0197405 , 0.        ,
       0.03376384, 0.03418396, 0.07807102, 0.00974804, 0.01963973,
       0.01940918, 0.03870326, 0.03919136, 0.03926144, 0.03922597,
       0.03948442, 0.04826409, 0.19818627, 0.19197667, 0.        ,
       0.02443413, 0.0195202 , 0.06887677, 0.59342683, 0.06912395,
       0.09770207, 0.03843216], [0.05784966, 0.0393264 , 0.07941631, 0.        , 0.01930019,
       0.09866251, 0.09679256, 0.        , 0.01959197, 0.        ,
       0.03376142, 0.03398296, 0.07893526, 0.0097706 , 0.01946427,
       0.01933862, 0.03944607, 0.03895203, 0.0389845 , 0.03893043,
       0.03901979, 0.04849818, 0.19663248, 0.19654611, 0.        ,
       0.02457136, 0.01947651, 0.06884304, 0.58660003, 0.06812253,
       0.0964937 , 0.03885599], [0.05899442, 0.03887225, 0.07859541, 0.        , 0.01973854,
       0.09896143, 0.09937936, 0.        , 0.0193516 , 0.        ,
       0.03415889, 0.03449538, 0.07772426, 0.0099188 , 0.01937418,
       0.01932122, 0.03870091, 0.038576  , 0.03878278, 0.03877356,
       0.038902  , 0.04811112, 0.19678364, 0.1947137 , 0.        ,
       0.0247727 , 0.0196039 , 0.06875462, 0.5908534 , 0.06792175,
       0.09772717, 0.03851584], [0.06031634, 0.04014264, 0.07973192, 0.        , 0.01971551,
       0.10004878, 0.09864314, 0.        , 0.02003635, 0.        ,
       0.03486403, 0.03524982, 0.07963015, 0.01013499, 0.02003799,
       0.01994274, 0.03966006, 0.03996097, 0.03980222, 0.03996155,
       0.03989635, 0.050283  , 0.20256382, 0.19881905, 0.        ,
       0.02527407, 0.02011439, 0.07044054, 0.60111127, 0.0703827 ,
       0.10048386, 0.03982409],[0.05945771, 0.03967513, 0.08008211, 0.        , 0.02020856,
       0.09923034, 0.09963995, 0.        , 0.02010728, 0.        ,
       0.03509258, 0.03470668, 0.07971943, 0.01010646, 0.02008164,
       0.02016611, 0.04029614, 0.03997582, 0.03996218, 0.03968804,
       0.0399471 , 0.04938758, 0.20056356, 0.20068051, 0.        ,
       0.02497409, 0.02018065, 0.0697278 , 0.60029236, 0.07004125,
       0.10032879, 0.03982784], [0.05998165, 0.04045908, 0.08053391, 0.        , 0.02005064,
       0.09945194, 0.09910255, 0.        , 0.02010844, 0.        ,
       0.03478307, 0.03470407, 0.08010573, 0.00990543, 0.01977941,
       0.01989684, 0.04024936, 0.04012342, 0.03986993, 0.03969201,
       0.04006766, 0.05036022, 0.2006405 , 0.19835799, 0.        ,
       0.02510469, 0.02023678, 0.06952617, 0.59860497, 0.07008071,
       0.0987868 , 0.04001827], [0.06014211, 0.04038439, 0.08030219, 0.        , 0.01974063,
       0.09926643, 0.099731  , 0.        , 0.02007327, 0.        ,
       0.03551237, 0.03524281, 0.0809305 , 0.01008284, 0.02007878,
       0.02025854, 0.03994121, 0.04000334, 0.03963104, 0.03983514,
       0.04027285, 0.05025681, 0.20010505, 0.20197204, 0.        ,
       0.02483936, 0.02022557, 0.07058686, 0.59993318, 0.06961383,
       0.09962085, 0.04004518], [0.06066926, 0.03986407, 0.08102903, 0.        , 0.01993682,
       0.10008911, 0.10167273, 0.        , 0.01999824, 0.        ,
       0.03523383, 0.0353654 , 0.07920826, 0.01011987, 0.02015185,
       0.0200657 , 0.0398014 , 0.04006878, 0.0401899 , 0.04010298,
       0.04023046, 0.05057039, 0.2010016 , 0.19935382, 0.        ,
       0.02479267, 0.01999799, 0.07037927, 0.60651523, 0.06997505,
       0.09956229, 0.03994713], [0.06062926, 0.04020254, 0.08031908, 0.        , 0.01996219,
       0.10005394, 0.1003764 , 0.        , 0.01994045, 0.        ,
       0.03526665, 0.03493714, 0.08054941, 0.00992632, 0.02001949,
       0.02001307, 0.04009696, 0.03985234, 0.03998236, 0.040148  ,
       0.039774  , 0.0496749 , 0.19803301, 0.20190178, 0.        ,
       0.02489407, 0.02000125, 0.06974387, 0.6021744 , 0.06973132,
       0.09980375, 0.03992287], [0.06185079, 0.04153551, 0.08154156, 0.        , 0.02046497,
       0.10211169, 0.10225669, 0.        , 0.02030756, 0.        ,
       0.0361013 , 0.03579225, 0.08270891, 0.01017735, 0.0206004 ,
       0.02063422, 0.04050874, 0.04031485, 0.04102137, 0.04102575,
       0.04097415, 0.05165458, 0.20468361, 0.20430766, 0.        ,
       0.02561712, 0.02032396, 0.07247733, 0.60599555, 0.07135352,
       0.10289278, 0.04058571], [0.06059917, 0.04081335, 0.08052494, 0.        , 0.02035013,
       0.10236426, 0.10171773, 0.        , 0.02058641, 0.        ,
       0.03608537, 0.03603759, 0.08204133, 0.01015982, 0.02058593,
       0.0204525 , 0.04089285, 0.04139524, 0.0406189 , 0.04065335,
       0.04057491, 0.05099448, 0.20526905, 0.20628222, 0.        ,
       0.02516422, 0.02084206, 0.07210733, 0.62144736, 0.07139891,
       0.10231632, 0.04112602], [0.06134268, 0.04148944, 0.08214414, 0.        , 0.02056478,
       0.10241744, 0.1026441 , 0.        , 0.02049937, 0.        ,
       0.03618277, 0.03607427, 0.08156592, 0.01028685, 0.02073978,
       0.02032826, 0.04122299, 0.04103946, 0.04116887, 0.04135158,
       0.04072135, 0.05115394, 0.20585248, 0.20385591, 0.        ,
       0.02553575, 0.02035253, 0.07176552, 0.60950938, 0.07284789,
       0.10335609, 0.04120674], [0.06094561, 0.04068547, 0.08292723, 0.        , 0.02052795,
       0.10229383, 0.10147257, 0.        , 0.02029691, 0.        ,
       0.03582382, 0.03586406, 0.08124692, 0.01010146, 0.02087151,
       0.02036958, 0.04131398, 0.04031661, 0.04117018, 0.04133758,
       0.04110096, 0.05185717, 0.20408485, 0.20434409, 0.        ,
       0.02544933, 0.02041476, 0.07199016, 0.61525814, 0.07198915,
       0.10242101, 0.04051266], [0.06207974, 0.04083004, 0.08214626, 0.        , 0.0206421 ,
       0.10224229, 0.10303629, 0.        , 0.02051813, 0.        ,
       0.03586778, 0.0357607 , 0.08262659, 0.01017506, 0.0200997 ,
       0.02038469, 0.04136599, 0.04089249, 0.04107744, 0.04064717,
       0.04103665, 0.05105678, 0.20278732, 0.20668545, 0.        ,
       0.02608542, 0.02046922, 0.0722484 , 0.61047873, 0.07187958,
       0.10309784, 0.04095256], [0.06088576, 0.04158059, 0.08211845, 0.        , 0.02028062,
       0.10289453, 0.10264898, 0.        , 0.02048105, 0.        ,
       0.0357923 , 0.03580948, 0.08136201, 0.01029089, 0.02045564,
       0.02058994, 0.04031692, 0.04132111, 0.04103686, 0.04111084,
       0.04095724, 0.0504032 , 0.20226992, 0.20703785, 0.        ,
       0.02568538, 0.02016464, 0.07247197, 0.61316735, 0.07085296,
       0.10238613, 0.04059427], [0.05992223, 0.03948254, 0.07982275, 0.        , 0.01998489,
       0.09983108, 0.10059847, 0.        , 0.01998241, 0.        ,
       0.03503024, 0.03519335, 0.08078792, 0.00998256, 0.01993397,
       0.01999564, 0.04004889, 0.03999255, 0.04001633, 0.04010007,
       0.03997629, 0.05035495, 0.20044921, 0.1988921 , 0.        ,
       0.02489577, 0.02012977, 0.06969702, 0.60487189, 0.06995706,
       0.10057405, 0.03977062], [0.05998065, 0.0398794 , 0.08035475, 0.        , 0.01993253,
       0.10004207, 0.09960142, 0.        , 0.01997288, 0.        ,
       0.03527229, 0.03496691, 0.07913809, 0.00999848, 0.01988566,
       0.02014682, 0.03986043, 0.04035596, 0.04047666, 0.04000414,
       0.03967461, 0.04984592, 0.20027626, 0.20223401, 0.        ,
       0.02490982, 0.02009531, 0.07080918, 0.5945521 , 0.07000499,
       0.09865023, 0.03991903], [0.0596219 , 0.03997338, 0.07996784, 0.        , 0.01991961,
       0.10043414, 0.10005214, 0.        , 0.02007557, 0.        ,
       0.03503025, 0.03499687, 0.08009858, 0.01007726, 0.02004998,
       0.01995676, 0.04010714, 0.03945614, 0.04044341, 0.04005733,
       0.04015013, 0.05031684, 0.20057486, 0.20034679, 0.        ,
       0.02494714, 0.01986496, 0.07031615, 0.60307225, 0.07006769,
       0.1001293 , 0.03990319], [0.05914155, 0.04014246, 0.07972415, 0.        , 0.02007751,
       0.10036375, 0.10139076, 0.        , 0.02009949, 0.        ,
       0.03490072, 0.03517354, 0.08012229, 0.00994629, 0.02004847,
       0.02000152, 0.03988862, 0.04010344, 0.03971639, 0.04044461,
       0.04024985, 0.05005296, 0.2020601 , 0.1989836 , 0.        ,
       0.02515264, 0.0197664 , 0.0699846 , 0.60863901, 0.06986053,
       0.09886629, 0.04063683], [0.05984638, 0.03986821, 0.0811457 , 0.        , 0.02018101,
       0.09999611, 0.10080059, 0.        , 0.0198264 , 0.        ,
       0.03499186, 0.03494257, 0.08048129, 0.00998929, 0.01996104,
       0.01992389, 0.04014031, 0.03998996, 0.04043805, 0.04009139,
       0.03980771, 0.04996391, 0.20235445, 0.19810845, 0.        ,
       0.02480023, 0.01981593, 0.07006064, 0.59528632, 0.06970045,
       0.09981816, 0.04027183], [0.06037081, 0.04024609, 0.07935095, 0.        , 0.02009249,
       0.09989933, 0.09933985, 0.        , 0.02021213, 0.        ,
       0.03489312, 0.03482309, 0.08007972, 0.00995595, 0.01994892,
       0.02018647, 0.03998386, 0.04053398, 0.03957567, 0.03963559,
       0.04048425, 0.04956614, 0.19771913, 0.19748894, 0.        ,
       0.02508353, 0.01994764, 0.06999713, 0.60061125, 0.07030863,
       0.0994569 , 0.03944754]])

    _test_agent_1(env, agent, list_feng_11, list_feng_26, list_fu_he, list_fu_he_Q)
elif kk == 1:
    list_feng_11 = np.array([1.208013065757516, 1.5177410972085807, 1.411028536294012, 1.6923464057582893, 1.412253774401898, 1.5121911531360688, 1.5958727701509305, 1.6869132085661735, 1.4949964182243736, 1.5370983593855045, 1.6871869147017817, 1.6538115382024468, 1.5103844111652167, 1.469717793698099, 1.6499343451490038, 1.4556644531439051, 1.5448744200499807, 1.5155085798695613, 1.5729151827992718, 1.3862189863895733, 1.5839887388126455, 1.6977855074601362, 1.4257531638960725, 1.4045912183412481])
    list_feng_26 = np.array([0.9141717928329255, 1.0258123976480513, 0.9627353819482014, 1.0946202539429553, 1.0651920717175232, 0.9634778416345705, 1.0907555126404354, 0.9323995433710079, 1.1031997388096468, 0.966755978110634, 1.0180367635388934, 0.92981274879683, 0.999525091418454, 0.9216435346969593, 1.049074451984762, 0.9723295692875407, 0.9737077283526017, 0.967692976559972, 0.9019021015668081, 0.9409787360770332, 0.9769852533214559, 0.9815848115993113, 0.9248205318912444, 0.8949645513575093])
    list_fu_he = np.array([
        [0.07987763, 0.06994091, 0.07509186, 0.        , 0.04263143,
       0.12972156, 0.14243042, 0.        , 0.03707262, 0.        ,
       0.04431404, 0.04640205, 0.08622402, 0.04361402, 0.04549255,
       0.04688759, 0.05558916, 0.06627808, 0.06813567, 0.06677895,
       0.06757405, 0.06760868, 0.30218471, 0.32876642, 0.        ,
       0.03715642, 0.04389417, 0.09380513, 0.13921425, 0.11318237,
       0.14636725, 0.0441754 ], [0.06226389, 0.06481787, 0.08706591, 0.        , 0.0460512 ,
       0.14389783, 0.1280528 , 0.        , 0.04407384, 0.        ,
       0.03958894, 0.04555893, 0.08829235, 0.04201913, 0.04366822,
       0.04583552, 0.06667229, 0.05539998, 0.06266262, 0.05810924,
       0.06000462, 0.0655754 , 0.30968208, 0.32039792, 0.        ,
       0.04185162, 0.03743617, 0.08265368, 0.12321134, 0.09630512,
       0.13904958, 0.03832168], [0.07429909, 0.06503682, 0.08406133, 0.        , 0.04328936,
       0.13879484, 0.13918283, 0.        , 0.0395392 , 0.        ,
       0.04138175, 0.04184278, 0.08446445, 0.04388042, 0.04654155,
       0.03971104, 0.06681189, 0.0594165 , 0.06385673, 0.06618357,
       0.06732847, 0.06783414, 0.32450202, 0.30558856, 0.        ,
       0.03840806, 0.0384178 , 0.08398319, 0.12987915, 0.10562104,
       0.13869362, 0.03914261], [0.07459334, 0.06201534, 0.08286738, 0.        , 0.04458068,
       0.14162822, 0.13647281, 0.        , 0.04241779, 0.        ,
       0.04501061, 0.04022577, 0.08694   , 0.03756157, 0.04618427,
       0.04435096, 0.06026082, 0.0696785 , 0.06238084, 0.06734162,
       0.06293799, 0.05870856, 0.3046168 , 0.29666305, 0.        ,
       0.03773157, 0.04113101, 0.07696187, 0.14225563, 0.10550568,
       0.14471569, 0.0421108 ], [0.06713666, 0.06403326, 0.08342252, 0.        , 0.04412279,
       0.13979369, 0.1336129 , 0.        , 0.04132036, 0.        ,
       0.04047878, 0.042679  , 0.08056865, 0.03916907, 0.04232826,
       0.04076526, 0.05892643, 0.0663945 , 0.05911227, 0.06517946,
       0.06634653, 0.06416516, 0.30175838, 0.28867905, 0.        ,
       0.04290761, 0.04057361, 0.0786599 , 0.13198772, 0.10391207,
       0.13626033, 0.03881522], [0.07012356, 0.06529772, 0.08739451, 0.        , 0.04546645,
       0.14407497, 0.14202238, 0.        , 0.03848505, 0.        ,
       0.04019723, 0.04365304, 0.08972378, 0.03796276, 0.04150701,
       0.04168656, 0.06419671, 0.06220251, 0.06344996, 0.05847857,
       0.06457093, 0.06392888, 0.28465516, 0.30183243, 0.        ,
       0.04575488, 0.04113784, 0.08872487, 0.14194008, 0.10540778,
       0.14967131, 0.04228049], [0.08965832, 0.07706893, 0.10813686, 0.        , 0.05363466,
       0.18434674, 0.174991  , 0.        , 0.05196736, 0.        ,
       0.05324617, 0.05615978, 0.10460082, 0.05057692, 0.05400549,
       0.05506405, 0.08366549, 0.08297175, 0.07413537, 0.08039931,
       0.08360218, 0.08786507, 0.39908215, 0.36157067, 0.        ,
       0.05660069, 0.05114291, 0.11067631, 0.17586358, 0.12369077,
       0.18980145, 0.05848854], [0.08801484, 0.08231864, 0.09975021, 0.        , 0.05362438,
       0.16819413, 0.17101684, 0.        , 0.05789711, 0.        ,
       0.05411692, 0.05431469, 0.10965804, 0.05303881, 0.055714  ,
       0.05225351, 0.08424191, 0.07696801, 0.07968477, 0.08288975,
       0.08194226, 0.07842985, 0.3879578 , 0.36186061, 0.        ,
       0.05144964, 0.05498216, 0.11155357, 0.18209763, 0.13629657,
       0.19962904, 0.05506933], [0.0920692 , 0.07333922, 0.11080993, 0.        , 0.05417629,
       0.17967128, 0.17973738, 0.        , 0.05445394, 0.        ,
       0.05087509, 0.05181963, 0.10658318, 0.05519061, 0.05409869,
       0.05396776, 0.07989759, 0.07645285, 0.07874112, 0.07695702,
       0.08255935, 0.07515861, 0.35753036, 0.39209517, 0.        ,
       0.05309811, 0.05564704, 0.10247809, 0.18174215, 0.13527303,
       0.19293484, 0.05162003], [0.08601046, 0.08015285, 0.10878901, 0.        , 0.05308549,
       0.16533874, 0.17699337, 0.        , 0.05593221, 0.        ,
       0.05509587, 0.05216291, 0.10558163, 0.05389671, 0.05309424,
       0.05498246, 0.08543478, 0.08627147, 0.07780776, 0.07795114,
       0.08741059, 0.08859865, 0.37783737, 0.38369454, 0.        ,
       0.04977658, 0.05124457, 0.1148603 , 0.17253165, 0.12341142,
       0.20042279, 0.05136201], [0.0908366 , 0.07666883, 0.11224969, 0.        , 0.05501134,
       0.17390065, 0.1783617 , 0.        , 0.05497215, 0.        ,
       0.05660619, 0.05400105, 0.11176092, 0.05404653, 0.0528491 ,
       0.05581173, 0.07620223, 0.08187481, 0.08257553, 0.07806832,
       0.07828518, 0.0808455 , 0.36750602, 0.39307643, 0.        ,
       0.05319844, 0.05147088, 0.10590284, 0.17143618, 0.12998724,
       0.17296928, 0.0546619 ], [0.09112275, 0.0840961 , 0.1127898 , 0.        , 0.05700617,
       0.18162711, 0.18737522, 0.        , 0.05152925, 0.        ,
       0.0520035 , 0.05610683, 0.10962797, 0.05556312, 0.05546691,
       0.05567799, 0.07700162, 0.07838687, 0.07984308, 0.08191177,
       0.07652382, 0.07848763, 0.36042805, 0.36576326, 0.        ,
       0.0537523 , 0.05286398, 0.11032764, 0.18581082, 0.13512546,
       0.20371422, 0.05530535], [0.10910067, 0.09472539, 0.13471281, 0.        , 0.06662027,
       0.21280671, 0.22454694, 0.        , 0.06564091, 0.        ,
       0.0642197 , 0.06683948, 0.132127  , 0.06300726, 0.06339169,
       0.06505056, 0.09791929, 0.09836788, 0.09781721, 0.0951767 ,
       0.09421733, 0.09656968, 0.47069079, 0.4529368 , 0.        ,
       0.06439363, 0.06568264, 0.12562887, 0.21514846, 0.16773955,
       0.22786156, 0.06524645], [0.11236306, 0.09410045, 0.13831218, 0.        , 0.06345932,
       0.2283182 , 0.22156018, 0.        , 0.06481731, 0.        ,
       0.06682042, 0.06862372, 0.13363019, 0.06422626, 0.06704041,
       0.06331512, 0.10019355, 0.10224309, 0.0959293 , 0.10133557,
       0.10277789, 0.10036987, 0.44810416, 0.45414334, 0.        ,
       0.06954766, 0.06508299, 0.13591499, 0.21913263, 0.16032083,
       0.22717612, 0.0635777 ], [0.1061445 , 0.10226939, 0.13062902, 0.        , 0.06371619,
       0.22086852, 0.22188748, 0.        , 0.06646184, 0.        ,
       0.06827259, 0.0654947 , 0.13422375, 0.06770796, 0.06766885,
       0.06229885, 0.10002383, 0.09581314, 0.09991545, 0.10039577,
       0.09822158, 0.09657457, 0.48452   , 0.47531905, 0.        ,
       0.06343942, 0.06731732, 0.12715999, 0.21353155, 0.16374433,
       0.22920614, 0.06566104], [0.10699565, 0.09992681, 0.13587492, 0.        , 0.06739625,
       0.20531852, 0.22205073, 0.        , 0.06870496, 0.        ,
       0.06459759, 0.06878577, 0.13447976, 0.06651379, 0.06348992,
       0.06661452, 0.09523469, 0.09787891, 0.09955466, 0.1033745 ,
       0.09940577, 0.09880741, 0.45745798, 0.46670644, 0.        ,
       0.06482826, 0.06494103, 0.13448296, 0.22249868, 0.15842629,
       0.22581729, 0.06551256], [0.10807368, 0.10157858, 0.1325336 , 0.        , 0.06761411,
       0.21574731, 0.21365339, 0.        , 0.06804974, 0.        ,
       0.06441934, 0.06916943, 0.12860564, 0.0645459 , 0.06746029,
       0.06863065, 0.10325515, 0.1012817 , 0.09130101, 0.09484775,
       0.09901393, 0.09801411, 0.46232953, 0.47172508, 0.        ,
       0.06550258, 0.06590619, 0.12632297, 0.22616022, 0.16445088,
       0.22689708, 0.06221088], [0.10805196, 0.09238849, 0.12332907, 0.        , 0.06461509,
       0.22930659, 0.23087866, 0.        , 0.06712835, 0.        ,
       0.06356548, 0.06933506, 0.12660219, 0.06650583, 0.06545707,
       0.06357513, 0.10085417, 0.09873388, 0.10230559, 0.10108483,
       0.10165869, 0.10116341, 0.44342854, 0.44509162, 0.        ,
       0.06539728, 0.06539627, 0.13352227, 0.21674487, 0.15874471,
       0.2342799 , 0.06549894], [0.08456019, 0.07729627, 0.112093  , 0.        , 0.05526648,
       0.18324415, 0.19187206, 0.        , 0.05148515, 0.        ,
       0.05534314, 0.05616065, 0.11232704, 0.05016128, 0.04989951,
       0.05601341, 0.08547035, 0.07958522, 0.0785796 , 0.08167662,
       0.07872789, 0.08372717, 0.38210706, 0.36832007, 0.        ,
       0.05491155, 0.05132237, 0.10919597, 0.18563036, 0.13200024,
       0.19315572, 0.05437819], [0.09385829, 0.08046588, 0.10650593, 0.        , 0.0531995 ,
       0.17987264, 0.18789393, 0.        , 0.05384941, 0.        ,
       0.05403362, 0.05152307, 0.10426868, 0.05368594, 0.05771328,
       0.05405097, 0.0766417 , 0.08406787, 0.0885965 , 0.08300351,
       0.07528698, 0.07902276, 0.36120388, 0.35644368, 0.        ,
       0.05668002, 0.05338114, 0.10609953, 0.16947391, 0.14368084,
       0.180546  , 0.05264242], [0.0912356 , 0.08698973, 0.11095817, 0.        , 0.04982577,
       0.18349746, 0.17208832, 0.        , 0.05246526, 0.        ,
       0.05539241, 0.05329745, 0.11306287, 0.05168917, 0.05360301,
       0.05395036, 0.08664272, 0.08542999, 0.07947323, 0.08268237,
       0.07876796, 0.07980769, 0.37432845, 0.38362257, 0.        ,
       0.05451351, 0.05569528, 0.10805206, 0.18867252, 0.13198951,
       0.19721648, 0.05472937], [0.09598765, 0.08137315, 0.10593034, 0.        , 0.0559242 ,
       0.18042402, 0.17845281, 0.        , 0.05233476, 0.        ,
       0.05039971, 0.05684373, 0.10904199, 0.05289435, 0.05192544,
       0.05484707, 0.08040655, 0.08032228, 0.08152782, 0.07527305,
       0.08287549, 0.08407603, 0.36905302, 0.38817662, 0.        ,
       0.05223722, 0.05349705, 0.11186252, 0.18072921, 0.13279266,
       0.17998023, 0.05722962], [0.08720748, 0.08034083, 0.11078035, 0.        , 0.05331842,
       0.19475555, 0.18164194, 0.        , 0.05477384, 0.        ,
       0.05032771, 0.05497418, 0.10692219, 0.05079049, 0.05276061,
       0.05445571, 0.08121497, 0.08079164, 0.08271641, 0.08339534,
       0.08141941, 0.08029132, 0.4009648 , 0.38311225, 0.        ,
       0.05645644, 0.05551245, 0.10530139, 0.17713985, 0.13230121,
       0.18533934, 0.05362848], [0.09304038, 0.07541045, 0.10902449, 0.        , 0.05020673,
       0.18348127, 0.18481952, 0.        , 0.04994544, 0.        ,
       0.05438159, 0.05422558, 0.10693414, 0.05171841, 0.05159468,
       0.0566146 , 0.08284414, 0.07428623, 0.08337731, 0.07917966,
       0.08217288, 0.08075275, 0.38824721, 0.38014441, 0.        ,
       0.05645029, 0.0523308 , 0.11083761, 0.18242352, 0.12676543,
       0.19461246, 0.05682582]])
    list_fu_he_Q = np.array([
        [0.05938436, 0.03942027, 0.0798421 , 0.        , 0.01999451,
       0.09968516, 0.09732844, 0.        , 0.01901276, 0.        ,
       0.03365671, 0.034174  , 0.07916808, 0.0096877 , 0.01935384,
       0.01937047, 0.03942448, 0.03900865, 0.03811965, 0.03853418,
       0.03914674, 0.04919586, 0.19036907, 0.19917297, 0.        ,
       0.02471553, 0.01968209, 0.067642  , 0.57941445, 0.06772954,
       0.09801538, 0.03958708], [0.05757595, 0.03918021, 0.07899784, 0.        , 0.01982919,
       0.09734399, 0.09788697, 0.        , 0.01903561, 0.        ,
       0.0347016 , 0.03463924, 0.07821641, 0.00973154, 0.01966208,
       0.01951514, 0.03921456, 0.03826939, 0.03864625, 0.03854363,
       0.03872448, 0.04880649, 0.19775698, 0.19697143, 0.        ,
       0.0245186 , 0.01982056, 0.06717903, 0.58042003, 0.06747468,
       0.09738462, 0.03875171], [0.05873819, 0.03931114, 0.07863411, 0.        , 0.01975544,
       0.0969674 , 0.09702371, 0.        , 0.0192131 , 0.        ,
       0.03397198, 0.03432774, 0.07679329, 0.00987297, 0.01942131,
       0.01947415, 0.038793  , 0.03894574, 0.03970452, 0.03951096,
       0.03854369, 0.04878743, 0.19509426, 0.19662227, 0.        ,
       0.0243171 , 0.01928307, 0.06855285, 0.57657508, 0.06761464,
       0.09690646, 0.03845917], [0.05798873, 0.03921016, 0.07766852, 0.        , 0.01963025,
       0.09642705, 0.09737799, 0.        , 0.01933828, 0.        ,
       0.03412737, 0.03364242, 0.07783936, 0.00977935, 0.01971485,
       0.01936586, 0.03891413, 0.03956709, 0.03863502, 0.03940059,
       0.03888637, 0.048286  , 0.19461369, 0.19353806, 0.        ,
       0.02427338, 0.01932379, 0.06789724, 0.57982861, 0.06876967,
       0.09898326, 0.03919745], [0.05883319, 0.0392498 , 0.07819085, 0.        , 0.01934114,
       0.09861146, 0.0959591 , 0.        , 0.01963607, 0.        ,
       0.03407244, 0.03391512, 0.07698331, 0.00972345, 0.0192632 ,
       0.01969828, 0.03883674, 0.03917905, 0.03926185, 0.03867607,
       0.03964738, 0.04955291, 0.19764375, 0.19501934, 0.        ,
       0.02445475, 0.01954064, 0.06910452, 0.5904206 , 0.0692489 ,
       0.0972637 , 0.03896471], [0.05842904, 0.03876262, 0.07777919, 0.        , 0.01946651,
       0.0977111 , 0.09777144, 0.        , 0.01966704, 0.        ,
       0.03420735, 0.03436703, 0.07709307, 0.00974813, 0.01959926,
       0.01954911, 0.03883371, 0.038836  , 0.03868429, 0.03923069,
       0.03948148, 0.04851001, 0.19549813, 0.194472  , 0.        ,
       0.02431927, 0.0198024 , 0.06798593, 0.5919202 , 0.06874067,
       0.09692735, 0.03920988], [0.05982143, 0.03970635, 0.08087239, 0.        , 0.01986882,
       0.09968892, 0.10117046, 0.        , 0.01986865, 0.        ,
       0.03492579, 0.0346738 , 0.08002303, 0.00997734, 0.02013258,
       0.01985313, 0.04043655, 0.03955172, 0.04056193, 0.04003671,
       0.0394513 , 0.0504205 , 0.20006315, 0.19959348, 0.        ,
       0.02476981, 0.01969262, 0.07014184, 0.60229011, 0.06958816,
       0.10100326, 0.04055266], [0.06062907, 0.04005901, 0.07998286, 0.        , 0.01998493,
       0.10073051, 0.09960315, 0.        , 0.01970164, 0.        ,
       0.03500334, 0.0350868 , 0.07958303, 0.00999427, 0.01997557,
       0.01992745, 0.039687  , 0.03993439, 0.04050017, 0.03953919,
       0.03977833, 0.04978103, 0.20259158, 0.19773481, 0.        ,
       0.02505345, 0.02014022, 0.07059938, 0.59150586, 0.06987763,
       0.1006787 , 0.04032248], [0.06031564, 0.04002435, 0.07887793, 0.        , 0.01981972,
       0.0993858 , 0.09936525, 0.        , 0.01970466, 0.        ,
       0.03509143, 0.03501822, 0.08059642, 0.00982122, 0.01981162,
       0.02006579, 0.04015816, 0.04008277, 0.03980892, 0.04020415,
       0.04022073, 0.04994569, 0.19969109, 0.20125349, 0.        ,
       0.02514989, 0.02000134, 0.06995454, 0.59624666, 0.07004998,
       0.09908293, 0.04022105], [0.05996532, 0.03992157, 0.07981541, 0.        , 0.01984164,
       0.09991262, 0.10095033, 0.        , 0.02000004, 0.        ,
       0.0349219 , 0.03503829, 0.08005114, 0.0100434 , 0.02003346,
       0.01985601, 0.04048271, 0.03990201, 0.04041954, 0.03991198,
       0.03964356, 0.04958512, 0.20063709, 0.19827394, 0.        ,
       0.02508473, 0.02008288, 0.07026698, 0.59822901, 0.06993558,
       0.10017541, 0.03953046], [0.05963668, 0.04047764, 0.07991485, 0.        , 0.01998563,
       0.1006275 , 0.09978285, 0.        , 0.02006427, 0.        ,
       0.03496358, 0.03512626, 0.08093154, 0.01006834, 0.02025966,
       0.02011707, 0.04005613, 0.03981839, 0.0398856 , 0.04037368,
       0.03996093, 0.05030077, 0.20003063, 0.1991383 , 0.        ,
       0.02492962, 0.01988808, 0.07061413, 0.59618123, 0.070503  ,
       0.10029291, 0.03988267], [0.05978026, 0.03994418, 0.07957499, 0.        , 0.0201287 ,
       0.09966188, 0.10028183, 0.        , 0.01989544, 0.        ,
       0.03502855, 0.03544497, 0.0799648 , 0.00990169, 0.01993217,
       0.02013592, 0.03992863, 0.03967202, 0.03998236, 0.03943919,
       0.03996073, 0.04931901, 0.19931373, 0.20083727, 0.        ,
       0.0250304 , 0.01986062, 0.06920877, 0.59516662, 0.07095004,
       0.09972898, 0.04021806], [0.06157839, 0.04057396, 0.08268101, 0.        , 0.02026068,
       0.10298246, 0.10268825, 0.        , 0.02073929, 0.        ,
       0.03604502, 0.03570379, 0.08271132, 0.01035319, 0.02071373,
       0.02032532, 0.04056861, 0.04137518, 0.04099699, 0.04058088,
       0.04110243, 0.05123639, 0.20618663, 0.2071679 , 0.        ,
       0.02585119, 0.02040357, 0.07095678, 0.61257001, 0.07232985,
       0.10326402, 0.04144875],[0.06185523, 0.04097871, 0.08274373, 0.        , 0.02061183,
       0.10276397, 0.10205225, 0.        , 0.02034505, 0.        ,
       0.03539522, 0.03529298, 0.08245682, 0.01032274, 0.02050347,
       0.02066341, 0.04088488, 0.04102056, 0.04100263, 0.04173605,
       0.04170121, 0.05146322, 0.20378068, 0.20772574, 0.        ,
       0.02562203, 0.02042814, 0.07053639, 0.61296383, 0.07175281,
       0.10398032, 0.04125297], [0.06114604, 0.04045996, 0.08268927, 0.        , 0.02014786,
       0.10300622, 0.10244475, 0.        , 0.02050732, 0.        ,
       0.03594575, 0.03576379, 0.08170881, 0.01035353, 0.02048971,
       0.02044437, 0.04096898, 0.04131617, 0.04052545, 0.0411899 ,
       0.04116109, 0.05106845, 0.2059301 , 0.20915919, 0.        ,
       0.02563481, 0.02024796, 0.07197387, 0.60633842, 0.07133577,
       0.10205918, 0.04090561], [0.06122519, 0.0414351 , 0.08126772, 0.        , 0.02055284,
       0.10458769, 0.10182915, 0.        , 0.02047717, 0.        ,
       0.03611946, 0.03588457, 0.08297181, 0.01036992, 0.02038269,
       0.02024321, 0.04069004, 0.04094075, 0.04147616, 0.04121963,
       0.04091902, 0.05020931, 0.20633732, 0.2060593 , 0.        ,
       0.02586416, 0.02041681, 0.07152564, 0.60925545, 0.07186225,
       0.10167363, 0.0408908 ], [0.06128043, 0.04045822, 0.08214218, 0.        , 0.02045237,
       0.10331728, 0.10148957, 0.        , 0.02080523, 0.        ,
       0.03600496, 0.03625265, 0.08200855, 0.0102183 , 0.02040108,
       0.02072102, 0.04088951, 0.0409079 , 0.04083128, 0.04140052,
       0.04062735, 0.05092311, 0.20242189, 0.2024145 , 0.        ,
       0.02562452, 0.02062424, 0.07204018, 0.61022687, 0.0727106 ,
       0.10270602, 0.04052821], [0.06165844, 0.04073801, 0.08190902, 0.        , 0.02048817,
       0.10383918, 0.10303145, 0.        , 0.0205175 , 0.        ,
       0.03627572, 0.03559198, 0.0822755 , 0.01025362, 0.02052516,
       0.02055723, 0.04096573, 0.04130393, 0.04136436, 0.04152493,
       0.04105102, 0.05174564, 0.20191241, 0.20328117, 0.        ,
       0.02549103, 0.02013771, 0.07286424, 0.62485656, 0.07167654,
       0.1008198 , 0.0411939 ], [0.05942824, 0.0399048 , 0.07937293, 0.        , 0.02000689,
       0.10103246, 0.1005895 , 0.        , 0.02013561, 0.        ,
       0.03512946, 0.03498033, 0.08011523, 0.01013941, 0.01972567,
       0.01997346, 0.03964108, 0.04004005, 0.03993747, 0.03982383,
       0.04007211, 0.04928167, 0.19983874, 0.20028015, 0.        ,
       0.02512312, 0.02005051, 0.07030072, 0.59789797, 0.06952983,
       0.09947792, 0.03991795], [0.05924416, 0.03979483, 0.07972099, 0.        , 0.02003858,
       0.1010313 , 0.10100474, 0.        , 0.02007531, 0.        ,
       0.03468654, 0.03499546, 0.0795269 , 0.0100234 , 0.01977149,
       0.02023138, 0.04049457, 0.04011592, 0.04003191, 0.0399973 ,
       0.03986002, 0.05033658, 0.20116084, 0.20053199, 0.        ,
       0.02513405, 0.0199931 , 0.07019719, 0.59728866, 0.07059046,
       0.09921739, 0.03991514], [0.06003496, 0.04013366, 0.08017253, 0.        , 0.0198574 ,
       0.09940377, 0.10046814, 0.        , 0.01986266, 0.        ,
       0.03505611, 0.03497112, 0.08049995, 0.00998297, 0.02019599,
       0.02003604, 0.0398308 , 0.04007762, 0.0396275 , 0.0402821 ,
       0.03951129, 0.05007559, 0.1991797 , 0.19737218, 0.        ,
       0.02497769, 0.02007008, 0.07031811, 0.60053458, 0.0693573 ,
       0.10066741, 0.03992689], [0.05964985, 0.0394729 , 0.07950497, 0.        , 0.01999304,
       0.10013809, 0.10027909, 0.        , 0.02009441, 0.        ,
       0.03490157, 0.03489911, 0.08038507, 0.01006448, 0.01981672,
       0.02003309, 0.04031669, 0.04047537, 0.03974886, 0.04028269,
       0.03928294, 0.04979241, 0.20206968, 0.20015007, 0.        ,
       0.02490013, 0.0199204 , 0.07060768, 0.60043205, 0.07061732,
       0.09979275, 0.04002117], [0.05967296, 0.03981068, 0.08035899, 0.        , 0.02009813,
       0.10156809, 0.10113579, 0.        , 0.0197779 , 0.        ,
       0.03504918, 0.03504933, 0.08050686, 0.0101536 , 0.02026078,
       0.01986957, 0.03998365, 0.0396271 , 0.04011631, 0.03994622,
       0.04003605, 0.05026721, 0.19984698, 0.20077544, 0.        ,
       0.02530884, 0.02017881, 0.06927481, 0.5942947 , 0.06986042,
       0.09963648, 0.04036052], [0.05991955, 0.03955896, 0.08053259, 0.        , 0.02015539,
       0.09908059, 0.09999936, 0.        , 0.0201156 , 0.        ,
       0.03536613, 0.03515888, 0.08028093, 0.01001322, 0.02014639,
       0.01996993, 0.03983656, 0.04044909, 0.04029595, 0.03970119,
       0.04015024, 0.04995517, 0.20141107, 0.20130475, 0.        ,
       0.02495645, 0.02006575, 0.06993318, 0.6075392 , 0.0701815 ,
       0.09917871, 0.04037286]])

    _test_agent_1(env, agent, list_feng_11, list_feng_26, list_fu_he, list_fu_he_Q)
elif kk == 2:
    list_feng_11 = np.array([1.5869341060316773, 1.5251573343253797, 1.4509670023679, 1.4095275723088085, 1.446228156898842, 1.509339607265972, 1.5432883823721666, 1.4580270651356457, 1.4997327683304167, 1.5365828234610845, 1.656484505213365, 1.3959810405672837, 1.55918287410902, 1.5181833821767754, 1.5209567668166482, 1.544177535500228, 1.4918432194192612, 1.6510043510767072, 1.5283899043095246, 1.6000433587532656, 1.6316557939364955, 1.424095756526058, 1.5030260769952672, 1.4132699046214015])
    list_feng_26 = np.array([1.0861729604378176, 0.8907649293168796, 0.9932949808689355, 1.0736234072020139, 1.0931371563714571, 0.9035006535434696, 0.9152820129751831, 1.009536664204697, 0.9323703688388457, 0.8482227850012084, 1.1504487066372244, 0.847518628212424, 0.9464160392348846, 0.9427043279586038, 0.9938478953366569, 0.9606195545741965, 0.976509778304762, 1.0050821344019667, 1.105898459465684, 1.0186170549630706, 1.1316791645598576, 0.9729633517127605, 0.9205704372696882, 1.0292583634857322])
    list_fu_he = np.array([
        [0.06739778, 0.06763283, 0.07614136, 0.        , 0.04500004,
       0.12710168, 0.12436806, 0.        , 0.04774207, 0.        ,
       0.03996534, 0.03733335, 0.08275598, 0.04359583, 0.03822918,
       0.03945176, 0.06419804, 0.06019482, 0.06722285, 0.07139392,
       0.06499425, 0.06158721, 0.30309027, 0.25686119, 0.        ,
       0.04252052, 0.04519047, 0.08132476, 0.15277446, 0.09309946,
       0.15305335, 0.03822745], [0.07576021, 0.06421038, 0.07696441, 0.        , 0.03937159,
       0.12634318, 0.15068718, 0.        , 0.04333672, 0.        ,
       0.0457358 , 0.0441219 , 0.07399034, 0.03923635, 0.03938418,
       0.03843209, 0.06195349, 0.06388021, 0.06892018, 0.05968718,
       0.0618109 , 0.06333586, 0.26245267, 0.29878881, 0.        ,
       0.0422897 , 0.04238165, 0.08581804, 0.14119537, 0.09504894,
       0.14804322, 0.04211207], [0.06595821, 0.0602879 , 0.09012206, 0.        , 0.04417066,
       0.13883635, 0.15019513, 0.        , 0.04394872, 0.        ,
       0.03720978, 0.04212715, 0.08970092, 0.0402006 , 0.04026547,
       0.04071625, 0.05802986, 0.06381571, 0.06168958, 0.06117625,
       0.06565718, 0.06341779, 0.28953451, 0.26894313, 0.        ,
       0.04394063, 0.04231226, 0.08422088, 0.1414191 , 0.1119004 ,
       0.14246538, 0.04317943], [0.07301075, 0.06428356, 0.08791565, 0.        , 0.04347446,
       0.14437919, 0.14514476, 0.        , 0.04369328, 0.        ,
       0.04353686, 0.0443227 , 0.08665809, 0.04206497, 0.0400315 ,
       0.04362833, 0.0616386 , 0.06438519, 0.05838056, 0.06451371,
       0.06424779, 0.06853178, 0.27860905, 0.27951885, 0.        ,
       0.04406326, 0.04073435, 0.09204631, 0.14840089, 0.10530914,
       0.15295056, 0.0409977 ], [0.0668305 , 0.06321789, 0.07776335, 0.        , 0.03812618,
       0.14732025, 0.14340412, 0.        , 0.03873242, 0.        ,
       0.04405757, 0.0448964 , 0.08154446, 0.04016171, 0.04303169,
       0.04155685, 0.06486113, 0.06010286, 0.06371954, 0.06504712,
       0.05706812, 0.06839802, 0.2612956 , 0.28245701, 0.        ,
       0.04127082, 0.03839985, 0.08250492, 0.14950965, 0.10671349,
       0.15154104, 0.04465692], [0.07232815, 0.06341748, 0.08426155, 0.        , 0.04100296,
       0.13454581, 0.14186502, 0.        , 0.04198497, 0.        ,
       0.03986211, 0.04336658, 0.08440012, 0.04010197, 0.04293924,
       0.04261339, 0.06596592, 0.05993532, 0.06666301, 0.06507182,
       0.05985664, 0.06765691, 0.31997299, 0.26797145, 0.        ,
       0.04163814, 0.03980643, 0.08389146, 0.13422151, 0.10836762,
       0.1321016 , 0.04262074], [0.0832036 , 0.08082804, 0.09990003, 0.        , 0.0527539 ,
       0.16583623, 0.18426729, 0.        , 0.05526424, 0.        ,
       0.05101725, 0.05379652, 0.11386623, 0.04982341, 0.05472545,
       0.05140556, 0.08612442, 0.08652631, 0.07616294, 0.08277078,
       0.08064376, 0.08333451, 0.36921977, 0.41006602, 0.        ,
       0.05165365, 0.05580641, 0.1009507 , 0.1700086 , 0.1350953 ,
       0.18343619, 0.05434029], [0.09098616, 0.08368596, 0.10912396, 0.        , 0.05329065,
       0.17867338, 0.17536983, 0.        , 0.05531901, 0.        ,
       0.05342734, 0.05511249, 0.10725806, 0.05526022, 0.05512675,
       0.05435599, 0.08030879, 0.08110555, 0.08739559, 0.07679594,
       0.07896468, 0.08329715, 0.40595855, 0.39425822, 0.        ,
       0.05397177, 0.05276825, 0.11163752, 0.1781633 , 0.13266516,
       0.18297202, 0.05482815], [0.08878871, 0.08321321, 0.10386112, 0.        , 0.05556106,
       0.17398313, 0.1904767 , 0.        , 0.05518781, 0.        ,
       0.05377815, 0.05438894, 0.10563757, 0.05401922, 0.05520401,
       0.05313043, 0.08335171, 0.07756256, 0.07752477, 0.0809715 ,
       0.07715455, 0.08139172, 0.36572465, 0.36083904, 0.        ,
       0.05308828, 0.0524853 , 0.1090808 , 0.17174646, 0.12839514,
       0.2028139 , 0.05440081], [0.09193082, 0.08026627, 0.10869019, 0.        , 0.05208122,
       0.1820154 , 0.17771765, 0.        , 0.04901098, 0.        ,
       0.05803866, 0.05590788, 0.11005091, 0.05729087, 0.05106   ,
       0.05266312, 0.08181398, 0.08246377, 0.08035584, 0.07926663,
       0.08127515, 0.07703087, 0.36116914, 0.35196517, 0.        ,
       0.05645765, 0.05559206, 0.1091395 , 0.1857234 , 0.13926264,
       0.19595815, 0.05307706], [0.09299702, 0.07639913, 0.11165133, 0.        , 0.05326006,
       0.18164681, 0.1907869 , 0.        , 0.05107126, 0.        ,
       0.05568796, 0.05668061, 0.11237379, 0.05336086, 0.05758963,
       0.05649176, 0.08057781, 0.08032911, 0.08340036, 0.07955282,
       0.07986814, 0.08004779, 0.37841192, 0.3912427 , 0.        ,
       0.05255563, 0.05569333, 0.11304625, 0.17470808, 0.14402687,
       0.19627927, 0.05069342], [0.08991316, 0.07732291, 0.10856543, 0.        , 0.05637912,
       0.17231073, 0.17998126, 0.        , 0.0537486 , 0.        ,
       0.05330784, 0.05676383, 0.10621921, 0.05718098, 0.05631796,
       0.05791384, 0.08582554, 0.0812306 , 0.08297482, 0.07954876,
       0.08550154, 0.07922534, 0.36832854, 0.3719335 , 0.        ,
       0.05255921, 0.05266181, 0.10538525, 0.17298703, 0.13026566,
       0.19221089, 0.05420697], [0.11501841, 0.09953168, 0.12859553, 0.        , 0.06518446,
       0.2225706 , 0.22263289, 0.        , 0.06484892, 0.        ,
       0.06540612, 0.06880127, 0.13523997, 0.06834395, 0.06517329,
       0.06567478, 0.09882705, 0.09772173, 0.10067176, 0.100419  ,
       0.09464283, 0.10008154, 0.45342285, 0.47092692, 0.        ,
       0.06306639, 0.06525921, 0.14074105, 0.20348577, 0.16333228,
       0.23432405, 0.06753714], [0.10937501, 0.09795724, 0.13534509, 0.        , 0.06278169,
       0.22151212, 0.21589523, 0.        , 0.06684624, 0.        ,
       0.06711462, 0.06554852, 0.1302726 , 0.06710557, 0.06482525,
       0.066798  , 0.09690619, 0.09378626, 0.09637932, 0.09412284,
       0.10157499, 0.09701712, 0.46641414, 0.46024236, 0.        ,
       0.068028  , 0.0660322 , 0.13305822, 0.21500027, 0.16441126,
       0.22018878, 0.0691775 ], [0.11519273, 0.10015374, 0.13950162, 0.        , 0.06648809,
       0.23157071, 0.21491667, 0.        , 0.06545908, 0.        ,
       0.06425932, 0.06449994, 0.12211525, 0.06487501, 0.06382927,
       0.06879593, 0.1021238 , 0.10344275, 0.10292013, 0.10027828,
       0.10119439, 0.09813844, 0.43977254, 0.45677978, 0.        ,
       0.06858623, 0.06602399, 0.13285255, 0.22043276, 0.16320003,
       0.22620831, 0.06725156], [0.1134185 , 0.10090866, 0.12492715, 0.        , 0.06596852,
       0.21051997, 0.20831694, 0.        , 0.06448783, 0.        ,
       0.06607561, 0.06730115, 0.12508134, 0.06445984, 0.06559016,
       0.06347692, 0.10177323, 0.09685156, 0.09393747, 0.09826073,
       0.09374869, 0.10658829, 0.45960505, 0.46418719, 0.        ,
       0.06643339, 0.06347115, 0.12640883, 0.22030417, 0.16214937,
       0.22611552, 0.06473611], [0.10868669, 0.09664673, 0.12615896, 0.        , 0.06638955,
       0.21928494, 0.2211363 , 0.        , 0.06718056, 0.        ,
       0.06649348, 0.06423601, 0.13193092, 0.06383567, 0.06797683,
       0.06556513, 0.10075618, 0.09906232, 0.10289686, 0.10210173,
       0.09783464, 0.0990368 , 0.47544708, 0.44861647, 0.        ,
       0.06473016, 0.0652862 , 0.12833198, 0.22291707, 0.16863011,
       0.23675462, 0.06656466], [0.11474345, 0.10183638, 0.13128587, 0.        , 0.06902707,
       0.22444207, 0.20917509, 0.        , 0.0660729 , 0.        ,
       0.06283667, 0.06646811, 0.13587978, 0.06648935, 0.06715053,
       0.06872843, 0.09705611, 0.10061416, 0.10182045, 0.09641203,
       0.10077401, 0.09739854, 0.46785579, 0.46593885, 0.        ,
       0.06402675, 0.06200326, 0.12361998, 0.21881085, 0.16860148,
       0.23689118, 0.06639647], [0.08784964, 0.08168942, 0.10977842, 0.        , 0.05560204,
       0.18282769, 0.17417146, 0.        , 0.0521317 , 0.        ,
       0.05035106, 0.05346021, 0.1057017 , 0.05357199, 0.0527543 ,
       0.05214177, 0.08253238, 0.0837963 , 0.08310949, 0.08261516,
       0.08345434, 0.07976505, 0.38181992, 0.38721059, 0.        ,
       0.05454577, 0.054298  , 0.10691344, 0.19199189, 0.14144758,
       0.18638324, 0.05471335], [0.09158995, 0.07667011, 0.10421675, 0.        , 0.05422787,
       0.18511657, 0.18515698, 0.        , 0.05335736, 0.        ,
       0.05179423, 0.05449268, 0.10862687, 0.05865026, 0.05093291,
       0.05773018, 0.07584915, 0.07898574, 0.08235732, 0.08603587,
       0.08250551, 0.08316329, 0.38688009, 0.36806112, 0.        ,
       0.05428951, 0.05199645, 0.10146991, 0.18190242, 0.14057114,
       0.1960732 , 0.05574077], [0.08949727, 0.08221139, 0.11339254, 0.        , 0.0502717 ,
       0.18665201, 0.17134505, 0.        , 0.05434153, 0.        ,
       0.05344318, 0.0534935 , 0.10466038, 0.05295006, 0.05289745,
       0.05562215, 0.07558194, 0.08206261, 0.0793263 , 0.07684662,
       0.08191326, 0.07873333, 0.4083915 , 0.36929657, 0.        ,
       0.05383866, 0.05456926, 0.11339286, 0.17893172, 0.1360994 ,
       0.19349122, 0.054967  ], [0.08332418, 0.08213908, 0.10762532, 0.        , 0.05531778,
       0.17086563, 0.18443205, 0.        , 0.05123381, 0.        ,
       0.05444215, 0.05661494, 0.10967208, 0.05692423, 0.05691552,
       0.05549633, 0.07428282, 0.08134809, 0.07932166, 0.08433087,
       0.0804681 , 0.07888314, 0.38104132, 0.38378616, 0.        ,
       0.05783635, 0.05573706, 0.11196413, 0.18681337, 0.13729639,
       0.19302242, 0.0557695 ], [0.09510091, 0.08045717, 0.10844856, 0.        , 0.05365517,
       0.17212505, 0.18221823, 0.        , 0.05322731, 0.        ,
       0.0532458 , 0.05538802, 0.10986333, 0.0521706 , 0.05086619,
       0.0516976 , 0.07754631, 0.08021317, 0.08015226, 0.08183381,
       0.08064611, 0.08160191, 0.36709515, 0.36530237, 0.        ,
       0.05560563, 0.05597805, 0.11030058, 0.18578904, 0.13706306,
       0.17897156, 0.05463894], [0.09034605, 0.08738429, 0.11452747, 0.        , 0.05297868,
       0.17890641, 0.17552206, 0.        , 0.0551096 , 0.        ,
       0.05337761, 0.05445107, 0.10598464, 0.05698106, 0.05401496,
       0.05363547, 0.08494678, 0.07754713, 0.07914091, 0.08287596,
       0.08144805, 0.08051705, 0.37268033, 0.38373891, 0.        ,
       0.05124532, 0.05251322, 0.11059192, 0.18134841, 0.13516845,
       0.18670221, 0.05592676]])
    list_fu_he_Q = np.array([
        [0.0579211 , 0.03963868, 0.07803698, 0.        , 0.01974106,
       0.09519068, 0.09651281, 0.        , 0.01965064, 0.        ,
       0.03338619, 0.03331567, 0.07762538, 0.00998499, 0.0191219 ,
       0.01987362, 0.03937574, 0.03897777, 0.03835888, 0.03989691,
       0.03829545, 0.04973889, 0.19884277, 0.1966943 , 0.        ,
       0.0239604 , 0.01981267, 0.06862501, 0.59419624, 0.06937414,
       0.09825962, 0.03963714], [0.05896926, 0.03881529, 0.07825129, 0.        , 0.01938646,
       0.09844856, 0.09781637, 0.        , 0.01951779, 0.        ,
       0.03460343, 0.03388837, 0.07729457, 0.00963705, 0.01932449,
       0.01927711, 0.03897209, 0.03961825, 0.0382366 , 0.03936579,
       0.03914581, 0.04922614, 0.19774004, 0.19937145, 0.        ,
       0.02485701, 0.0196516 , 0.0685871 , 0.57980557, 0.06846041,
       0.09754545, 0.03963677], [0.05926021, 0.03855318, 0.07922781, 0.        , 0.01970552,
       0.09693514, 0.09762038, 0.        , 0.01926081, 0.        ,
       0.03396336, 0.03373663, 0.0774572 , 0.00978366, 0.01944196,
       0.01941539, 0.03937449, 0.03905077, 0.03929058, 0.03833928,
       0.03916865, 0.04967419, 0.19746245, 0.19415099, 0.        ,
       0.02453784, 0.01936074, 0.06760687, 0.58656451, 0.06905447,
       0.09653136, 0.0393079 ], [0.05732601, 0.03862417, 0.07634798, 0.        , 0.01949529,
       0.09892247, 0.09639427, 0.        , 0.0194716 , 0.        ,
       0.0334721 , 0.03355677, 0.07718323, 0.00972216, 0.01951866,
       0.01959014, 0.03907833, 0.0390913 , 0.03875225, 0.03940953,
       0.03909866, 0.04839492, 0.19550526, 0.19558434, 0.        ,
       0.02410413, 0.01948898, 0.06924096, 0.58384659, 0.06807235,
       0.09900873, 0.03950702], [0.05856378, 0.0392956 , 0.07732226, 0.        , 0.01961086,
       0.09705375, 0.09725826, 0.        , 0.01957868, 0.        ,
       0.03397687, 0.0336847 , 0.07906146, 0.00972389, 0.01976239,
       0.01920425, 0.03929709, 0.03864986, 0.03971773, 0.03895833,
       0.03871694, 0.04952376, 0.19910478, 0.19190622, 0.        ,
       0.02452083, 0.01930237, 0.06767486, 0.58923247, 0.067906  ,
       0.09778057, 0.03932605], [0.05922119, 0.0384095 , 0.077758  , 0.        , 0.01943226,
       0.09697254, 0.09747907, 0.        , 0.01956445, 0.        ,
       0.03348714, 0.03423822, 0.07760215, 0.00977282, 0.01928964,
       0.0195744 , 0.03887293, 0.03884692, 0.03904461, 0.03871522,
       0.03861082, 0.04865565, 0.19514827, 0.1949888 , 0.        ,
       0.02429251, 0.01966084, 0.06888259, 0.5784545 , 0.06788417,
       0.09866702, 0.03908433], [0.05932287, 0.04004917, 0.08038824, 0.        , 0.02006827,
       0.09971288, 0.09926821, 0.        , 0.02020563, 0.        ,
       0.03490455, 0.03542986, 0.08014102, 0.01001077, 0.01981649,
       0.02000913, 0.0402986 , 0.03957786, 0.03991714, 0.03984458,
       0.03981542, 0.04966551, 0.20289892, 0.20042344, 0.        ,
       0.02516625, 0.019855  , 0.06928109, 0.60173648, 0.07033583,
       0.10077129, 0.04004298], [0.05954615, 0.0403517 , 0.08019263, 0.        , 0.01985515,
       0.10010334, 0.10020416, 0.        , 0.02012639, 0.        ,
       0.03486197, 0.0348423 , 0.08061221, 0.00985084, 0.020043  ,
       0.01996004, 0.04009685, 0.04014369, 0.03987084, 0.03972663,
       0.04028551, 0.05038541, 0.1975428 , 0.20108575, 0.        ,
       0.02469113, 0.01986086, 0.07044699, 0.60277504, 0.0696302 ,
       0.09938109, 0.03954054], [0.05970938, 0.04015386, 0.08044518, 0.        , 0.02025372,
       0.09972037, 0.09953617, 0.        , 0.0201833 , 0.        ,
       0.03541328, 0.03515623, 0.08080805, 0.01005313, 0.02022647,
       0.01987173, 0.03980625, 0.039945  , 0.03977992, 0.0401966 ,
       0.04065593, 0.04984231, 0.20300674, 0.19972156, 0.        ,
       0.02482635, 0.01977217, 0.06946197, 0.6026534 , 0.07033119,
       0.09927451, 0.04051654], [0.05942643, 0.03981408, 0.07994341, 0.        , 0.02006137,
       0.10105387, 0.10005339, 0.        , 0.02005472, 0.        ,
       0.0349389 , 0.03512241, 0.08092097, 0.00998688, 0.02005857,
       0.02008746, 0.04016881, 0.03981428, 0.04016724, 0.03958432,
       0.03990761, 0.05006328, 0.19762451, 0.20028189, 0.        ,
       0.02508359, 0.02032308, 0.07047192, 0.59891612, 0.06942247,
       0.10000459, 0.04033046], [0.06050404, 0.03973858, 0.08035182, 0.        , 0.01967366,
       0.10049892, 0.09886042, 0.        , 0.020022  , 0.        ,
       0.03461427, 0.03500181, 0.08108824, 0.00989919, 0.01999092,
       0.02023018, 0.04022479, 0.03960384, 0.04004782, 0.04001614,
       0.03998772, 0.05020859, 0.19917052, 0.20219816, 0.        ,
       0.02531843, 0.020081  , 0.07009592, 0.60006114, 0.06997058,
       0.09874557, 0.04006085], [0.06015688, 0.03973977, 0.08008234, 0.        , 0.02001912,
       0.09953818, 0.10063388, 0.        , 0.02008741, 0.        ,
       0.03502026, 0.03483376, 0.07945582, 0.01007271, 0.02009541,
       0.01992971, 0.03966847, 0.04021055, 0.03987677, 0.04038884,
       0.0400005 , 0.05039882, 0.2001537 , 0.20084781, 0.        ,
       0.02483175, 0.01983509, 0.06983946, 0.59914285, 0.07109003,
       0.09865003, 0.03962977],[0.06129257, 0.04063445, 0.08062862, 0.        , 0.02064048,
       0.10377979, 0.10361427, 0.        , 0.02045151, 0.        ,
       0.03626484, 0.03582612, 0.08205385, 0.01007338, 0.02019002,
       0.0205299 , 0.04127021, 0.04057383, 0.04144166, 0.04034351,
       0.04094483, 0.05137507, 0.20421761, 0.20669963, 0.        ,
       0.02530281, 0.02061974, 0.07081691, 0.62142921, 0.07216849,
       0.10193192, 0.04106851], [0.06127787, 0.04049656, 0.08372382, 0.        , 0.02075541,
       0.1010177 , 0.10364043, 0.        , 0.02029201, 0.        ,
       0.03569632, 0.03603102, 0.08240565, 0.01015648, 0.02050021,
       0.02048594, 0.04150722, 0.04092728, 0.04109701, 0.04142824,
       0.04103577, 0.0514031 , 0.2020953 , 0.20794275, 0.        ,
       0.02533039, 0.0206479 , 0.07185518, 0.6132911 , 0.07089643,
       0.10182675, 0.0405927 ], [0.06184885, 0.04140681, 0.08343176, 0.        , 0.02034844,
       0.10240749, 0.10096413, 0.        , 0.02045531, 0.        ,
       0.03587351, 0.0363387 , 0.08144075, 0.0101002 , 0.02055858,
       0.0203702 , 0.04114036, 0.04103482, 0.04135766, 0.04041855,
       0.04095603, 0.05100706, 0.2038908 , 0.20735993, 0.        ,
       0.02577094, 0.02033901, 0.07136894, 0.61191248, 0.07282139,
       0.10228759, 0.04062555], [0.0606681 , 0.04085454, 0.08211857, 0.        , 0.0202431 ,
       0.10231309, 0.10244837, 0.        , 0.02039715, 0.        ,
       0.03611982, 0.03551424, 0.08181978, 0.01011558, 0.0204036 ,
       0.02038764, 0.04087448, 0.04153609, 0.04098444, 0.04107861,
       0.04063779, 0.05122589, 0.20597445, 0.2073811 , 0.        ,
       0.02534065, 0.02070757, 0.07204153, 0.60858229, 0.07175064,
       0.1029513 , 0.040891  ], [0.06202403, 0.04104986, 0.08202175, 0.        , 0.02018529,
       0.10200362, 0.10294871, 0.        , 0.02036508, 0.        ,
       0.0360358 , 0.03623651, 0.08324361, 0.01023818, 0.0204575 ,
       0.02034847, 0.04143814, 0.0404831 , 0.04115698, 0.04110143,
       0.04147414, 0.05160355, 0.20404861, 0.20621202, 0.        ,
       0.02559516, 0.02044161, 0.07179668, 0.61516806, 0.07219607,
       0.10134469, 0.04118529], [0.06108404, 0.04054964, 0.08222614, 0.        , 0.02048309,
       0.10308116, 0.10398214, 0.        , 0.02046675, 0.        ,
       0.03596122, 0.03561128, 0.08214553, 0.01027508, 0.02049243,
       0.0205605 , 0.04156683, 0.04141243, 0.04142852, 0.04093935,
       0.04074889, 0.05076539, 0.20157736, 0.20339595, 0.        ,
       0.02568173, 0.02068935, 0.07080539, 0.61453961, 0.07230706,
       0.10295984, 0.04102175], [0.05993608, 0.03959755, 0.08136924, 0.        , 0.02001949,
       0.1005662 , 0.09933224, 0.        , 0.01993796, 0.        ,
       0.03516328, 0.03537612, 0.08028947, 0.00991778, 0.01975642,
       0.01999804, 0.03987176, 0.04027319, 0.04019481, 0.03999751,
       0.03967542, 0.04936007, 0.20064654, 0.19776581, 0.        ,
       0.02496051, 0.019923  , 0.0691579 , 0.59104531, 0.07048384,
       0.09880565, 0.03991938], [0.05945284, 0.04054786, 0.08011426, 0.        , 0.02007227,
       0.10036811, 0.10053155, 0.        , 0.02004147, 0.        ,
       0.03508994, 0.0351287 , 0.07970602, 0.01002376, 0.01990216,
       0.01984021, 0.04018248, 0.03983004, 0.04020166, 0.04024716,
       0.04006691, 0.05053679, 0.19885586, 0.20265927, 0.        ,
       0.02512124, 0.0199398 , 0.06994033, 0.60532112, 0.06990492,
       0.09974293, 0.04004127], [0.05982824, 0.03978354, 0.07944441, 0.        , 0.01983554,
       0.10046681, 0.1004959 , 0.        , 0.02004212, 0.        ,
       0.03500496, 0.03457403, 0.08097024, 0.00993595, 0.02011478,
       0.02012646, 0.04024879, 0.04005759, 0.0398701 , 0.04049843,
       0.03997966, 0.04995616, 0.19982893, 0.1978956 , 0.        ,
       0.02494621, 0.02007921, 0.07095917, 0.60122065, 0.07031322,
       0.10151608, 0.0398423 ], [0.05996198, 0.03973713, 0.07940178, 0.        , 0.0201928 ,
       0.09923151, 0.10118979, 0.        , 0.01996817, 0.        ,
       0.03526298, 0.03504022, 0.07976874, 0.00990075, 0.02009172,
       0.02016903, 0.04012766, 0.04012679, 0.03967171, 0.0400801 ,
       0.03991977, 0.04986672, 0.20088735, 0.20042461, 0.        ,
       0.02494518, 0.02019233, 0.06985106, 0.60503943, 0.07042478,
       0.10127737, 0.04002386], [0.05983652, 0.03994862, 0.08087793, 0.        , 0.02005306,
       0.09933526, 0.10058542, 0.        , 0.01988899, 0.        ,
       0.03493458, 0.0346576 , 0.07960629, 0.01005179, 0.01971768,
       0.01986336, 0.04042494, 0.03957146, 0.04048297, 0.03997541,
       0.03963879, 0.04979002, 0.19994922, 0.19842786, 0.        ,
       0.02508255, 0.01997365, 0.06903877, 0.59876813, 0.07019304,
       0.1003445 , 0.03978088], [0.05940792, 0.04017696, 0.08000141, 0.        , 0.01980469,
       0.099384  , 0.09968989, 0.        , 0.02015983, 0.        ,
       0.034866  , 0.03525553, 0.08003485, 0.00999743, 0.01994285,
       0.02013673, 0.04014192, 0.03964832, 0.04029276, 0.03964342,
       0.0403763 , 0.04991701, 0.20062051, 0.20146851, 0.        ,
       0.02521632, 0.01987291, 0.06966016, 0.59162595, 0.06940199,
       0.10014228, 0.0395873 ]])

    _test_agent_1(env, agent, list_feng_11, list_feng_26, list_fu_he, list_fu_he_Q)
elif kk == 3:
    list_feng_11 = np.array([1.400304245098777, 1.5659118536067118, 1.4937667234506091, 1.2676696922146993, 1.4750236146535602, 1.3114384973212627, 1.3888815822951353, 1.6173767960838512, 1.561591140507757, 1.4967745355784543, 1.5859010023912894, 1.448452491060891, 1.5572244274979385, 1.357315002504886, 1.5149692701445394, 1.6089589352656521, 1.5501969745231565, 1.365853857755976, 1.571056540974019, 1.4434901112642073, 1.4815458751464106, 1.347711813674, 1.3161854619570748, 1.4874121965481581])
    list_feng_26 = np.array([0.8377127323669031, 1.0441482739438648, 0.9387188254894712, 0.9733204308397396, 0.9848842836725693, 1.0561329234533887, 0.9817786887688598, 0.9786967526539845, 0.992651446404079, 1.0507069036459467, 0.9567796438251381, 1.0817156382638033, 0.9279922851466478, 0.9530429525976842, 1.11241474156372, 1.0035834597181363, 0.967573457883111, 1.028989079895291, 0.9676740616813577, 0.9253453572080068, 0.9452703366184043, 1.0383506811972798, 0.9995837204425022, 1.022661242280919])
    list_fu_he = np.array([
        [0.06428981, 0.05629709, 0.08121979, 0.        , 0.04208058,
       0.13155808, 0.15874046, 0.        , 0.03886105, 0.        ,
       0.04450565, 0.04045501, 0.07664304, 0.03870599, 0.0385462 ,
       0.04218521, 0.05653758, 0.07093227, 0.06915965, 0.05731599,
       0.05472533, 0.05919361, 0.29358141, 0.29656548, 0.        ,
       0.04603232, 0.04564123, 0.08463806, 0.12440103, 0.10037261,
       0.13364291, 0.04168189],[0.06752212, 0.06304821, 0.08743867, 0.        , 0.0391624 ,
       0.15153006, 0.14759625, 0.        , 0.04084502, 0.        ,
       0.04025578, 0.04003531, 0.07490413, 0.03987063, 0.04176303,
       0.03991626, 0.06603578, 0.06126178, 0.07005722, 0.0623016 ,
       0.05981612, 0.06434644, 0.2808755 , 0.27075085, 0.        ,
       0.03868889, 0.04070462, 0.07828285, 0.12429874, 0.09483121,
       0.16180909, 0.03878435],[0.06591881, 0.06708339, 0.08191104, 0.        , 0.04679683,
       0.1447209 , 0.14571187, 0.        , 0.0450712 , 0.        ,
       0.04372568, 0.04238231, 0.08020835, 0.04080091, 0.04208811,
       0.04482862, 0.06594515, 0.06172918, 0.06327614, 0.06190139,
       0.06496877, 0.06094955, 0.31462587, 0.28776719, 0.        ,
       0.04022679, 0.0417757 , 0.07747186, 0.14135814, 0.11284729,
       0.1458801 , 0.04058136],[0.07205352, 0.06801703, 0.08181784, 0.        , 0.04367585,
       0.14376151, 0.13738933, 0.        , 0.04193018, 0.        ,
       0.03783519, 0.04374473, 0.09230713, 0.04314957, 0.0414143 ,
       0.04162139, 0.06166133, 0.060117  , 0.06500227, 0.06560122,
       0.06449541, 0.0619351 , 0.29573482, 0.3125805 , 0.        ,
       0.04451226, 0.04090325, 0.08817866, 0.13814133, 0.1117533 ,
       0.14851251, 0.04725575],[0.07211066, 0.05918954, 0.08413947, 0.        , 0.03890268,
       0.13221875, 0.13681495, 0.        , 0.04327569, 0.        ,
       0.04568953, 0.04406456, 0.08480333, 0.04410032, 0.04095458,
       0.04211905, 0.0609943 , 0.06279078, 0.06516397, 0.06260829,
       0.06129342, 0.05857033, 0.27746858, 0.31639525, 0.        ,
       0.04441218, 0.04130259, 0.08653713, 0.14924663, 0.11153304,
       0.15444919, 0.04168902],[0.06478962, 0.05893277, 0.08753627, 0.        , 0.03989974,
       0.14140378, 0.13047186, 0.        , 0.04150225, 0.        ,
       0.04096805, 0.04392507, 0.086195  , 0.04057682, 0.0411574 ,
       0.04313822, 0.06190919, 0.06383747, 0.06354616, 0.05810199,
       0.05896578, 0.06365371, 0.27305077, 0.30412342, 0.        ,
       0.0388607 , 0.0454106 , 0.08568737, 0.14634019, 0.11035265,
       0.15503943, 0.03917281],[0.08746203, 0.08463837, 0.10755187, 0.        , 0.05212826,
       0.18856485, 0.17859202, 0.        , 0.05377221, 0.        ,
       0.0542935 , 0.05487632, 0.10323553, 0.05172006, 0.05253158,
       0.05025356, 0.07861535, 0.08429158, 0.07725145, 0.07973771,
       0.08357967, 0.08057115, 0.3604177 , 0.39647617, 0.        ,
       0.05660719, 0.05394201, 0.10094523, 0.18409286, 0.14197558,
       0.19798863, 0.05332188],[0.08929718, 0.0863371 , 0.10762036, 0.        , 0.05405433,
       0.1757946 , 0.1864664 , 0.        , 0.05449767, 0.        ,
       0.05179849, 0.05447004, 0.10391657, 0.05540484, 0.0517533 ,
       0.05407285, 0.08108604, 0.08033516, 0.08390122, 0.07765243,
       0.08005818, 0.08640669, 0.3733266 , 0.36160206, 0.        ,
       0.05231157, 0.05529834, 0.10829109, 0.18749246, 0.1326484 ,
       0.18768163, 0.05304491],[0.0895181 , 0.08006704, 0.1104714 , 0.        , 0.05666571,
       0.18644427, 0.18376808, 0.        , 0.05868905, 0.        ,
       0.05478362, 0.05312795, 0.11112958, 0.05571153, 0.05486107,
       0.05385827, 0.07895123, 0.08225026, 0.0849124 , 0.08172236,
       0.07964427, 0.08007194, 0.3699571 , 0.34976745, 0.        ,
       0.05559103, 0.05536315, 0.11036443, 0.16897744, 0.13447906,
       0.18831281, 0.0527637 ],[0.09102647, 0.08098992, 0.10964378, 0.        , 0.05158366,
       0.17969135, 0.17992195, 0.        , 0.0550299 , 0.        ,
       0.05557035, 0.05541059, 0.11170138, 0.05345649, 0.05679124,
       0.04925952, 0.08119236, 0.08160292, 0.07846504, 0.0835864 ,
       0.08108496, 0.08121792, 0.37867666, 0.3801671 , 0.        ,
       0.0539581 , 0.05313557, 0.10746697, 0.187129  , 0.13272398,
       0.18871229, 0.05272088],[0.09261969, 0.07977524, 0.10909179, 0.        , 0.05483026,
       0.16454312, 0.17032281, 0.        , 0.05303833, 0.        ,
       0.05463321, 0.05749897, 0.11644715, 0.05148776, 0.05280534,
       0.05529684, 0.07645028, 0.08189477, 0.08316025, 0.08517112,
       0.08264661, 0.08173672, 0.39819691, 0.36022755, 0.        ,
       0.050579  , 0.04972664, 0.10776616, 0.18594655, 0.13946252,
       0.18039572, 0.05833143],[0.08636924, 0.08739695, 0.10704965, 0.        , 0.05392158,
       0.17932038, 0.17497297, 0.        , 0.0527713 , 0.        ,
       0.04997685, 0.05584555, 0.10551412, 0.05429471, 0.05311172,
       0.05220174, 0.08142987, 0.08010159, 0.08648422, 0.0837463 ,
       0.0820906 , 0.08153326, 0.35793328, 0.38065258, 0.        ,
       0.05032044, 0.05521957, 0.11081543, 0.17724221, 0.12969485,
       0.19535358, 0.05688376],[0.10882211, 0.09830813, 0.13821128, 0.        , 0.06482797,
       0.22921088, 0.21191436, 0.        , 0.06715402, 0.        ,
       0.07053876, 0.06424189, 0.13224411, 0.06717452, 0.06975298,
       0.0707361 , 0.10539506, 0.09853244, 0.10020581, 0.0942516 ,
       0.09901834, 0.09954896, 0.443011  , 0.45555815, 0.        ,
       0.06493363, 0.06337884, 0.12719001, 0.22203187, 0.15983817,
       0.23713209, 0.06334516],[0.10833518, 0.09333093, 0.13152599, 0.        , 0.06521338,
       0.22209636, 0.22154142, 0.        , 0.06964837, 0.        ,
       0.06311927, 0.06538049, 0.12934236, 0.06849944, 0.06190995,
       0.06894298, 0.1008111 , 0.09848396, 0.09788455, 0.10020019,
       0.09497271, 0.09696206, 0.46678724, 0.46764977, 0.        ,
       0.06670044, 0.06597989, 0.12994138, 0.2156237 , 0.15662972,
       0.22628859, 0.06627983],[0.11078384, 0.10035012, 0.13612055, 0.        , 0.06603418,
       0.21588503, 0.21605431, 0.        , 0.06986178, 0.        ,
       0.06703876, 0.06653979, 0.12992045, 0.06672857, 0.0676671 ,
       0.06626767, 0.10428548, 0.09603265, 0.10090432, 0.10208268,
       0.10095951, 0.09924069, 0.4870749 , 0.45140569, 0.        ,
       0.06607933, 0.06564009, 0.13042551, 0.22481265, 0.16911332,
       0.22431016, 0.06857319],[0.11166616, 0.09971544, 0.13374723, 0.        , 0.06623761,
       0.20967323, 0.22777001, 0.        , 0.0678839 , 0.        ,
       0.06855538, 0.06414423, 0.13465615, 0.06678554, 0.06592503,
       0.06373143, 0.1034333 , 0.10201609, 0.09582311, 0.0996973 ,
       0.09840892, 0.10155168, 0.4639267 , 0.47333763, 0.        ,
       0.06993246, 0.06285108, 0.1299241 , 0.22182653, 0.16447275,
       0.23733274, 0.06678439],[0.10773407, 0.09563722, 0.1415054 , 0.        , 0.0665137 ,
       0.22598215, 0.22427319, 0.        , 0.06807828, 0.        ,
       0.06480115, 0.06325233, 0.13284336, 0.06388186, 0.06254871,
       0.06983688, 0.09846325, 0.09666599, 0.09716729, 0.10301742,
       0.10313776, 0.09636246, 0.4661979 , 0.48260066, 0.        ,
       0.06749609, 0.06597092, 0.13239497, 0.22302841, 0.16692186,
       0.22915331, 0.06666607],[0.10138704, 0.09965226, 0.14108186, 0.        , 0.06847289,
       0.22206394, 0.21182399, 0.        , 0.06575799, 0.        ,
       0.06923478, 0.06626974, 0.12400163, 0.06357168, 0.06190382,
       0.06660155, 0.09874719, 0.10100426, 0.09823984, 0.09617986,
       0.09551833, 0.10247969, 0.46587456, 0.46735932, 0.        ,
       0.06641961, 0.06699295, 0.13441425, 0.21895449, 0.16740708,
       0.23559913, 0.06371182],[0.0890766 , 0.07706374, 0.11401031, 0.        , 0.05629818,
       0.17857975, 0.17536362, 0.        , 0.05325354, 0.        ,
       0.05479536, 0.05793017, 0.10380108, 0.05202173, 0.05295021,
       0.05679141, 0.0832588 , 0.08009398, 0.07839606, 0.07878631,
       0.08452142, 0.08260921, 0.35555671, 0.37985824, 0.        ,
       0.0559056 , 0.05464302, 0.10929871, 0.18008759, 0.13735771,
       0.19539518, 0.05144586],[0.08534271, 0.08584062, 0.1101785 , 0.        , 0.05260857,
       0.16871279, 0.18330722, 0.        , 0.0506609 , 0.        ,
       0.05534511, 0.05593285, 0.10871098, 0.05214377, 0.0543057 ,
       0.05178552, 0.08164859, 0.08401139, 0.0821217 , 0.07542018,
       0.08125242, 0.07710051, 0.38183529, 0.36226604, 0.        ,
       0.05823251, 0.05275616, 0.11555573, 0.19045424, 0.13215858,
       0.18265968, 0.05146207],[0.08761711, 0.07610914, 0.10819002, 0.        , 0.05013975,
       0.18441348, 0.17592928, 0.        , 0.0537464 , 0.        ,
       0.05663508, 0.05243038, 0.10898747, 0.05659502, 0.05377408,
       0.0555923 , 0.0813963 , 0.08301024, 0.07983112, 0.07981253,
       0.08485563, 0.08386023, 0.36304556, 0.35740315, 0.        ,
       0.05103069, 0.05146515, 0.11278138, 0.17760993, 0.13552343,
       0.186826  , 0.05458396],[0.09251165, 0.08137667, 0.11344956, 0.        , 0.05352938,
       0.19651973, 0.19085946, 0.        , 0.05100323, 0.        ,
       0.05549782, 0.05075722, 0.10929549, 0.05836926, 0.05302426,
       0.05237815, 0.0797244 , 0.07891088, 0.08430572, 0.08111073,
       0.08103169, 0.08110713, 0.38396565, 0.36073832, 0.        ,
       0.05361823, 0.05614966, 0.10696514, 0.17847118, 0.13319282,
       0.18204158, 0.05165016],[0.08836983, 0.08656017, 0.1088064 , 0.        , 0.05272826,
       0.18241973, 0.16820088, 0.        , 0.05148518, 0.        ,
       0.05441991, 0.05563535, 0.11033303, 0.05241634, 0.05666401,
       0.05396054, 0.0846091 , 0.08163591, 0.08342545, 0.08459509,
       0.07956389, 0.07719944, 0.36438877, 0.38158637, 0.        ,
       0.05210526, 0.05421022, 0.10798301, 0.17653226, 0.13618497,
       0.1810541 , 0.05421475],[0.08950698, 0.07993096, 0.10387874, 0.        , 0.05260777,
       0.16769805, 0.18055245, 0.        , 0.05765867, 0.        ,
       0.05064651, 0.05190608, 0.10947089, 0.05466435, 0.05704078,
       0.05316289, 0.07531589, 0.07958027, 0.0854474 , 0.08649241,
       0.08334933, 0.08451858, 0.35413086, 0.37973741, 0.        ,
       0.05406131, 0.05514283, 0.10905975, 0.18055193, 0.13889343,
       0.17845076, 0.05057175]])
    list_fu_he_Q = np.array([
        [0.05864316, 0.03923159, 0.07785958, 0.        , 0.01928789,
       0.09853793, 0.09593231, 0.        , 0.01982042, 0.        ,
       0.03381306, 0.03497361, 0.07680895, 0.00996724, 0.01926408,
       0.01912385, 0.03905491, 0.03816217, 0.03959102, 0.0396879 ,
       0.03860264, 0.04963085, 0.19826033, 0.19029197, 0.        ,
       0.02404503, 0.01917676, 0.06948508, 0.5758884 , 0.06767657,
       0.09779435, 0.03888784], [0.05868228, 0.0386371 , 0.07894582, 0.        , 0.01908023,
       0.09899182, 0.09661817, 0.        , 0.01937344, 0.        ,
       0.03489229, 0.03422526, 0.07732837, 0.00985268, 0.01974532,
       0.01950358, 0.03907911, 0.03919309, 0.03829379, 0.03826565,
       0.03865191, 0.04874279, 0.19617934, 0.19813608, 0.        ,
       0.02478814, 0.01960631, 0.06842439, 0.57551785, 0.06800391,
       0.09798085, 0.03857597], [0.0587168 , 0.03899133, 0.07803078, 0.        , 0.01960803,
       0.09702773, 0.09746839, 0.        , 0.01964541, 0.        ,
       0.03368013, 0.03414403, 0.07977126, 0.00962926, 0.01950631,
       0.01978336, 0.03912173, 0.03900959, 0.03943785, 0.03913748,
       0.03886522, 0.04900783, 0.19490676, 0.19773616, 0.        ,
       0.02442017, 0.01943077, 0.06888439, 0.58502551, 0.0683666 ,
       0.09752682, 0.03889455], [0.05891958, 0.03917654, 0.07756935, 0.        , 0.01933865,
       0.09703582, 0.09811739, 0.        , 0.01941784, 0.        ,
       0.03417468, 0.03431874, 0.07851647, 0.00968296, 0.01984703,
       0.01961426, 0.03893298, 0.03890819, 0.03945291, 0.03860861,
       0.03857449, 0.048362  , 0.1963538 , 0.19627055, 0.        ,
       0.0242491 , 0.0195526 , 0.06870644, 0.59378084, 0.06834511,
       0.09685217, 0.03838505], [0.0586531 , 0.03867863, 0.07888064, 0.        , 0.01939257,
       0.09838212, 0.09723537, 0.        , 0.0193245 , 0.        ,
       0.03395263, 0.03402879, 0.07960961, 0.00967151, 0.01959981,
       0.01942715, 0.03833162, 0.03844548, 0.03875533, 0.03983418,
       0.03959857, 0.04863933, 0.19399183, 0.19522129, 0.        ,
       0.02439202, 0.01922316, 0.06752215, 0.59276175, 0.06857128,
       0.09763134, 0.03918636],[0.05903032, 0.03958646, 0.07741352, 0.        , 0.01969399,
       0.0971875 , 0.09808638, 0.        , 0.01966507, 0.        ,
       0.03431109, 0.03377505, 0.07746667, 0.00960754, 0.0193467 ,
       0.01923213, 0.03895848, 0.03903457, 0.03861105, 0.03966537,
       0.0389203 , 0.0492873 , 0.1931834 , 0.19443702, 0.        ,
       0.02461015, 0.01949564, 0.06828132, 0.58973419, 0.06798057,
       0.09837706, 0.03883742],[0.0601777 , 0.04006886, 0.07977288, 0.        , 0.02008342,
       0.09906384, 0.10015271, 0.        , 0.02001182, 0.        ,
       0.03480033, 0.03488689, 0.0803081 , 0.01008928, 0.01989643,
       0.02015029, 0.03981842, 0.04003115, 0.03965853, 0.04013733,
       0.03990999, 0.0493283 , 0.19834628, 0.20214736, 0.        ,
       0.024954  , 0.02004551, 0.07054936, 0.59835187, 0.07106658,
       0.10106496, 0.04018552],[0.06019112, 0.0402145 , 0.08058739, 0.        , 0.01999107,
       0.0998693 , 0.10044441, 0.        , 0.01989824, 0.        ,
       0.03498244, 0.03495167, 0.08028687, 0.00990248, 0.02017544,
       0.01998726, 0.04001718, 0.03985356, 0.04017967, 0.040595  ,
       0.04008929, 0.04945309, 0.19926721, 0.19664767, 0.        ,
       0.02508237, 0.02015574, 0.07047186, 0.60132194, 0.06926143,
       0.10060849, 0.04006461],[0.05955681, 0.0396567 , 0.08033444, 0.        , 0.01990038,
       0.09946856, 0.10046653, 0.        , 0.02008409, 0.        ,
       0.03496546, 0.03495871, 0.08048045, 0.01001275, 0.01993974,
       0.02012365, 0.03990906, 0.0396892 , 0.039792  , 0.03999729,
       0.03984981, 0.05008241, 0.19789738, 0.19811915, 0.        ,
       0.02512848, 0.01992072, 0.0705862 , 0.60104498, 0.06960163,
       0.10033857, 0.04009464],[0.06003722, 0.0398102 , 0.0795696 , 0.        , 0.02010707,
       0.09959658, 0.09978403, 0.        , 0.02016795, 0.        ,
       0.03512694, 0.03527948, 0.0802452 , 0.00989323, 0.02018423,
       0.02014277, 0.03967937, 0.04053348, 0.03970535, 0.03983982,
       0.04035849, 0.05031708, 0.19896844, 0.1992002 , 0.        ,
       0.02480818, 0.02005725, 0.07083963, 0.60544554, 0.06978784,
       0.10058095, 0.03986058],[0.06014397, 0.03994928, 0.07940731, 0.        , 0.01998499,
       0.09988365, 0.09945083, 0.        , 0.02002896, 0.        ,
       0.03504963, 0.03521477, 0.08133358, 0.00993349, 0.02017002,
       0.01999232, 0.03990559, 0.03996943, 0.0397763 , 0.03995365,
       0.03995489, 0.04993001, 0.20089222, 0.19738714, 0.        ,
       0.02490069, 0.01990568, 0.06947686, 0.60440389, 0.07012876,
       0.0988004 , 0.04032529],[0.06029896, 0.03978928, 0.07978057, 0.        , 0.02007816,
       0.10101061, 0.09897259, 0.        , 0.02022998, 0.        ,
       0.03494737, 0.03443009, 0.07872703, 0.01001348, 0.0198689 ,
       0.01995787, 0.04040811, 0.03945851, 0.03985556, 0.0400276 ,
       0.03974174, 0.04951055, 0.20320318, 0.19872728, 0.        ,
       0.02510218, 0.01995907, 0.07046018, 0.60039629, 0.069948  ,
       0.10043289, 0.03991082],[0.06149789, 0.04099902, 0.0826288 , 0.        , 0.0203358 ,
       0.10264717, 0.10194663, 0.        , 0.02062412, 0.        ,
       0.03567045, 0.03581405, 0.08137385, 0.01025134, 0.02066144,
       0.02037674, 0.04106207, 0.04067956, 0.04082691, 0.0410622 ,
       0.04069688, 0.05107568, 0.20742652, 0.20582671, 0.        ,
       0.02581418, 0.02071086, 0.0718161 , 0.61860002, 0.0726258 ,
       0.10258896, 0.04099106],[0.06127214, 0.04096968, 0.08182175, 0.        , 0.02027721,
       0.10117901, 0.10229662, 0.        , 0.0205928 , 0.        ,
       0.0361258 , 0.0360258 , 0.08373449, 0.01035931, 0.02064326,
       0.02057179, 0.04119922, 0.04042411, 0.04088499, 0.04080382,
       0.0406252 , 0.05090272, 0.20316342, 0.20428008, 0.        ,
       0.02572462, 0.02032555, 0.07165391, 0.61430599, 0.07108909,
       0.10384832, 0.04075153],[0.06171998, 0.04117865, 0.08194929, 0.        , 0.020555  ,
       0.10318969, 0.10174871, 0.        , 0.0203349 , 0.        ,
       0.03604521, 0.03592528, 0.08157717, 0.01037785, 0.02043815,
       0.02053098, 0.04094708, 0.0412073 , 0.04050505, 0.04069697,
       0.04099241, 0.05141711, 0.2027259 , 0.20257088, 0.        ,
       0.02560438, 0.02023407, 0.07191578, 0.62088702, 0.07076197,
       0.10360203, 0.04163497],[0.06068872, 0.04052129, 0.08110251, 0.        , 0.02050594,
       0.1031908 , 0.10274191, 0.        , 0.02045324, 0.        ,
       0.03568039, 0.03577782, 0.08185374, 0.01016471, 0.02027519,
       0.02050257, 0.04068679, 0.04120223, 0.04113191, 0.04066139,
       0.0414877 , 0.05158982, 0.20284846, 0.2050607 , 0.        ,
       0.02585703, 0.02049007, 0.07204282, 0.61299158, 0.07183483,
       0.10326399, 0.04114798],[0.06119761, 0.04014861, 0.08192292, 0.        , 0.02061912,
       0.10298505, 0.10249188, 0.        , 0.02057623, 0.        ,
       0.03607205, 0.03632799, 0.08103556, 0.01038797, 0.02042087,
       0.02061672, 0.04110681, 0.04113716, 0.04170115, 0.04110915,
       0.04036077, 0.05136675, 0.20296551, 0.20618594, 0.        ,
       0.0260242 , 0.02040593, 0.07250228, 0.62349576, 0.07220958,
       0.10155094, 0.0412607 ],[0.06134804, 0.04101969, 0.08254506, 0.        , 0.02039359,
       0.1028709 , 0.10200761, 0.        , 0.0207068 , 0.        ,
       0.03575486, 0.03576607, 0.08112291, 0.01013464, 0.0204035 ,
       0.02064373, 0.0407052 , 0.04064573, 0.04109485, 0.0409419 ,
       0.04097026, 0.05126498, 0.20378766, 0.20459858, 0.        ,
       0.02551936, 0.02048997, 0.07174277, 0.61752683, 0.07241334,
       0.10288626, 0.04131094],[0.05945844, 0.04009776, 0.0802605 , 0.        , 0.01983002,
       0.09934731, 0.10009135, 0.        , 0.02011503, 0.        ,
       0.03494631, 0.03508639, 0.08065599, 0.00994562, 0.02000589,
       0.01991385, 0.0399693 , 0.03955553, 0.03969515, 0.03989792,
       0.04054137, 0.05014372, 0.20189419, 0.19992106, 0.        ,
       0.02493423, 0.02006904, 0.07002006, 0.59337517, 0.0689261 ,
       0.10134298, 0.03959995],[0.06087178, 0.04006153, 0.08035457, 0.        , 0.02023468,
       0.09955651, 0.09959085, 0.        , 0.02009821, 0.        ,
       0.0350341 , 0.03516408, 0.07955702, 0.01011599, 0.02000977,
       0.01979031, 0.04009401, 0.03996195, 0.03998523, 0.03993458,
       0.03965153, 0.05014927, 0.20023444, 0.20116189, 0.        ,
       0.02515927, 0.01988646, 0.07057052, 0.59984512, 0.0699726 ,
       0.09987284, 0.03983679],[0.0599369 , 0.03968249, 0.08024397, 0.        , 0.01998058,
       0.09965863, 0.09888953, 0.        , 0.02015715, 0.        ,
       0.03477843, 0.03507388, 0.08062245, 0.00990005, 0.01979599,
       0.02017466, 0.0397887 , 0.04007523, 0.03990653, 0.03998737,
       0.0399436 , 0.04930171, 0.19809403, 0.20070855, 0.        ,
       0.02497727, 0.01972056, 0.07006097, 0.5969596 , 0.06966638,
       0.09962396, 0.03967496],[0.05983862, 0.04001811, 0.08006148, 0.        , 0.01995177,
       0.09923888, 0.10135298, 0.        , 0.01986108, 0.        ,
       0.03494811, 0.03484384, 0.07981175, 0.01013923, 0.01985633,
       0.02005301, 0.0399082 , 0.03996609, 0.04015443, 0.03980713,
       0.03990108, 0.04983569, 0.19939207, 0.19773919, 0.        ,
       0.02486479, 0.01999998, 0.07037123, 0.6033357 , 0.06973303,
       0.09940429, 0.04041678],[0.0598592 , 0.0398303 , 0.07983005, 0.        , 0.01984303,
       0.10091437, 0.09894427, 0.        , 0.01995034, 0.        ,
       0.03521308, 0.03489197, 0.08087032, 0.00999827, 0.02009861,
       0.0198884 , 0.0400662 , 0.04013684, 0.04036729, 0.03970545,
       0.04003909, 0.04991518, 0.20148247, 0.2006586 , 0.        ,
       0.02509116, 0.01990913, 0.06930181, 0.60079772, 0.07042917,
       0.10004937, 0.04017843],[0.06020451, 0.04007698, 0.08013665, 0.        , 0.0199159 ,
       0.10016059, 0.09987606, 0.        , 0.02014511, 0.        ,
       0.03488844, 0.03495354, 0.08009836, 0.01001688, 0.01997755,
       0.02015647, 0.0401228 , 0.03973378, 0.04030504, 0.04004652,
       0.03978658, 0.05010506, 0.19835644, 0.20159277, 0.        ,
       0.02459017, 0.02010828, 0.06883435, 0.60482651, 0.07028587,
       0.09981627, 0.03959531]])

    _test_agent_1(env, agent, list_feng_11, list_feng_26, list_fu_he, list_fu_he_Q)
elif kk == 4:
    list_feng_11 = np.array([1.5038232309028068, 1.441084310344463, 1.4808999793664335, 1.303966010051795, 1.3425386058689366, 1.598736306938759, 1.6782030842517783, 1.6299418300466444, 1.5325548612773603, 1.4181003769258282, 1.4090216721231217, 1.340100739967923, 1.6414104028627727, 1.6419497014287188, 1.5112416774809656, 1.6366455235071384, 1.496355243023677, 1.4764888252839332, 1.3999987873276718, 1.4916062263666514, 1.5148552719847808, 1.6888834834073396, 1.3778829811697246, 1.448928309948203])
    list_feng_26 = np.array([1.1797893234578383, 1.0090299914466736, 1.0624694015069889, 0.898091906613436, 0.9366565229667609, 0.9666871893570754, 1.0268886526932592, 1.0046775250087847, 1.083658684930386, 0.9984952698492011, 0.9806731833362154, 1.0644453858904392, 1.0663069388015338, 0.9708806097790149, 0.9293397277022353, 0.9516994502792642, 1.047976819858602, 1.060304076125751, 0.9592849082118735, 1.0730469736935007, 0.9266941274613509, 0.9967483176388885, 1.0960781579873071, 1.0391122294938049])
    list_fu_he = np.array([
       [0.07862953, 0.06796085, 0.07621811, 0.        , 0.04253161,
       0.13197055, 0.14319019, 0.        , 0.04212938, 0.        ,
       0.04239458, 0.03732294, 0.07968753, 0.03810424, 0.03631838,
       0.0385246 , 0.06722058, 0.06168244, 0.07177038, 0.0638923 ,
       0.0574999 , 0.07157201, 0.27011734, 0.27373878, 0.        ,
       0.03719453, 0.04005319, 0.09450951, 0.12374977, 0.11764641,
       0.13000892, 0.0376754 ], [0.06816889, 0.07142063, 0.09312651, 0.        , 0.04148405,
       0.15120773, 0.13972094, 0.        , 0.04317897, 0.        ,
       0.04131971, 0.03893315, 0.07875787, 0.04436559, 0.03754991,
       0.04220352, 0.06365803, 0.06811116, 0.06751838, 0.06636347,
       0.06838912, 0.06253465, 0.28556711, 0.31671413, 0.        ,
       0.0456375 , 0.0438085 , 0.08600725, 0.14562467, 0.11333346,
       0.15051433, 0.04307677], [0.07239435, 0.06275879, 0.08288055, 0.        , 0.04216113,
       0.14791715, 0.13521142, 0.        , 0.03806014, 0.        ,
       0.04350727, 0.04469044, 0.08826346, 0.04382645, 0.04158876,
       0.04135513, 0.05870302, 0.0637914 , 0.06607003, 0.06462701,
       0.06669508, 0.06841903, 0.27861531, 0.27480754, 0.        ,
       0.04495554, 0.04483194, 0.08311432, 0.14525881, 0.10071448,
       0.14757242, 0.04112456], [0.06610106, 0.06621457, 0.08099628, 0.        , 0.04080085,
       0.13175602, 0.14416842, 0.        , 0.04476574, 0.        ,
       0.04398599, 0.0419658 , 0.09148888, 0.0414757 , 0.04155084,
       0.04552802, 0.06616729, 0.05893193, 0.06849088, 0.06706553,
       0.06400654, 0.0621148 , 0.28888607, 0.28309746, 0.        ,
       0.04330064, 0.04309095, 0.09399186, 0.13151422, 0.10461916,
       0.15188982, 0.04377366], [0.06631807, 0.05811266, 0.07934894, 0.        , 0.04390307,
       0.15226014, 0.13021354, 0.        , 0.04204243, 0.        ,
       0.03854834, 0.04293566, 0.08561606, 0.04375882, 0.04191412,
       0.04025774, 0.06449452, 0.061046  , 0.0583853 , 0.06390218,
       0.06466667, 0.06555815, 0.27514848, 0.31160986, 0.        ,
       0.04428741, 0.04029968, 0.08804797, 0.14976404, 0.09740437,
       0.13851154, 0.04214413], [0.07179627, 0.06807061, 0.07979759, 0.        , 0.04560553,
       0.14750097, 0.14420165, 0.        , 0.04262857, 0.        ,
       0.04228332, 0.04120405, 0.08204868, 0.04301837, 0.03915454,
       0.042419  , 0.06246606, 0.06649261, 0.06505015, 0.0658248 ,
       0.0647089 , 0.05873562, 0.30998669, 0.28143192, 0.        ,
       0.04225943, 0.04509136, 0.08881876, 0.13769882, 0.11191451,
       0.15118173, 0.04148585], [0.08914762, 0.07881831, 0.09792797, 0.        , 0.05268152,
       0.17134079, 0.1667017 , 0.        , 0.05618626, 0.        ,
       0.0548261 , 0.05576523, 0.11019618, 0.05495119, 0.05436883,
       0.0563177 , 0.08033719, 0.08340932, 0.07947516, 0.07964707,
       0.0843715 , 0.07884002, 0.36507647, 0.36559088, 0.        ,
       0.05192356, 0.05507893, 0.10928995, 0.19357061, 0.13592532,
       0.19034972, 0.05519401], [0.09361638, 0.08121884, 0.10915365, 0.        , 0.05503574,
       0.17889142, 0.18725815, 0.        , 0.05370658, 0.        ,
       0.0555922 , 0.05076334, 0.11088746, 0.05645388, 0.05515868,
       0.05305767, 0.08456967, 0.07946897, 0.08035609, 0.07878667,
       0.07623616, 0.07662448, 0.37748605, 0.39152676, 0.        ,
       0.05494881, 0.0512798 , 0.10858752, 0.18174557, 0.12851089,
       0.18607654, 0.05580232], [0.0906846 , 0.0847187 , 0.11211167, 0.        , 0.05066735,
       0.18977668, 0.18336825, 0.        , 0.0544349 , 0.        ,
       0.05465483, 0.05403215, 0.11357126, 0.05556805, 0.05767674,
       0.05249923, 0.08081998, 0.08409656, 0.08490767, 0.07589398,
       0.08613049, 0.07804547, 0.41310942, 0.3772511 , 0.        ,
       0.05565041, 0.05632404, 0.10883445, 0.18202613, 0.13094646,
       0.19123864, 0.05044519], [0.08964508, 0.07718761, 0.10668108, 0.        , 0.05263678,
       0.1724426 , 0.18713531, 0.        , 0.05417027, 0.        ,
       0.05089861, 0.05396691, 0.10943898, 0.05536489, 0.05443161,
       0.05866475, 0.08001642, 0.07863001, 0.080755  , 0.07740914,
       0.08197937, 0.07753387, 0.39750407, 0.35046307, 0.        ,
       0.05177237, 0.0516173 , 0.10230555, 0.1790957 , 0.14041185,
       0.1846708 , 0.05454851], [0.09227437, 0.08591235, 0.10839473, 0.        , 0.0537981 ,
       0.18249227, 0.17051566, 0.        , 0.05299164, 0.        ,
       0.05327337, 0.05460138, 0.11601972, 0.05825682, 0.05413975,
       0.05641888, 0.08212139, 0.08271028, 0.07530919, 0.07557525,
       0.07820217, 0.07619469, 0.37599001, 0.39426094, 0.        ,
       0.05409773, 0.04935318, 0.10602175, 0.18164803, 0.14056648,
       0.19292637, 0.05573571], [0.09481125, 0.08325723, 0.11497417, 0.        , 0.05560221,
       0.18337608, 0.17241271, 0.        , 0.05448342, 0.        ,
       0.05164125, 0.05179032, 0.11002592, 0.05426994, 0.05632471,
       0.05572466, 0.08524245, 0.07988604, 0.08353156, 0.0813717 ,
       0.0743746 , 0.08096261, 0.3709006 , 0.388923  , 0.        ,
       0.05052922, 0.05095639, 0.10820044, 0.17346698, 0.14619641,
       0.19115117, 0.05155794], [0.10220479, 0.09723521, 0.12883643, 0.        , 0.06307611,
       0.22474716, 0.22669847, 0.        , 0.06432241, 0.        ,
       0.06655658, 0.06713681, 0.13292774, 0.06566038, 0.06459463,
       0.06987082, 0.09505477, 0.10130541, 0.09872339, 0.10354738,
       0.10250027, 0.09570749, 0.4648264 , 0.47311511, 0.        ,
       0.06878949, 0.06754626, 0.12746607, 0.21969629, 0.15772571,
       0.23091904, 0.06825171], [0.1103723 , 0.10416203, 0.12970126, 0.        , 0.06467094,
       0.21155002, 0.21434376, 0.        , 0.06574663, 0.        ,
       0.06543859, 0.06506841, 0.12913002, 0.06560879, 0.06737947,
       0.07020365, 0.10403984, 0.10365732, 0.09859674, 0.10193953,
       0.09493577, 0.09878167, 0.43218619, 0.47724533, 0.        ,
       0.06520063, 0.06385246, 0.12940078, 0.22525598, 0.16958557,
       0.2287204 , 0.06687407], [0.10663045, 0.10180105, 0.13191421, 0.        , 0.06595557,
       0.2310262 , 0.22319837, 0.        , 0.06611233, 0.        ,
       0.06794792, 0.06381546, 0.13713358, 0.0664859 , 0.06388887,
       0.0648794 , 0.09953992, 0.09772217, 0.09932937, 0.09280986,
       0.10100451, 0.09951774, 0.44493246, 0.45662811, 0.        ,
       0.06830078, 0.06840791, 0.12638548, 0.22190485, 0.16504441,
       0.22588202, 0.06499206], [0.10427217, 0.09856206, 0.13473529, 0.        , 0.06592651,
       0.22596977, 0.21626314, 0.        , 0.06545406, 0.        ,
       0.06875136, 0.06758966, 0.13421437, 0.06303975, 0.06774322,
       0.06506965, 0.1028532 , 0.09268525, 0.10086581, 0.09908434,
       0.09883991, 0.09877708, 0.46329374, 0.43232794, 0.        ,
       0.06570573, 0.06441544, 0.12587069, 0.22622071, 0.16299042,
       0.23224315, 0.06526472], [0.10678924, 0.10361993, 0.1345187 , 0.        , 0.06800687,
       0.217969  , 0.21508777, 0.        , 0.06630877, 0.        ,
       0.0690624 , 0.06538405, 0.12327424, 0.06322368, 0.06445627,
       0.06888119, 0.10444127, 0.09767634, 0.10648188, 0.09623134,
       0.09971183, 0.10361661, 0.44408595, 0.44599659, 0.        ,
       0.06189524, 0.06629356, 0.13538394, 0.2190097 , 0.15966251,
       0.22979069, 0.0636126 ], [0.11448917, 0.09809028, 0.13332037, 0.        , 0.06658917,
       0.22123722, 0.21902058, 0.        , 0.06513815, 0.        ,
       0.06342942, 0.06443289, 0.13746594, 0.06705315, 0.06602199,
       0.06889226, 0.10166803, 0.1026933 , 0.09991782, 0.09979147,
       0.10136089, 0.09759861, 0.46806066, 0.45062009, 0.        ,
       0.06264889, 0.0683844 , 0.13470084, 0.22492037, 0.1635714 ,
       0.22784219, 0.06552374], [0.08986165, 0.07956208, 0.11514668, 0.        , 0.05400362,
       0.18800192, 0.17421666, 0.        , 0.05495865, 0.        ,
       0.05372355, 0.05208665, 0.11086976, 0.05584989, 0.05498754,
       0.05817414, 0.07924089, 0.08080009, 0.08336288, 0.08202161,
       0.07461504, 0.07993112, 0.41077366, 0.38695221, 0.        ,
       0.05577544, 0.05438443, 0.11191421, 0.17536206, 0.13406403,
       0.18411369, 0.05531789], [0.08583436, 0.07783481, 0.10632684, 0.        , 0.05180643,
       0.1900523 , 0.18136515, 0.        , 0.05501893, 0.        ,
       0.052059  , 0.05608829, 0.10407232, 0.0569923 , 0.0566169 ,
       0.0544975 , 0.08226075, 0.08404254, 0.07945902, 0.0801956 ,
       0.08058571, 0.08379108, 0.3897242 , 0.38218237, 0.        ,
       0.05240689, 0.05776237, 0.10859083, 0.17463506, 0.13236273,
       0.18940354, 0.05155787], [0.08986701, 0.0847974 , 0.10811369, 0.        , 0.05577601,
       0.18142074, 0.17926292, 0.        , 0.05689821, 0.        ,
       0.05376167, 0.05270622, 0.1083329 , 0.05200621, 0.05143567,
       0.05546973, 0.08015495, 0.0875754 , 0.08123009, 0.07985275,
       0.0798659 , 0.08221208, 0.38203014, 0.39654046, 0.        ,
       0.05241328, 0.05645008, 0.10792113, 0.17177941, 0.1332175 ,
       0.17418331, 0.05691044], [0.08542462, 0.0780909 , 0.10709812, 0.        , 0.05158124,
       0.18137694, 0.18578834, 0.        , 0.05272491, 0.        ,
       0.0561643 , 0.05379233, 0.11369089, 0.05634979, 0.05273997,
       0.0544055 , 0.0765638 , 0.07966885, 0.08016589, 0.07967255,
       0.08061877, 0.08417697, 0.35830332, 0.36628538, 0.        ,
       0.05381296, 0.05234108, 0.10963637, 0.18209849, 0.13775971,
       0.20023951, 0.04835702], [0.08801999, 0.0841273 , 0.10564042, 0.        , 0.05563498,
       0.17247951, 0.18054905, 0.        , 0.05394066, 0.        ,
       0.04996706, 0.05095431, 0.10860121, 0.05232639, 0.05424441,
       0.05403791, 0.07994868, 0.08188514, 0.07813214, 0.08100799,
       0.07697528, 0.07976555, 0.38240169, 0.38444466, 0.        ,
       0.04923029, 0.05224333, 0.10097602, 0.17840848, 0.1389188 ,
       0.19637729, 0.05160376], [0.0927247 , 0.08099973, 0.10747738, 0.        , 0.0536074 ,
       0.17489882, 0.18237757, 0.        , 0.05167295, 0.        ,
       0.05571436, 0.05589167, 0.10946132, 0.05250073, 0.05497778,
       0.05533697, 0.08260798, 0.08609011, 0.07956198, 0.08219044,
       0.07807164, 0.07928406, 0.38000254, 0.36541198, 0.        ,
       0.05420197, 0.05188574, 0.10656414, 0.18570669, 0.13497915,
       0.18782324, 0.05538409]])
    list_fu_he_Q = np.array([
        [0.0590544 , 0.0390361 , 0.07656901, 0.        , 0.01974277,
       0.09580461, 0.09803863, 0.        , 0.01993807, 0.        ,
       0.03363483, 0.03456886, 0.07710241, 0.00961796, 0.01950737,
       0.01924117, 0.03957365, 0.03930203, 0.0398363 , 0.03963377,
       0.03988544, 0.04922309, 0.19928777, 0.19610784, 0.        ,
       0.02419788, 0.01951916, 0.06700814, 0.57648488, 0.06716339,
       0.09722702, 0.03927994], [0.05947167, 0.03889539, 0.07742336, 0.        , 0.01938826,
       0.09743812, 0.09809014, 0.        , 0.01929525, 0.        ,
       0.03376035, 0.03441879, 0.07655621, 0.00985805, 0.01984907,
       0.01941753, 0.03902349, 0.03883279, 0.03867064, 0.03924845,
       0.03937922, 0.04955046, 0.19283682, 0.1995265 , 0.        ,
       0.02448321, 0.01943656, 0.06734069, 0.5762725 , 0.06893108,
       0.09679303, 0.03947985], [0.0582168 , 0.03925084, 0.07796557, 0.        , 0.019388  ,
       0.09716736, 0.09680629, 0.        , 0.01942432, 0.        ,
       0.03408897, 0.03413389, 0.07738954, 0.00984357, 0.01960989,
       0.01952307, 0.03944437, 0.03886442, 0.03866151, 0.03888986,
       0.03887202, 0.04939915, 0.19695528, 0.19779283, 0.        ,
       0.02449943, 0.01958144, 0.06885489, 0.58667278, 0.06790308,
       0.09816865, 0.03880678], [0.05871737, 0.03895714, 0.07837905, 0.        , 0.01970948,
       0.09695185, 0.09821645, 0.        , 0.01949903, 0.        ,
       0.03408065, 0.03425394, 0.07862527, 0.00975166, 0.01961767,
       0.0196884 , 0.03909789, 0.03904718, 0.03921658, 0.03889545,
       0.03911649, 0.04880021, 0.19390112, 0.19482762, 0.        ,
       0.02446597, 0.01940029, 0.06841681, 0.58915525, 0.06734595,
       0.09650013, 0.0387812 ], [0.05821038, 0.03834447, 0.07669456, 0.        , 0.01958137,
       0.09703715, 0.09672327, 0.        , 0.0194316 , 0.        ,
       0.0337631 , 0.03448119, 0.07782898, 0.00981269, 0.01927321,
       0.01973434, 0.03872245, 0.03875673, 0.03879931, 0.03906746,
       0.03877464, 0.04871952, 0.19581661, 0.19491595, 0.        ,
       0.02455071, 0.0193373 , 0.06862344, 0.59208607, 0.06888562,
       0.09895112, 0.03912533], [0.0579592 , 0.03923962, 0.07852522, 0.        , 0.01926907,
       0.09741555, 0.09661999, 0.        , 0.01940946, 0.        ,
       0.03378235, 0.03405552, 0.07705653, 0.00976558, 0.01964625,
       0.01951084, 0.03925336, 0.03919251, 0.03889485, 0.03952563,
       0.03876746, 0.04871125, 0.19395845, 0.19524844, 0.        ,
       0.02402069, 0.01967231, 0.06811617, 0.58621582, 0.0676003 ,
       0.09770683, 0.0386079 ], [0.06055712, 0.04003032, 0.08103973, 0.        , 0.01996111,
       0.10028603, 0.09989409, 0.        , 0.02005698, 0.        ,
       0.03506212, 0.03457714, 0.0801689 , 0.00995178, 0.01989863,
       0.01973613, 0.03953165, 0.03956622, 0.04008682, 0.03995526,
       0.0401014 , 0.05032512, 0.19836427, 0.19942762, 0.        ,
       0.02519665, 0.01998083, 0.07018805, 0.60371946, 0.0703176 ,
       0.09876578, 0.040133  ], [0.06003764, 0.03996167, 0.08025383, 0.        , 0.02001324,
       0.09985041, 0.0994674 , 0.        , 0.02020057, 0.        ,
       0.03549492, 0.0352229 , 0.07936887, 0.01001674, 0.02001364,
       0.01980372, 0.04023612, 0.03976115, 0.04040339, 0.03969793,
       0.04020421, 0.04976162, 0.1988254 , 0.19783851, 0.        ,
       0.02527527, 0.02000424, 0.0702563 , 0.59341395, 0.06992556,
       0.10046079, 0.03991537], [0.06041109, 0.04015151, 0.07963094, 0.        , 0.01976474,
       0.09948908, 0.09986741, 0.        , 0.02013133, 0.        ,
       0.03505509, 0.03495376, 0.07956269, 0.00997908, 0.01990924,
       0.02029219, 0.04028229, 0.03973098, 0.03958421, 0.04026864,
       0.04011526, 0.04956359, 0.20038108, 0.20056276, 0.        ,
       0.02536522, 0.019949  , 0.06988964, 0.60541555, 0.06932247,
       0.0998099 , 0.03989345], [0.05977736, 0.03957978, 0.08072346, 0.        , 0.02021995,
       0.10064871, 0.09865237, 0.        , 0.02002368, 0.        ,
       0.03494406, 0.03502426, 0.08054229, 0.0099971 , 0.01990323,
       0.02013592, 0.0402779 , 0.04005336, 0.04021428, 0.03988268,
       0.03977528, 0.05006161, 0.20096734, 0.19960337, 0.        ,
       0.02483511, 0.01984781, 0.06941512, 0.60183797, 0.06996756,
       0.10018628, 0.03977269], [0.05925773, 0.03961215, 0.08033615, 0.        , 0.01984692,
       0.10062343, 0.10117334, 0.        , 0.01978995, 0.        ,
       0.03528091, 0.03494343, 0.08065133, 0.00994236, 0.01998384,
       0.02028907, 0.0401525 , 0.03969891, 0.04014656, 0.0396516 ,
       0.03965494, 0.04916481, 0.19941198, 0.20198754, 0.        ,
       0.02499256, 0.02007367, 0.06905813, 0.59594129, 0.07005211,
       0.10022683, 0.03984321], [0.0604632 , 0.04016082, 0.08012662, 0.        , 0.02005251,
       0.10059236, 0.09947682, 0.        , 0.01998546, 0.        ,
       0.03527555, 0.03488287, 0.08002764, 0.00995172, 0.01984871,
       0.01983993, 0.0401386 , 0.04046324, 0.04024209, 0.04008098,
       0.0397693 , 0.04979293, 0.19858442, 0.19940417, 0.        ,
       0.02498469, 0.01998399, 0.06973495, 0.60354554, 0.06985912,
       0.10050918, 0.04023761], [0.06153493, 0.04123097, 0.08276372, 0.        , 0.02053354,
       0.10244742, 0.10279958, 0.        , 0.0207773 , 0.        ,
       0.03563349, 0.03642444, 0.08156826, 0.0102473 , 0.020292  ,
       0.02063945, 0.04045643, 0.04103991, 0.04132081, 0.04134314,
       0.04065285, 0.05184559, 0.20410465, 0.2058665 , 0.        ,
       0.02588953, 0.02059991, 0.07154076, 0.62058672, 0.07229712,
       0.10310237, 0.04044244], [0.06145259, 0.04123795, 0.08097206, 0.        , 0.02084995,
       0.10257937, 0.1031064 , 0.        , 0.02046676, 0.        ,
       0.03577201, 0.0358953 , 0.08227747, 0.01015796, 0.02076043,
       0.02013977, 0.0415351 , 0.04147807, 0.04095644, 0.04136332,
       0.04159333, 0.05180515, 0.20640776, 0.20595362, 0.        ,
       0.02547349, 0.02044891, 0.07141348, 0.61997994, 0.07154763,
       0.10239505, 0.04110147], [0.06060299, 0.04058701, 0.08164748, 0.        , 0.02020907,
       0.10265684, 0.10269685, 0.        , 0.02062569, 0.        ,
       0.03600571, 0.0355312 , 0.08155999, 0.01030015, 0.02063614,
       0.02078122, 0.04136279, 0.04113781, 0.04100122, 0.040397  ,
       0.04047   , 0.05128543, 0.20584756, 0.20460254, 0.        ,
       0.02595475, 0.02060721, 0.07272038, 0.61102343, 0.07191642,
       0.101206  , 0.04068876], [0.06124645, 0.04115534, 0.08091539, 0.        , 0.02036212,
       0.10266176, 0.10287208, 0.        , 0.02070888, 0.        ,
       0.03635636, 0.03589945, 0.08212524, 0.01037822, 0.02051445,
       0.0204051 , 0.0406388 , 0.04177939, 0.04064235, 0.04123724,
       0.04089889, 0.05084689, 0.20655468, 0.20436307, 0.        ,
       0.02520375, 0.02068841, 0.07218297, 0.61539815, 0.07123628,
       0.1016937 , 0.04145926], [0.06156106, 0.04120299, 0.08280771, 0.        , 0.02013817,
       0.10191803, 0.10178519, 0.        , 0.02051052, 0.        ,
       0.03620835, 0.03612911, 0.08123617, 0.01041951, 0.02036026,
       0.02045659, 0.04061776, 0.04053689, 0.04062707, 0.04092797,
       0.04142045, 0.0515156 , 0.2046226 , 0.20262333, 0.        ,
       0.02584766, 0.0206141 , 0.07169331, 0.61250723, 0.07277455,
       0.10277089, 0.0410589 ], [0.06141484, 0.040598  , 0.08163081, 0.        , 0.02041011,
       0.10228743, 0.10143655, 0.        , 0.02066152, 0.        ,
       0.03567215, 0.03536639, 0.08239023, 0.01033512, 0.020649  ,
       0.02046628, 0.0413701 , 0.04128084, 0.04164645, 0.04122151,
       0.04162778, 0.05060378, 0.20443431, 0.20305042, 0.        ,
       0.02563171, 0.02047312, 0.07239016, 0.61731873, 0.0721484 ,
       0.1024384 , 0.04094321], [0.05963428, 0.03993473, 0.07956869, 0.        , 0.01975987,
       0.09976075, 0.09863087, 0.        , 0.01986775, 0.        ,
       0.03514189, 0.03529295, 0.0792133 , 0.01005926, 0.02016645,
       0.02011926, 0.04011666, 0.04012385, 0.04032469, 0.03970487,
       0.04020487, 0.05052689, 0.20261316, 0.19973359, 0.        ,
       0.02508768, 0.01974519, 0.06980959, 0.60022215, 0.0699095 ,
       0.09966484, 0.03950385], [0.06072197, 0.03961772, 0.08001864, 0.        , 0.01997238,
       0.0996174 , 0.09989904, 0.        , 0.02019914, 0.        ,
       0.03515162, 0.03499185, 0.08095435, 0.00997472, 0.02014243,
       0.01998816, 0.04035087, 0.03984571, 0.03990409, 0.03987295,
       0.03953274, 0.05036187, 0.2008071 , 0.20128591, 0.        ,
       0.02494601, 0.02003394, 0.07004103, 0.60634265, 0.06900243,
       0.09944836, 0.04025444], [0.06021646, 0.03960805, 0.08032646, 0.        , 0.02004005,
       0.0992965 , 0.09982445, 0.        , 0.02033911, 0.        ,
       0.03471695, 0.03511017, 0.08090947, 0.00999671, 0.01980574,
       0.01981697, 0.0400379 , 0.04027377, 0.03988867, 0.04018586,
       0.03973469, 0.05025103, 0.19739987, 0.20069815, 0.        ,
       0.02538458, 0.02013748, 0.07101361, 0.59731499, 0.07041783,
       0.10040017, 0.03944282], [0.05973988, 0.0398461 , 0.07997803, 0.        , 0.01997502,
       0.09920209, 0.09934954, 0.        , 0.02016166, 0.        ,
       0.03467277, 0.03490931, 0.07955905, 0.01000672, 0.020026  ,
       0.01995941, 0.03992882, 0.03967812, 0.04028168, 0.0396845 ,
       0.03981493, 0.05013506, 0.19965361, 0.19726648, 0.        ,
       0.02532272, 0.01978541, 0.06911091, 0.59548044, 0.07061574,
       0.10101112, 0.03992072], [0.060188  , 0.0399582 , 0.08004799, 0.        , 0.02003901,
       0.1004296 , 0.10001248, 0.        , 0.02002607, 0.        ,
       0.035087  , 0.03529196, 0.07969001, 0.01000742, 0.02023454,
       0.01996946, 0.03998217, 0.04013457, 0.04004273, 0.04017104,
       0.03950493, 0.05015132, 0.20256582, 0.19994013, 0.        ,
       0.02494234, 0.0199725 , 0.06982951, 0.59734813, 0.06947538,
       0.10173687, 0.03955453], [0.05931801, 0.04021104, 0.08009503, 0.        , 0.01983001,
       0.09966157, 0.10033357, 0.        , 0.02003683, 0.        ,
       0.03513543, 0.03463056, 0.08086964, 0.01000742, 0.01976403,
       0.02018975, 0.04023764, 0.03969433, 0.03993975, 0.04027227,
       0.04044797, 0.04952894, 0.20239347, 0.19903911, 0.        ,
       0.02510073, 0.02002983, 0.07049956, 0.60480512, 0.07035748,
       0.09932156, 0.04002829]])

    _test_agent_1(env, agent, list_feng_11, list_feng_26, list_fu_he, list_fu_he_Q)
elif kk == 5:
    list_feng_11 = np.array([1.2845388194454506, 1.466599337153487, 1.4769159047624494, 1.4787022361553182, 1.6053780997584868, 1.3041603260328969, 1.6224449439513897, 1.479215531454288, 1.6136838466874224, 1.4706460882986638, 1.6370064148871561, 1.477015833844466, 1.3681664732703396, 1.585744324991875, 1.2903761967025287, 1.4029148407585486, 1.6578528452342758, 1.4905702056393557, 1.5298091332845698, 1.6134263967634757, 1.5466384160143924, 1.3866291201501713, 1.5051073388613845, 1.3428310228148814])
    list_feng_26 = np.array([1.0857184988015516, 0.9410466275188001, 1.0261629884095613, 1.0103724562597263, 1.1004066844331788, 0.9970658485458441, 0.979939464667238, 0.9938389208838067, 1.0820618510916218, 1.0826905632648038, 0.8600545197856397, 0.9273991413611833, 1.1116309385266714, 1.0156221853769187, 1.0837380509054608, 1.005433093522413, 1.0813004654738412, 0.9579213793849666, 1.0300456950540982, 0.9362120803130761, 0.9207348580297965, 0.9903939393114668, 0.9762637898445454, 1.0791144697180106])
    list_fu_he = np.array([
        [0.07696283, 0.06732341, 0.07458069, 0.        , 0.04299188,
       0.14823348, 0.132056  , 0.        , 0.03771728, 0.        ,
       0.04025277, 0.04703868, 0.08871465, 0.04561162, 0.04269487,
       0.03679929, 0.06335321, 0.06775533, 0.06713298, 0.06423429,
       0.05652167, 0.06076688, 0.33153678, 0.28989188, 0.        ,
       0.0397679 , 0.04017852, 0.07260868, 0.15293083, 0.11483033,
       0.14580118, 0.03640254],[0.07834756, 0.05662719, 0.0815476 , 0.        , 0.04441818,
       0.15003894, 0.13770328, 0.        , 0.0429893 , 0.        ,
       0.04094619, 0.04026428, 0.08592182, 0.03873345, 0.03876778,
       0.03902411, 0.06555124, 0.06130346, 0.06304936, 0.0703166 ,
       0.05897813, 0.06488666, 0.29065253, 0.29818063, 0.        ,
       0.03975782, 0.04045052, 0.08299085, 0.14422337, 0.11016448,
       0.13031732, 0.04094491],[0.07098031, 0.06027384, 0.08937654, 0.        , 0.04363869,
       0.14483098, 0.14059247, 0.        , 0.04527419, 0.        ,
       0.04482275, 0.03968049, 0.0798635 , 0.03724361, 0.03873381,
       0.03978385, 0.06308491, 0.05977661, 0.0590697 , 0.06812267,
       0.06478942, 0.06454532, 0.27711547, 0.31579634, 0.        ,
       0.0374865 , 0.04300318, 0.09015285, 0.14107911, 0.10522284,
       0.1493468 , 0.04246677],[0.06738623, 0.06675703, 0.0816229 , 0.        , 0.04562773,
       0.13073285, 0.14837009, 0.        , 0.04203101, 0.        ,
       0.04346556, 0.04409049, 0.08013989, 0.04264979, 0.04294046,
       0.04188707, 0.06579871, 0.0582208 , 0.06223556, 0.06615627,
       0.05968552, 0.06364027, 0.30092417, 0.28311758, 0.        ,
       0.03927186, 0.04596699, 0.09061285, 0.15080917, 0.10356076,
       0.14987742, 0.04334151],[0.07279157, 0.06298928, 0.07943264, 0.        , 0.04055234,
       0.13534361, 0.1381223 , 0.        , 0.03987646, 0.        ,
       0.04386757, 0.03875946, 0.08944214, 0.04292042, 0.03892001,
       0.04154025, 0.06528926, 0.0615528 , 0.05743389, 0.05880126,
       0.06294825, 0.06477711, 0.30818403, 0.28053503, 0.        ,
       0.04436737, 0.04029896, 0.08896184, 0.14740757, 0.10025139,
       0.15693111, 0.03808432],[0.07339377, 0.06047066, 0.08662766, 0.        , 0.04330249,
       0.14179623, 0.14097126, 0.        , 0.04097605, 0.        ,
       0.04658358, 0.04348866, 0.08711975, 0.04220966, 0.04361013,
       0.04240377, 0.06283286, 0.063041  , 0.06409442, 0.05705971,
       0.06749774, 0.0655585 , 0.2894247 , 0.3077562 , 0.        ,
       0.04087549, 0.04027846, 0.08449494, 0.12871271, 0.10771371,
       0.15620696, 0.04489793],[0.09115103, 0.07958808, 0.1100603 , 0.        , 0.05498291,
       0.18235051, 0.17263991, 0.        , 0.05098076, 0.        ,
       0.05194467, 0.0542153 , 0.11581855, 0.05497195, 0.05605761,
       0.05213743, 0.08487685, 0.07745794, 0.07761728, 0.07833817,
       0.07918949, 0.07855357, 0.3781186 , 0.39803331, 0.        ,
       0.05343333, 0.0551755 , 0.10630462, 0.18116046, 0.12905218,
       0.19233497, 0.05379284],[0.0896949 , 0.07992188, 0.10535541, 0.        , 0.05555717,
       0.17268559, 0.17809592, 0.        , 0.0504108 , 0.        ,
       0.05152307, 0.05351606, 0.10953553, 0.05516339, 0.05341362,
       0.05365584, 0.08715302, 0.07645862, 0.0816217 , 0.0808443 ,
       0.07843165, 0.08271273, 0.36886567, 0.36728144, 0.        ,
       0.05111219, 0.05231628, 0.1080625 , 0.19155468, 0.12569388,
       0.18485055, 0.05612624],[0.08498985, 0.07869836, 0.11609335, 0.        , 0.05388261,
       0.17818704, 0.18585388, 0.        , 0.05231435, 0.        ,
       0.05679452, 0.05682879, 0.11040184, 0.05360229, 0.05562161,
       0.05270014, 0.08087224, 0.08502868, 0.08373676, 0.08319019,
       0.0789925 , 0.08037604, 0.3671843 , 0.36505413, 0.        ,
       0.05137419, 0.05097731, 0.10856998, 0.17898785, 0.13264632,
       0.18244145, 0.05395856],[0.08734468, 0.0802289 , 0.11570498, 0.        , 0.05288091,
       0.17724034, 0.17961282, 0.        , 0.05640104, 0.        ,
       0.055951  , 0.0529448 , 0.1029536 , 0.05430532, 0.04983214,
       0.05400112, 0.07876446, 0.08522542, 0.08100323, 0.08381961,
       0.07622284, 0.08433557, 0.38053498, 0.35738965, 0.        ,
       0.05439479, 0.05430445, 0.10458306, 0.17800807, 0.13604941,
       0.19860444, 0.05047155],[0.08955231, 0.08438243, 0.09920736, 0.        , 0.05236888,
       0.1887676 , 0.18465983, 0.        , 0.05418919, 0.        ,
       0.05512155, 0.05591496, 0.10797161, 0.05528902, 0.05134262,
       0.05179242, 0.08046875, 0.08102177, 0.08153713, 0.08481007,
       0.08487157, 0.08257343, 0.39102713, 0.35843437, 0.        ,
       0.05310156, 0.05344303, 0.10609324, 0.17118514, 0.13914146,
       0.18090507, 0.05372474],[0.08511485, 0.08302107, 0.105879  , 0.        , 0.04912438,
       0.18805115, 0.17368341, 0.        , 0.05647493, 0.        ,
       0.05198226, 0.05522182, 0.1004643 , 0.05448203, 0.05760495,
       0.05232201, 0.07538399, 0.07959722, 0.08527994, 0.07766163,
       0.08476119, 0.08230207, 0.36282712, 0.36583262, 0.        ,
       0.05115427, 0.05476029, 0.10967884, 0.18796117, 0.13336235,
       0.2013826 , 0.0523263 ],[0.11100361, 0.09986186, 0.13563449, 0.        , 0.06565709,
       0.22139389, 0.21455506, 0.        , 0.06441744, 0.        ,
       0.06553025, 0.06496579, 0.13472251, 0.06707818, 0.06528657,
       0.06657973, 0.09283303, 0.09563885, 0.0977083 , 0.0999037 ,
       0.09633028, 0.09924693, 0.46149916, 0.47647321, 0.        ,
       0.0663988 , 0.06354504, 0.13029982, 0.22388779, 0.16138541,
       0.23886961, 0.06450226],[0.10955276, 0.09663508, 0.13435086, 0.        , 0.06402809,
       0.2112056 , 0.22079266, 0.        , 0.06393612, 0.        ,
       0.06580724, 0.0664658 , 0.13270517, 0.06424438, 0.06766616,
       0.06636304, 0.1020553 , 0.0930861 , 0.10149786, 0.09954215,
       0.10149675, 0.10138713, 0.43951121, 0.46645632, 0.        ,
       0.06708859, 0.06593502, 0.12808853, 0.22560946, 0.16174514,
       0.23538918, 0.06469188],[0.10442376, 0.09495374, 0.13007069, 0.        , 0.06390357,
       0.22521757, 0.23593087, 0.        , 0.06881664, 0.        ,
       0.0711944 , 0.06878388, 0.13440179, 0.06830613, 0.06714784,
       0.06400993, 0.09649445, 0.09557132, 0.10069467, 0.10004736,
       0.10000718, 0.1008624 , 0.45779991, 0.47344121, 0.        ,
       0.06804207, 0.06523881, 0.13245047, 0.2187311 , 0.16485674,
       0.23198404, 0.06908625],[0.10776681, 0.09784361, 0.12488269, 0.        , 0.06976196,
       0.2178955 , 0.22751013, 0.        , 0.06729801, 0.        ,
       0.06383993, 0.06650794, 0.13181816, 0.06365637, 0.06787285,
       0.06457123, 0.10663286, 0.09955281, 0.10153369, 0.09913872,
       0.09771234, 0.09773104, 0.46777411, 0.45423129, 0.        ,
       0.06779394, 0.06905152, 0.13571938, 0.22156757, 0.15730807,
       0.22582112, 0.06767538],[0.11471939, 0.09756051, 0.13927612, 0.        , 0.06509644,
       0.21355295, 0.22291238, 0.        , 0.06214225, 0.        ,
       0.06440928, 0.06407202, 0.12824886, 0.06498471, 0.06494515,
       0.06583998, 0.10307158, 0.09682792, 0.10033875, 0.096277  ,
       0.09968437, 0.10011464, 0.47232623, 0.47493557, 0.        ,
       0.06730706, 0.06587625, 0.13608321, 0.21409106, 0.16339343,
       0.23403473, 0.0640628 ],[0.10920951, 0.10119797, 0.133908  , 0.        , 0.0663989 ,
       0.23044224, 0.22013471, 0.        , 0.06442798, 0.        ,
       0.0665344 , 0.06641419, 0.12862277, 0.06278215, 0.06781835,
       0.06702409, 0.09338205, 0.09971016, 0.09864342, 0.1022227 ,
       0.09547167, 0.0977611 , 0.46381148, 0.4828527 , 0.        ,
       0.06604275, 0.06620631, 0.13750072, 0.22686609, 0.16200267,
       0.22934465, 0.06230383],[0.08596456, 0.08465942, 0.11085096, 0.        , 0.05338433,
       0.16569157, 0.17373914, 0.        , 0.05307402, 0.        ,
       0.05344188, 0.05507533, 0.10262313, 0.05272156, 0.0529174 ,
       0.05299236, 0.07628116, 0.08150081, 0.08429247, 0.07965972,
       0.08545106, 0.08044434, 0.37197289, 0.3912321 , 0.        ,
       0.05935959, 0.05453021, 0.10216043, 0.17959284, 0.1353469 ,
       0.20159685, 0.05186689],[0.09079606, 0.07961567, 0.10281513, 0.        , 0.05543156,
       0.19285747, 0.17912257, 0.        , 0.05399375, 0.        ,
       0.05300081, 0.05645779, 0.10136916, 0.05328588, 0.05681547,
       0.05477201, 0.08240145, 0.08214271, 0.08115627, 0.08252907,
       0.0774056 , 0.08491859, 0.36176268, 0.38225319, 0.        ,
       0.05159236, 0.05844948, 0.10686085, 0.17659232, 0.13230135,
       0.18732407, 0.05476087],[0.08732897, 0.07933367, 0.11524584, 0.        , 0.05416868,
       0.18882853, 0.17650802, 0.        , 0.05005429, 0.        ,
       0.05567167, 0.05771549, 0.10833471, 0.05444623, 0.05657148,
       0.05507963, 0.07844618, 0.07758698, 0.08050943, 0.08557657,
       0.080794  , 0.08379298, 0.36868405, 0.38074502, 0.        ,
       0.05387967, 0.05416724, 0.09970565, 0.18034536, 0.13748186,
       0.18065546, 0.0507797 ],[0.09263166, 0.08227699, 0.11319582, 0.        , 0.05706645,
       0.18207524, 0.19104584, 0.        , 0.05253159, 0.        ,
       0.05033889, 0.05333947, 0.1136683 , 0.05463799, 0.05228168,
       0.05397133, 0.08405274, 0.0800364 , 0.0818654 , 0.07903034,
       0.08489132, 0.07885767, 0.39333195, 0.36495739, 0.        ,
       0.05626288, 0.05436074, 0.10789886, 0.17550558, 0.13741899,
       0.18206356, 0.05604409],[0.08970966, 0.08140369, 0.10533199, 0.        , 0.05396091,
       0.17332204, 0.18003482, 0.        , 0.05060236, 0.        ,
       0.05251673, 0.05285289, 0.11518485, 0.05430539, 0.05705556,
       0.05486746, 0.08361226, 0.08095909, 0.07541219, 0.07879321,
       0.08086455, 0.08143498, 0.39068593, 0.36354921, 0.        ,
       0.05374675, 0.05625887, 0.11121938, 0.18497863, 0.13689291,
       0.18638971, 0.05478132],[0.08976996, 0.07479679, 0.11412968, 0.        , 0.05308509,
       0.17614872, 0.17319176, 0.        , 0.05434886, 0.        ,
       0.05276328, 0.05246466, 0.10622003, 0.05698412, 0.05620278,
       0.0570642 , 0.07853756, 0.08294865, 0.08473843, 0.07774808,
       0.0782572 , 0.0783051 , 0.40643993, 0.3503638 , 0.        ,
       0.04928312, 0.05634239, 0.10454405, 0.16955925, 0.13449713,
       0.19448545, 0.05165044]])
    list_fu_he_Q = np.array([
        [0.05874887, 0.03935787, 0.07629036, 0.        , 0.0190127 ,
       0.09893342, 0.0983558 , 0.        , 0.01996932, 0.        ,
       0.0349448 , 0.03334855, 0.07699319, 0.00955078, 0.01957647,
       0.01915163, 0.03964091, 0.03871501, 0.03991854, 0.03834718,
       0.03991283, 0.04874542, 0.19268194, 0.1990497 , 0.        ,
       0.02467662, 0.0190369 , 0.06876419, 0.58622744, 0.06749794,
       0.09852773, 0.03907656],[0.05806798, 0.03920242, 0.07676988, 0.        , 0.01965595,
       0.0960711 , 0.09864259, 0.        , 0.01975987, 0.        ,
       0.03462311, 0.03397622, 0.07639391, 0.00971739, 0.01933025,
       0.01967951, 0.03904706, 0.03843348, 0.03901287, 0.03923709,
       0.03858245, 0.04901236, 0.19489852, 0.19535744, 0.        ,
       0.02453017, 0.0194858 , 0.067584  , 0.58504717, 0.0690731 ,
       0.09887606, 0.03952318],[0.05860531, 0.03833444, 0.07901147, 0.        , 0.01957694,
       0.09912406, 0.09758199, 0.        , 0.0192154 , 0.        ,
       0.03415213, 0.03425584, 0.07833261, 0.00996128, 0.019463  ,
       0.01954197, 0.03847535, 0.03932761, 0.03916523, 0.03828245,
       0.0387085 , 0.04781211, 0.19412695, 0.19769026, 0.        ,
       0.02454983, 0.01932605, 0.06895279, 0.57842004, 0.06835328,
       0.09749992, 0.03950944],[0.05873745, 0.03889588, 0.07856413, 0.        , 0.01969052,
       0.09826629, 0.09727401, 0.        , 0.01949235, 0.        ,
       0.03395826, 0.03454337, 0.07727817, 0.00991691, 0.01913525,
       0.01966422, 0.03855924, 0.03867021, 0.03935212, 0.03874441,
       0.03930399, 0.04815865, 0.1948915 , 0.19690241, 0.        ,
       0.02480685, 0.01933013, 0.06771006, 0.59026315, 0.06853267,
       0.0979165 , 0.03871245],[0.05773844, 0.03910924, 0.07812659, 0.        , 0.01962275,
       0.09656005, 0.09765659, 0.        , 0.01923682, 0.        ,
       0.03416729, 0.03444095, 0.07771817, 0.00971613, 0.01950744,
       0.01972451, 0.03896923, 0.03941965, 0.03896406, 0.03854574,
       0.03882374, 0.04908796, 0.19507474, 0.19355215, 0.        ,
       0.02456367, 0.01948158, 0.06803371, 0.58470553, 0.06908181,
       0.09790472, 0.03852967],[0.05946378, 0.03931518, 0.07689347, 0.        , 0.0194298 ,
       0.09870419, 0.09746762, 0.        , 0.01923121, 0.        ,
       0.03367383, 0.03407938, 0.07775526, 0.0097953 , 0.01929054,
       0.01957707, 0.0385326 , 0.03928947, 0.03942919, 0.03904246,
       0.03905148, 0.04882686, 0.19687809, 0.19242696, 0.        ,
       0.02423839, 0.01942013, 0.06798611, 0.58617385, 0.06925239,
       0.09649326, 0.03920222],[0.06002113, 0.0400301 , 0.07894843, 0.        , 0.01968588,
       0.09851196, 0.10013998, 0.        , 0.01980006, 0.        ,
       0.03470977, 0.03500254, 0.079971  , 0.01008399, 0.02009288,
       0.01996434, 0.03980334, 0.03975365, 0.04018431, 0.04044824,
       0.03962229, 0.05004949, 0.20052349, 0.20126102, 0.        ,
       0.02529173, 0.01989952, 0.06979789, 0.60188498, 0.070201  ,
       0.10026984, 0.04046886],[0.05955771, 0.03953752, 0.08001516, 0.        , 0.01986845,
       0.10013821, 0.0996336 , 0.        , 0.01996075, 0.        ,
       0.03453747, 0.03536473, 0.08031303, 0.01000937, 0.01991052,
       0.01987707, 0.04048657, 0.03999658, 0.04040243, 0.04048744,
       0.0398724 , 0.05057833, 0.19759078, 0.20110811, 0.        ,
       0.025204  , 0.019933  , 0.0704202 , 0.59871887, 0.07095607,
       0.10062553, 0.03968938],[0.0594483 , 0.04016921, 0.0799161 , 0.        , 0.02000795,
       0.09965539, 0.0989534 , 0.        , 0.01987359, 0.        ,
       0.03531883, 0.03501362, 0.07939049, 0.0101234 , 0.0197628 ,
       0.0198652 , 0.03991268, 0.04019821, 0.04008507, 0.03984782,
       0.03971979, 0.05055986, 0.20287795, 0.20236417, 0.        ,
       0.02482402, 0.01983464, 0.07030878, 0.59348036, 0.0701156 ,
       0.10070025, 0.0403781 ],[0.05981061, 0.03954233, 0.07869088, 0.        , 0.01978033,
       0.10010868, 0.09962675, 0.        , 0.02010455, 0.        ,
       0.03547594, 0.03475838, 0.07966084, 0.01000963, 0.02013832,
       0.02023837, 0.04014335, 0.03998909, 0.04039681, 0.04006958,
       0.03976287, 0.05028489, 0.19900867, 0.19997503, 0.        ,
       0.02477483, 0.0201416 , 0.07008307, 0.60211264, 0.07045011,
       0.09952512, 0.03990493],[0.06050516, 0.04044238, 0.07952001, 0.        , 0.01999393,
       0.09976328, 0.09977773, 0.        , 0.01978581, 0.        ,
       0.03456409, 0.03476089, 0.08022062, 0.00995185, 0.019725  ,
       0.01997069, 0.03998142, 0.03994434, 0.04040511, 0.04011716,
       0.03986336, 0.04937856, 0.19914143, 0.20082601, 0.        ,
       0.02484929, 0.01991116, 0.07042915, 0.60019903, 0.07000684,
       0.09934506, 0.03985656],[0.0596078 , 0.04016898, 0.07995485, 0.        , 0.02022061,
       0.09965472, 0.10074104, 0.        , 0.0201312 , 0.        ,
       0.03466454, 0.03507934, 0.08009799, 0.00994279, 0.0201218 ,
       0.01984814, 0.04034367, 0.03989429, 0.04018325, 0.03972419,
       0.04019986, 0.04962857, 0.20097229, 0.20094947, 0.        ,
       0.02516024, 0.01990263, 0.06972994, 0.59622909, 0.07029319,
       0.10038156, 0.03984278],[0.06124875, 0.04081282, 0.08084603, 0.        , 0.02080266,
       0.10225581, 0.10287066, 0.        , 0.02042391, 0.        ,
       0.03584373, 0.03630786, 0.08319995, 0.01027936, 0.02047589,
       0.02026905, 0.04115798, 0.04162233, 0.04118943, 0.04082494,
       0.0409734 , 0.05106588, 0.20369281, 0.20681974, 0.        ,
       0.02585609, 0.02061997, 0.07168126, 0.615609  , 0.07142966,
       0.10223853, 0.04110484],[0.0611974 , 0.04095845, 0.08129284, 0.        , 0.02030743,
       0.10200618, 0.10237066, 0.        , 0.02028948, 0.        ,
       0.03568459, 0.03610483, 0.08315831, 0.01026276, 0.02068609,
       0.02076302, 0.04103911, 0.04112517, 0.04112894, 0.04085145,
       0.04079233, 0.05079628, 0.20518122, 0.20404477, 0.        ,
       0.02586094, 0.02070891, 0.07096602, 0.61137509, 0.07236745,
       0.10218468, 0.0413025 ],[0.06072746, 0.04154319, 0.08303739, 0.        , 0.02058222,
       0.1026499 , 0.10245151, 0.        , 0.02038314, 0.        ,
       0.03573327, 0.03621439, 0.08334037, 0.01025637, 0.02053766,
       0.02040793, 0.04141289, 0.04129246, 0.04050498, 0.0404321 ,
       0.0408606 , 0.05125695, 0.2062541 , 0.20514547, 0.        ,
       0.02592546, 0.02047495, 0.07211171, 0.60979808, 0.07144166,
       0.10239789, 0.0409272 ],[0.06156146, 0.040674  , 0.08268916, 0.        , 0.0207464 ,
       0.10271325, 0.10262616, 0.        , 0.02036249, 0.        ,
       0.03608531, 0.03588593, 0.08066138, 0.01019916, 0.02062434,
       0.02059212, 0.04044171, 0.04129735, 0.04067585, 0.04073485,
       0.04096313, 0.05070692, 0.20141849, 0.20426203, 0.        ,
       0.02552349, 0.02047793, 0.07233254, 0.61552825, 0.07243496,
       0.10164389, 0.04116201],[0.06167403, 0.04093943, 0.08233241, 0.        , 0.02065867,
       0.10110322, 0.1014108 , 0.        , 0.02072616, 0.        ,
       0.0356698 , 0.03594894, 0.08181747, 0.01034551, 0.02026691,
       0.02032573, 0.04072269, 0.04054981, 0.04053078, 0.04134572,
       0.0409915 , 0.05175866, 0.20421618, 0.20572029, 0.        ,
       0.0252491 , 0.02065983, 0.07221184, 0.61100631, 0.07144934,
       0.10245634, 0.0408458 ],[0.06065486, 0.04163883, 0.08158016, 0.        , 0.02039012,
       0.10273199, 0.10276186, 0.        , 0.02049264, 0.        ,
       0.03597231, 0.0362839 , 0.08207711, 0.01031723, 0.02035172,
       0.0205896 , 0.04102722, 0.04070978, 0.04110831, 0.04052886,
       0.04114153, 0.05168051, 0.2039941 , 0.20478319, 0.        ,
       0.02575865, 0.02037399, 0.07246312, 0.61555953, 0.07195185,
       0.10365731, 0.04079269],[0.05915369, 0.0398119 , 0.07983675, 0.        , 0.02021937,
       0.10012727, 0.09987264, 0.        , 0.01994688, 0.        ,
       0.03512159, 0.03499312, 0.07998975, 0.01002186, 0.02001565,
       0.01998697, 0.03996353, 0.03967463, 0.03968983, 0.04021304,
       0.04058889, 0.05009141, 0.20080857, 0.19964248, 0.        ,
       0.02484273, 0.02017543, 0.06962819, 0.59706484, 0.07073233,
       0.1001137 , 0.03955616],[0.06039184, 0.04016535, 0.07951088, 0.        , 0.01981426,
       0.09983417, 0.1010453 , 0.        , 0.02006952, 0.        ,
       0.03498628, 0.03553241, 0.07962797, 0.01003935, 0.02000234,
       0.01974373, 0.03987063, 0.0400329 , 0.03967336, 0.03959682,
       0.04037767, 0.04989202, 0.19874779, 0.19984355, 0.        ,
       0.02461517, 0.01987002, 0.07005233, 0.59871705, 0.06946779,
       0.09983803, 0.03974008],[0.0607081 , 0.04015443, 0.07984481, 0.        , 0.02008618,
       0.099917  , 0.09980357, 0.        , 0.01998292, 0.        ,
       0.03539355, 0.03494342, 0.07989552, 0.00994445, 0.01995664,
       0.01982669, 0.03955409, 0.04020464, 0.04020631, 0.0402817 ,
       0.03983241, 0.05048176, 0.20089958, 0.20212962, 0.        ,
       0.0250866 , 0.01987696, 0.07025593, 0.59096363, 0.06924399,
       0.10015567, 0.03984435],[0.0604407 , 0.03979087, 0.07997534, 0.        , 0.02015067,
       0.09963887, 0.09924171, 0.        , 0.02020363, 0.        ,
       0.03516871, 0.03513527, 0.08052074, 0.01005257, 0.02009884,
       0.01989894, 0.03976873, 0.04000745, 0.03998426, 0.04010942,
       0.04016567, 0.04994253, 0.20036543, 0.19962371, 0.        ,
       0.0251499 , 0.02018093, 0.07110257, 0.60461207, 0.07006917,
       0.09957692, 0.03965678],[0.06029337, 0.03987074, 0.08020073, 0.        , 0.02017791,
       0.09961503, 0.10118313, 0.        , 0.01999257, 0.        ,
       0.03458341, 0.03499881, 0.08052017, 0.0099429 , 0.01992418,
       0.02012003, 0.040037  , 0.03987175, 0.04060107, 0.04014282,
       0.04021728, 0.05027643, 0.20063799, 0.20099059, 0.        ,
       0.02516092, 0.01991041, 0.07021487, 0.60264022, 0.07000849,
       0.09981579, 0.03998751],[0.05988008, 0.04009857, 0.08045858, 0.        , 0.02002692,
       0.10127077, 0.10001895, 0.        , 0.01991133, 0.        ,
       0.03477883, 0.03486347, 0.08059417, 0.00995734, 0.0198668 ,
       0.02023115, 0.04029247, 0.04014923, 0.03954054, 0.039997  ,
       0.04009418, 0.05029587, 0.20009948, 0.20113236, 0.        ,
       0.02516342, 0.01991044, 0.07072558, 0.59634087, 0.06998399,
       0.09943183, 0.04019633]])

    _test_agent_1(env, agent, list_feng_11, list_feng_26, list_fu_he, list_fu_he_Q)
elif kk == 6:
    list_feng_11 = np.array([1.3620999558013387, 1.3941736371676159, 1.5389656904750253, 1.5878059581979707, 1.6104655008054514, 1.448175236979864, 1.5823540130155032, 1.4903034632984793, 1.450746288524611, 1.4360263764762733, 1.6869093843163538, 1.5864477262468868, 1.3711970790852237, 1.3464067327285367, 1.4011947386903405, 1.5214133406058639, 1.4043089283578047, 1.4731933426606085, 1.3927155011401346, 1.5230342816674778, 1.5998538958875161, 1.550407623846473, 1.401317656611825, 1.4813398710274428])
    list_feng_26 = np.array([1.1091203075429061, 0.981702444034, 1.0317812376003528, 0.9906868971023903, 1.1348582607810225, 0.995908378044345, 0.9400629285549458, 1.0448657508451638, 1.0304526944480887, 1.0021352033179851, 1.0360266877513555, 1.115654601318009, 1.0062120152424094, 0.9794358664714675, 0.9728278533829054, 0.983098655903534, 0.9987294753872459, 1.0873047614549347, 0.965520161401226, 0.9840176408271157, 1.0389595411097097, 0.8854878409791404, 0.929403330332269, 1.0746445294349796])
    list_fu_he = np.array([
        [0.07363457, 0.0566504 , 0.07320215, 0.        , 0.04528769,
       0.14506939, 0.14633476, 0.        , 0.03602009, 0.        ,
       0.03635948, 0.04225999, 0.0814984 , 0.0397427 , 0.04750491,
       0.04250501, 0.05974002, 0.06344099, 0.05760204, 0.06110307,
       0.06212779, 0.0679303 , 0.32341255, 0.32792689, 0.        ,
       0.04242442, 0.04039615, 0.08789023, 0.14207341, 0.09815734,
       0.14406095, 0.04345254],[0.06958985, 0.06465364, 0.07588876, 0.        , 0.0402333 ,
       0.13036721, 0.1309099 , 0.        , 0.04028532, 0.        ,
       0.04624145, 0.0433725 , 0.08962139, 0.04129917, 0.03897739,
       0.0435917 , 0.06025535, 0.06172457, 0.0667922 , 0.06499735,
       0.06074983, 0.06015731, 0.29045375, 0.31319066, 0.        ,
       0.04418438, 0.04152991, 0.08181624, 0.14344146, 0.10693857,
       0.13853921, 0.04173935],[0.0720165 , 0.06674368, 0.07324133, 0.        , 0.04163002,
       0.1434408 , 0.14773572, 0.        , 0.04227449, 0.        ,
       0.04203094, 0.04350094, 0.09553631, 0.04576658, 0.04609668,
       0.03999004, 0.06559676, 0.06082481, 0.05852746, 0.06199195,
       0.06418376, 0.05810057, 0.30913365, 0.28691586, 0.        ,
       0.0385112 , 0.04059325, 0.08076332, 0.14471493, 0.10418452,
       0.1460654 , 0.04110027],[0.06836109, 0.06341689, 0.08378703, 0.        , 0.03995199,
       0.14426063, 0.12836163, 0.        , 0.04059682, 0.        ,
       0.04459107, 0.03935764, 0.08701115, 0.04300468, 0.04204601,
       0.04262523, 0.06669272, 0.06273439, 0.06405978, 0.06578446,
       0.06519038, 0.06656773, 0.27521284, 0.27701717, 0.        ,
       0.04018118, 0.04316524, 0.0807396 , 0.13796109, 0.10389097,
       0.1441713 , 0.04218331],[0.07260768, 0.06401207, 0.08261876, 0.        , 0.03909162,
       0.14037742, 0.14917723, 0.        , 0.04104453, 0.        ,
       0.04240984, 0.04362144, 0.08155561, 0.04102398, 0.04133025,
       0.04343637, 0.05909554, 0.06424243, 0.06209144, 0.06087944,
       0.07028371, 0.06482582, 0.31482702, 0.31213358, 0.        ,
       0.03906648, 0.04008476, 0.08999514, 0.13111182, 0.11345699,
       0.13832149, 0.03820401],[0.07269977, 0.06573333, 0.0840187 , 0.        , 0.0436926 ,
       0.13117004, 0.13097274, 0.        , 0.04028475, 0.        ,
       0.03941216, 0.04215531, 0.07849481, 0.04224192, 0.04198313,
       0.04150663, 0.0601207 , 0.05764838, 0.0576605 , 0.05588807,
       0.05970188, 0.05769809, 0.27356074, 0.30045879, 0.        ,
       0.0397501 , 0.04325291, 0.08331295, 0.12757087, 0.10529444,
       0.1415557 , 0.04140945],[0.08663768, 0.07693053, 0.11638572, 0.        , 0.05239772,
       0.1749592 , 0.17180014, 0.        , 0.05781401, 0.        ,
       0.05271471, 0.05485433, 0.1085017 , 0.05518135, 0.05438908,
       0.05845021, 0.08195172, 0.07803579, 0.08310393, 0.08498115,
       0.08178873, 0.07487687, 0.34699822, 0.38387075, 0.        ,
       0.05498971, 0.0525072 , 0.10934435, 0.18034354, 0.13671307,
       0.19034909, 0.05410925],[0.0878855 , 0.08225359, 0.11230059, 0.        , 0.05153188,
       0.17910916, 0.18739059, 0.        , 0.05399068, 0.        ,
       0.05311264, 0.05744986, 0.11017742, 0.05409548, 0.05653473,
       0.0587066 , 0.08063921, 0.08355069, 0.08340266, 0.0797615 ,
       0.07848289, 0.07872917, 0.36937223, 0.37330017, 0.        ,
       0.05523292, 0.05405152, 0.10846505, 0.17220542, 0.13376394,
       0.18321916, 0.05127335],[0.09050192, 0.07828315, 0.10945742, 0.        , 0.05169195,
       0.18826893, 0.17713007, 0.        , 0.05247302, 0.        ,
       0.05549854, 0.05156562, 0.10799891, 0.05452162, 0.05774299,
       0.05328974, 0.08071111, 0.07880314, 0.07485106, 0.08097299,
       0.07969279, 0.08020993, 0.35664255, 0.37659034, 0.        ,
       0.05603616, 0.0538246 , 0.10997654, 0.1723487 , 0.1338844 ,
       0.18608777, 0.05683051],[0.09310131, 0.08578311, 0.10746826, 0.        , 0.05313619,
       0.17650218, 0.18633975, 0.        , 0.05150065, 0.        ,
       0.05604343, 0.05241215, 0.11239399, 0.05713293, 0.05634636,
       0.0557702 , 0.07963535, 0.08346488, 0.08134274, 0.07868183,
       0.07928988, 0.0807529 , 0.36264164, 0.37636544, 0.        ,
       0.05355215, 0.05197345, 0.10642397, 0.17775502, 0.13554016,
       0.19787496, 0.05531569],[0.09374877, 0.08612035, 0.10485199, 0.        , 0.05515189,
       0.18308222, 0.19160275, 0.        , 0.0492583 , 0.        ,
       0.04891029, 0.04996548, 0.10271845, 0.05283653, 0.0535717 ,
       0.05404205, 0.08041559, 0.08112857, 0.08574161, 0.08186261,
       0.07891009, 0.07654488, 0.4092137 , 0.37283451, 0.        ,
       0.05576166, 0.05704917, 0.10830782, 0.1748654 , 0.12717371,
       0.19082342, 0.05542238],[0.08780737, 0.08481705, 0.10229543, 0.        , 0.05342041,
       0.17911672, 0.17700227, 0.        , 0.05325223, 0.        ,
       0.05693083, 0.0551222 , 0.11169068, 0.0545289 , 0.0524501 ,
       0.05516988, 0.08290888, 0.08177689, 0.07961555, 0.08292081,
       0.07876641, 0.08093696, 0.38297489, 0.37006639, 0.        ,
       0.05337007, 0.05600391, 0.10939976, 0.17776228, 0.13879772,
       0.17369916, 0.05219218],[0.11418315, 0.09751056, 0.12913635, 0.        , 0.06671888,
       0.21544526, 0.22645656, 0.        , 0.06171972, 0.        ,
       0.06385337, 0.06458507, 0.13036404, 0.07014383, 0.06212285,
       0.0664663 , 0.098063  , 0.10312828, 0.10365702, 0.10113173,
       0.09443553, 0.1001245 , 0.45048227, 0.44340178, 0.        ,
       0.06710773, 0.06655345, 0.12509821, 0.20997632, 0.1672133 ,
       0.23602066, 0.06551212],[0.11338597, 0.0977883 , 0.13425738, 0.        , 0.06484466,
       0.21033362, 0.22166197, 0.        , 0.06548546, 0.        ,
       0.06584123, 0.06593954, 0.13399046, 0.06460706, 0.06632318,
       0.06707027, 0.09564382, 0.09943886, 0.0966033 , 0.10015965,
       0.10243676, 0.09557694, 0.47498908, 0.45564358, 0.        ,
       0.06421231, 0.06241565, 0.13658317, 0.21844498, 0.16289619,
       0.24895028, 0.06562667],[0.10684765, 0.09774136, 0.13029482, 0.        , 0.06765925,
       0.22243634, 0.20731426, 0.        , 0.06313374, 0.        ,
       0.06657686, 0.06345953, 0.12239446, 0.06401825, 0.06788863,
       0.06615461, 0.09954213, 0.1033633 , 0.09777329, 0.10240769,
       0.09848563, 0.10086689, 0.48784907, 0.47785293, 0.        ,
       0.0637085 , 0.06276843, 0.12879242, 0.21304261, 0.16230308,
       0.24422063, 0.06733162],[0.10475304, 0.10040459, 0.14103776, 0.        , 0.06631064,
       0.20862449, 0.22364769, 0.        , 0.06348359, 0.        ,
       0.0669063 , 0.0656708 , 0.13304765, 0.0666963 , 0.0688824 ,
       0.06709288, 0.09780771, 0.10333406, 0.10337064, 0.09661114,
       0.10542518, 0.0946681 , 0.48393327, 0.44433861, 0.        ,
       0.06849618, 0.06584141, 0.12673568, 0.22906765, 0.16565594,
       0.23533398, 0.06622458],[0.11306362, 0.09720284, 0.13803823, 0.        , 0.06598159,
       0.22180402, 0.22088551, 0.        , 0.06907843, 0.        ,
       0.06406816, 0.06651002, 0.13446746, 0.06473616, 0.06570816,
       0.06450429, 0.09684657, 0.10089708, 0.1038577 , 0.09841231,
       0.09942116, 0.09999158, 0.44764902, 0.47143058, 0.        ,
       0.06529766, 0.06639645, 0.12972169, 0.21113699, 0.16384292,
       0.2265657 , 0.06364245],[0.11614688, 0.09764906, 0.13201849, 0.        , 0.06770651,
       0.21722199, 0.20792188, 0.        , 0.06720388, 0.        ,
       0.06656254, 0.06687473, 0.12364582, 0.06619504, 0.06827344,
       0.06591622, 0.09520351, 0.10352796, 0.10084704, 0.09740674,
       0.10321564, 0.09668739, 0.45998042, 0.46512242, 0.        ,
       0.06823779, 0.06680014, 0.13424062, 0.21676635, 0.1653531 ,
       0.22372758, 0.06507624],[0.08731914, 0.08243802, 0.10992819, 0.        , 0.05372083,
       0.18311732, 0.18627656, 0.        , 0.05224485, 0.        ,
       0.05193727, 0.05659418, 0.10941221, 0.04959269, 0.05436489,
       0.05435119, 0.0837855 , 0.08615592, 0.08452531, 0.07670673,
       0.07685417, 0.08324424, 0.40518128, 0.37959181, 0.        ,
       0.05440767, 0.05523286, 0.10254553, 0.17432114, 0.13345149,
       0.18274945, 0.05577087],[0.09162786, 0.08522037, 0.10282894, 0.        , 0.05330481,
       0.17389122, 0.17218277, 0.        , 0.05252796, 0.        ,
       0.05420789, 0.0514442 , 0.10982393, 0.05351484, 0.05369689,
       0.05374372, 0.08118677, 0.08084149, 0.07809068, 0.08239084,
       0.08433388, 0.08222825, 0.40834483, 0.36661612, 0.        ,
       0.05232056, 0.05427006, 0.11468358, 0.17528885, 0.12919073,
       0.18609715, 0.05381945],[0.09029399, 0.0802506 , 0.11239649, 0.        , 0.05676819,
       0.17999426, 0.18320206, 0.        , 0.05379883, 0.        ,
       0.05488894, 0.054944  , 0.10358967, 0.05643026, 0.05211133,
       0.05823779, 0.07909179, 0.08232035, 0.07629785, 0.07999948,
       0.08346144, 0.08421399, 0.37245296, 0.40040859, 0.        ,
       0.05339625, 0.05416528, 0.10321709, 0.1834148 , 0.1391411 ,
       0.19082133, 0.05163794],[0.0858703 , 0.08349427, 0.10372729, 0.        , 0.05150233,
       0.17454378, 0.18255204, 0.        , 0.05225399, 0.        ,
       0.0503665 , 0.0510884 , 0.10511834, 0.05525818, 0.05456428,
       0.05198834, 0.08298965, 0.08256365, 0.08188402, 0.08524419,
       0.07999363, 0.08213689, 0.39310674, 0.37632708, 0.        ,
       0.05289195, 0.05472612, 0.1106683 , 0.18194053, 0.13495688,
       0.18836293, 0.05282245],[0.09201333, 0.08525314, 0.10753151, 0.        , 0.05521196,
       0.17679029, 0.18659899, 0.        , 0.05229969, 0.        ,
       0.05528908, 0.05273189, 0.11014114, 0.05558645, 0.054218  ,
       0.05245964, 0.07654661, 0.07972659, 0.0765715 , 0.07592037,
       0.07768152, 0.07666496, 0.37797488, 0.37206487, 0.        ,
       0.05720038, 0.05076255, 0.10676614, 0.18598162, 0.14110585,
       0.18576176, 0.05246873],[0.09575553, 0.07794825, 0.10753576, 0.        , 0.05455484,
       0.17787951, 0.18604994, 0.        , 0.05289882, 0.        ,
       0.0567677 , 0.05410004, 0.10484722, 0.05280596, 0.05314774,
       0.05207024, 0.08107941, 0.08006879, 0.07997066, 0.07956165,
       0.08622529, 0.08505187, 0.36872948, 0.36097789, 0.        ,
       0.05370453, 0.05479198, 0.11017513, 0.17471257, 0.1399263 ,
       0.1933726 , 0.0565001 ]])
    list_fu_he_Q = np.array([
        [0.05910674, 0.03874589, 0.07723042, 0.        , 0.01936332,
       0.09502105, 0.09680307, 0.        , 0.01937276, 0.        ,
       0.03330401, 0.03382002, 0.0781552 , 0.00957489, 0.01941611,
       0.01963416, 0.03885837, 0.03912948, 0.03907421, 0.03909483,
       0.03852274, 0.04755352, 0.19366374, 0.19424288, 0.        ,
       0.02382208, 0.0194103 , 0.06809626, 0.57170267, 0.066753  ,
       0.09975111, 0.03977089],[0.05834886, 0.03821568, 0.07835002, 0.        , 0.01974084,
       0.09722274, 0.09657035, 0.        , 0.01947667, 0.        ,
       0.0348001 , 0.03421463, 0.07844948, 0.00979341, 0.01919728,
       0.01981698, 0.03980261, 0.03972243, 0.03886614, 0.03875457,
       0.03830962, 0.0487261 , 0.19115034, 0.19720461, 0.        ,
       0.02428244, 0.01967125, 0.06915788, 0.58461637, 0.06878931,
       0.09740391, 0.03829603],[0.05825882, 0.03914164, 0.07817142, 0.        , 0.01941755,
       0.09890422, 0.09852003, 0.        , 0.01978621, 0.        ,
       0.03397718, 0.03374153, 0.07784937, 0.00970785, 0.01944716,
       0.01948526, 0.03924513, 0.03872973, 0.03896545, 0.03878349,
       0.03864016, 0.0497075 , 0.197518  , 0.19307829, 0.        ,
       0.02431625, 0.01949868, 0.0691616 , 0.58784431, 0.06779642,
       0.09658977, 0.03833093],[0.05771873, 0.03941124, 0.07743389, 0.        , 0.01960771,
       0.0964992 , 0.09685703, 0.        , 0.01955993, 0.        ,
       0.03412301, 0.03370425, 0.07716983, 0.00976733, 0.0193509 ,
       0.01934097, 0.039086  , 0.0388266 , 0.03924616, 0.03875348,
       0.03876124, 0.0488807 , 0.19773861, 0.19531919, 0.        ,
       0.02468254, 0.01955595, 0.06910975, 0.57918092, 0.06851781,
       0.09720497, 0.03911443],[0.05832597, 0.03938655, 0.07793391, 0.        , 0.0195313 ,
       0.09643299, 0.0986366 , 0.        , 0.01928117, 0.        ,
       0.03405337, 0.03399822, 0.07731305, 0.00977414, 0.01956166,
       0.01930765, 0.03846175, 0.03843628, 0.03921356, 0.03925071,
       0.03931312, 0.0496247 , 0.19497969, 0.19404062, 0.        ,
       0.02453049, 0.01952919, 0.06835211, 0.59284099, 0.0682748 ,
       0.09797889, 0.03915218],[0.05740714, 0.0391924 , 0.07680076, 0.        , 0.0195333 ,
       0.09739503, 0.09760441, 0.        , 0.01915541, 0.        ,
       0.03426765, 0.0347288 , 0.07688635, 0.0097636 , 0.01943414,
       0.01935726, 0.03926441, 0.03938231, 0.03929494, 0.03826354,
       0.03920807, 0.04859044, 0.19252045, 0.19441577, 0.        ,
       0.02456202, 0.01970478, 0.06692385, 0.58603847, 0.06848022,
       0.09688302, 0.03936664],[0.05987865, 0.04011146, 0.07906848, 0.        , 0.01975017,
       0.09938656, 0.10038641, 0.        , 0.02006547, 0.        ,
       0.03472931, 0.03510161, 0.08038325, 0.01001283, 0.02003109,
       0.02006331, 0.04022796, 0.04010858, 0.03998554, 0.03971917,
       0.03977638, 0.04975375, 0.19872088, 0.19909825, 0.        ,
       0.02490436, 0.02005766, 0.07005904, 0.59817819, 0.07018139,
       0.1003784 , 0.0403256 ],[0.05972123, 0.03992293, 0.0800805 , 0.        , 0.02023471,
       0.09928879, 0.09976082, 0.        , 0.01993536, 0.        ,
       0.03497427, 0.03511129, 0.08021065, 0.01001937, 0.01997437,
       0.0198479 , 0.0394468 , 0.03960896, 0.04007052, 0.03974732,
       0.03988663, 0.05014946, 0.20159854, 0.20112461, 0.        ,
       0.02506058, 0.01983727, 0.06959535, 0.59838734, 0.07087123,
       0.10062184, 0.03991038],[0.05990704, 0.03992515, 0.08037409, 0.        , 0.02004684,
       0.10016519, 0.0993743 , 0.        , 0.02010536, 0.        ,
       0.03462478, 0.03526294, 0.07924623, 0.00992784, 0.02005221,
       0.019962  , 0.03977066, 0.03993352, 0.03932293, 0.03993972,
       0.039704  , 0.0503361 , 0.20168694, 0.20060235, 0.        ,
       0.02519374, 0.02000324, 0.06933189, 0.59635214, 0.07005781,
       0.10131405, 0.03989789],[0.06043406, 0.04015   , 0.07974581, 0.        , 0.02007739,
       0.10060038, 0.10037957, 0.        , 0.02004618, 0.        ,
       0.03485319, 0.03540637, 0.0799481 , 0.00994992, 0.0200172 ,
       0.01988868, 0.04004184, 0.04037855, 0.03997749, 0.03999353,
       0.04019891, 0.049633  , 0.19921365, 0.19957795, 0.        ,
       0.02472562, 0.02009894, 0.06993645, 0.59292792, 0.06949038,
       0.10047303, 0.03977004],[0.06022701, 0.03980223, 0.08040296, 0.        , 0.01989309,
       0.09973467, 0.09942484, 0.        , 0.01998635, 0.        ,
       0.03497968, 0.03484307, 0.08054172, 0.01000152, 0.01985598,
       0.02020458, 0.03969428, 0.03986638, 0.03989313, 0.03971839,
       0.04026789, 0.049503  , 0.20084204, 0.20115798, 0.        ,
       0.02501545, 0.01994193, 0.07010474, 0.59578905, 0.06944107,
       0.10030518, 0.04018512],[0.06000992, 0.03970036, 0.08084039, 0.        , 0.02011992,
       0.09988033, 0.09934479, 0.        , 0.01990171, 0.        ,
       0.03519713, 0.03500055, 0.07987791, 0.00992824, 0.02024041,
       0.01987026, 0.03987528, 0.03985044, 0.03973383, 0.04035403,
       0.03985107, 0.04980316, 0.19887195, 0.19864103, 0.        ,
       0.02482167, 0.02000938, 0.06986264, 0.59123323, 0.07018267,
       0.0998175 , 0.0404901 ],[0.0612779 , 0.0406777 , 0.08229282, 0.        , 0.02045764,
       0.1024964 , 0.10354347, 0.        , 0.02060232, 0.        ,
       0.03613583, 0.0361116 , 0.08264777, 0.01045948, 0.02047724,
       0.02065403, 0.04061243, 0.04079331, 0.04123434, 0.04089616,
       0.04100867, 0.05103938, 0.20540281, 0.20369869, 0.        ,
       0.0256938 , 0.0206963 , 0.07119681, 0.61606479, 0.07131627,
       0.10182542, 0.04154588],[0.06139981, 0.04136017, 0.08189813, 0.        , 0.02060026,
       0.10266062, 0.10138262, 0.        , 0.02042906, 0.        ,
       0.03596026, 0.0356507 , 0.08326013, 0.01033206, 0.02031045,
       0.02021615, 0.04043261, 0.04100345, 0.04063688, 0.04136679,
       0.04100621, 0.05142763, 0.20216078, 0.20592905, 0.        ,
       0.02564881, 0.02083981, 0.07233885, 0.61530857, 0.07250823,
       0.10229411, 0.04131593],[0.06077815, 0.04066999, 0.08177054, 0.        , 0.02035395,
       0.10266288, 0.10235192, 0.        , 0.02076519, 0.        ,
       0.0354163 , 0.03590363, 0.08214333, 0.01005421, 0.0205301 ,
       0.02067816, 0.04102767, 0.04117342, 0.04079879, 0.04139867,
       0.04136573, 0.0513295 , 0.20614786, 0.20570517, 0.        ,
       0.02585041, 0.02051488, 0.071565  , 0.61181544, 0.07119035,
       0.1026838 , 0.04127624],[0.06082803, 0.04109335, 0.0814098 , 0.        , 0.02043266,
       0.10304359, 0.10423463, 0.        , 0.02077435, 0.        ,
       0.03624682, 0.03568256, 0.08133079, 0.01013979, 0.02039568,
       0.02072689, 0.04082099, 0.04063305, 0.04073632, 0.04123713,
       0.04117206, 0.0510344 , 0.20795063, 0.20329911, 0.        ,
       0.02566898, 0.02061523, 0.07192823, 0.62068569, 0.07164808,
       0.10287005, 0.04129667],[0.06106293, 0.04098498, 0.08191654, 0.        , 0.02061335,
       0.10342293, 0.10174813, 0.        , 0.020435  , 0.        ,
       0.03543019, 0.03564456, 0.08290105, 0.01028756, 0.02024553,
       0.02022824, 0.04130358, 0.04106051, 0.04079449, 0.04076502,
       0.0409259 , 0.05073692, 0.20514596, 0.20521034, 0.        ,
       0.02566104, 0.02040744, 0.07134949, 0.60582126, 0.07221056,
       0.10215864, 0.04075796],[0.06033297, 0.04072442, 0.08221177, 0.        , 0.02034636,
       0.10352375, 0.10241031, 0.        , 0.02043909, 0.        ,
       0.03588356, 0.03588893, 0.08301144, 0.01014767, 0.02055627,
       0.0204344 , 0.04122258, 0.04040457, 0.04147836, 0.04083467,
       0.04147165, 0.05188456, 0.20357695, 0.20543317, 0.        ,
       0.02563755, 0.02042217, 0.07276816, 0.61416476, 0.07121725,
       0.10158168, 0.04099431],[0.06007573, 0.0401139 , 0.08051578, 0.        , 0.01996572,
       0.10097056, 0.10057194, 0.        , 0.02000424, 0.        ,
       0.03503442, 0.03478147, 0.08001434, 0.01000185, 0.02012545,
       0.01987816, 0.04001364, 0.04018637, 0.04048009, 0.04028547,
       0.03986383, 0.05034105, 0.20100152, 0.19938168, 0.        ,
       0.02469319, 0.01984124, 0.06995925, 0.59810863, 0.06911862,
       0.10048686, 0.04013121],[0.05977072, 0.03962039, 0.07973202, 0.        , 0.02019814,
       0.10062531, 0.0994902 , 0.        , 0.02017027, 0.        ,
       0.03495281, 0.03546649, 0.08001395, 0.01000437, 0.01991663,
       0.01983415, 0.03946788, 0.04018369, 0.04007597, 0.04025096,
       0.04042191, 0.05009382, 0.19931151, 0.19945316, 0.        ,
       0.02491155, 0.02008066, 0.06979733, 0.60394693, 0.06967535,
       0.09925284, 0.04049636],[0.06025247, 0.03964703, 0.08007376, 0.        , 0.02003166,
       0.10018392, 0.10107336, 0.        , 0.01967752, 0.        ,
       0.03489423, 0.03521842, 0.08003075, 0.01008986, 0.01978551,
       0.02010058, 0.03997846, 0.03994479, 0.03999897, 0.04012407,
       0.04011502, 0.05045199, 0.20104374, 0.20056121, 0.        ,
       0.0247284 , 0.01980336, 0.06915889, 0.60526262, 0.07053848,
       0.10052578, 0.0403993 ],[0.06055927, 0.03971968, 0.08010861, 0.        , 0.02012582,
       0.09936603, 0.10022266, 0.        , 0.01981371, 0.        ,
       0.03518487, 0.03524795, 0.07955507, 0.01011258, 0.02014262,
       0.02009851, 0.03940888, 0.040109  , 0.04053773, 0.04001281,
       0.04015158, 0.04925868, 0.20141868, 0.20081031, 0.        ,
       0.02523184, 0.02007633, 0.07029046, 0.59912938, 0.06938127,
       0.09974179, 0.03996718],[0.05989877, 0.0398847 , 0.07916399, 0.        , 0.02008899,
       0.09908803, 0.10028968, 0.        , 0.01993691, 0.        ,
       0.03499059, 0.03497707, 0.07911472, 0.00990751, 0.02009623,
       0.01983041, 0.04005919, 0.04007347, 0.03984997, 0.04020002,
       0.03972985, 0.04992255, 0.19884136, 0.1991647 , 0.        ,
       0.02487241, 0.01999086, 0.06949043, 0.5946083 , 0.07052053,
       0.09905987, 0.04013919],[0.05943959, 0.03992254, 0.07998986, 0.        , 0.01982264,
       0.1002794 , 0.09959901, 0.        , 0.02003158, 0.        ,
       0.03482264, 0.03541176, 0.07933452, 0.01000899, 0.01988224,
       0.01990405, 0.0399855 , 0.03959886, 0.04009049, 0.04005962,
       0.04011123, 0.05047169, 0.20006674, 0.20088938, 0.        ,
       0.02481478, 0.02004665, 0.06939585, 0.60074183, 0.0697026 ,
       0.10077475, 0.04012812]])

    _test_agent_1(env, agent, list_feng_11, list_feng_26, list_fu_he, list_fu_he_Q)
elif kk == 7:
    list_feng_11 = np.array([1.5335833329100172, 1.4576288911656483, 1.7183615283227889, 1.535821068924562, 1.5667849631181134, 1.5250333218473535, 1.681759348928865, 1.5204661686090213, 1.4663974548929262, 1.4382836079349668, 1.3272779557461083, 1.584940327325668, 1.6022192737114498, 1.5092830827563966, 1.3442081855590966, 1.5143505018455077, 1.525721239742911, 1.4937927174594736, 1.4921579767371864, 1.3817980181415352, 1.4898746388936113, 1.6006754067125073, 1.5432443319488773, 1.632726458862443])
    list_feng_26 = np.array([0.9289708868747186, 0.9491616182290339, 1.1478545976487426, 0.9305836937686047, 0.9984753500473704, 0.9335789497027727, 0.997311566692483, 1.0815879695801456, 0.9867476253952934, 1.052519692119843, 0.9105307651891201, 1.0173087828315914, 0.892933460991534, 1.0170591660608248, 1.0037569079479591, 0.928407859110413, 1.0047859270047064, 1.0141484949174309, 1.102615106213459, 1.1093024713889679, 0.8340955851902409, 1.0240710137548326, 1.0110996149865192, 0.9810587252947185])
    list_fu_he = np.array([
        [0.06140542, 0.06030083, 0.08328473, 0.        , 0.04200719,
       0.12220883, 0.12440442, 0.        , 0.03626083, 0.        ,
       0.03703883, 0.03784054, 0.07350197, 0.03891098, 0.03659729,
       0.04402626, 0.06165819, 0.06952651, 0.0600252 , 0.06938471,
       0.06336805, 0.06458694, 0.31455829, 0.29655506, 0.        ,
       0.03692649, 0.04543347, 0.07609596, 0.13675571, 0.11826543,
       0.14912741, 0.04689982],[0.07352324, 0.05929984, 0.08044125, 0.        , 0.04004766,
       0.14475446, 0.13435312, 0.        , 0.04254884, 0.        ,
       0.03983293, 0.04007135, 0.07699843, 0.04007178, 0.04521969,
       0.04157906, 0.06056407, 0.06593757, 0.05476019, 0.0583073 ,
       0.06281198, 0.06117979, 0.29426987, 0.29683403, 0.        ,
       0.03916522, 0.04011712, 0.08225759, 0.13081838, 0.11342508,
       0.14307433, 0.04706209],[0.07063177, 0.06395676, 0.08821494, 0.        , 0.04363903,
       0.14129844, 0.14153464, 0.        , 0.04273983, 0.        ,
       0.03908645, 0.04315524, 0.08128332, 0.04217719, 0.03751669,
       0.04454665, 0.0579549 , 0.06593012, 0.06340541, 0.06474088,
       0.06050923, 0.06273413, 0.28550588, 0.29312416, 0.        ,
       0.03968253, 0.04475979, 0.08503519, 0.13384443, 0.10954408,
       0.15105336, 0.04522395],[0.06860924, 0.06252217, 0.09208772, 0.        , 0.04299417,
       0.14936644, 0.14123415, 0.        , 0.04123243, 0.        ,
       0.04264165, 0.04127697, 0.07771366, 0.04262634, 0.0405682 ,
       0.0404277 , 0.06312027, 0.0662674 , 0.06140652, 0.06255974,
       0.05991664, 0.06702925, 0.29170616, 0.31424903, 0.        ,
       0.04013213, 0.04417613, 0.08017642, 0.1469912 , 0.10434505,
       0.14457488, 0.04668807],[0.06934037, 0.06341254, 0.08494749, 0.        , 0.04014513,
       0.15093245, 0.14818295, 0.        , 0.03934085, 0.        ,
       0.03811708, 0.04221366, 0.09118865, 0.04291674, 0.0408023 ,
       0.0428664 , 0.06463461, 0.0655996 , 0.06040937, 0.06370862,
       0.06369544, 0.05904544, 0.29211041, 0.28841246, 0.        ,
       0.03968099, 0.03873386, 0.07469849, 0.13276879, 0.11273138,
       0.14363827, 0.04065329],[0.06591952, 0.06460052, 0.08683315, 0.        , 0.04141104,
       0.14706604, 0.13528667, 0.        , 0.04152223, 0.        ,
       0.0419754 , 0.04041499, 0.08920807, 0.04179555, 0.04203325,
       0.04368268, 0.0577663 , 0.06556892, 0.06166496, 0.05922468,
       0.06420223, 0.06580342, 0.31699394, 0.27862757, 0.        ,
       0.04181544, 0.0423719 , 0.08423545, 0.14131702, 0.10698179,
       0.13301772, 0.04219662],[0.09367876, 0.0810673 , 0.11051225, 0.        , 0.05073236,
       0.16666404, 0.17698509, 0.        , 0.05089172, 0.        ,
       0.05268088, 0.05388504, 0.10934012, 0.05183323, 0.05307097,
       0.05349229, 0.08869687, 0.08009484, 0.07786275, 0.08490131,
       0.08148933, 0.07665418, 0.40981737, 0.38214124, 0.        ,
       0.05143241, 0.05582443, 0.10894352, 0.16664615, 0.13958882,
       0.19044346, 0.0532179 ],[0.09027758, 0.08549926, 0.11536114, 0.        , 0.05324013,
       0.18954081, 0.1779679 , 0.        , 0.05424908, 0.        ,
       0.0544627 , 0.05599236, 0.10138605, 0.05200025, 0.05382384,
       0.05437039, 0.08324319, 0.08371117, 0.083407  , 0.08247403,
       0.0874593 , 0.07617802, 0.37456361, 0.38055499, 0.        ,
       0.05306517, 0.05497648, 0.11314048, 0.17342864, 0.13948315,
       0.18397514, 0.0573159 ],[0.09035624, 0.08313177, 0.1149841 , 0.        , 0.05566829,
       0.17741505, 0.17814819, 0.        , 0.05458649, 0.        ,
       0.05649505, 0.05738768, 0.10321322, 0.05229124, 0.0583049 ,
       0.05634852, 0.0851627 , 0.07457502, 0.08437805, 0.08437858,
       0.08130481, 0.08301273, 0.37752023, 0.37614755, 0.        ,
       0.05331789, 0.05082067, 0.10721771, 0.18806036, 0.12721578,
       0.19863429, 0.0497834 ],[0.08958019, 0.08125221, 0.11060859, 0.        , 0.04968349,
       0.17943531, 0.17794382, 0.        , 0.05471147, 0.        ,
       0.051963  , 0.05385409, 0.11012422, 0.05535528, 0.05566845,
       0.0559752 , 0.08396387, 0.08190178, 0.08521747, 0.08393134,
       0.08064161, 0.08285802, 0.38136661, 0.39891537, 0.        ,
       0.05303438, 0.05051341, 0.10852524, 0.18018217, 0.1305994 ,
       0.18153837, 0.05123955],[0.0914448 , 0.07952052, 0.10953652, 0.        , 0.05769999,
       0.18073371, 0.17928184, 0.        , 0.05272081, 0.        ,
       0.05379157, 0.05537285, 0.0994521 , 0.05376687, 0.05495235,
       0.05507944, 0.08259505, 0.0826384 , 0.0844185 , 0.08351667,
       0.08815274, 0.07982231, 0.39532841, 0.36479424, 0.        ,
       0.05254898, 0.05437997, 0.10710344, 0.18869455, 0.13826629,
       0.19250854, 0.05143371],[0.08913326, 0.08046915, 0.10607889, 0.        , 0.05551767,
       0.18257699, 0.18062303, 0.        , 0.05049785, 0.        ,
       0.05421685, 0.05053471, 0.10263975, 0.0539437 , 0.05368584,
       0.05491764, 0.08398607, 0.08244389, 0.0830528 , 0.08194457,
       0.08070815, 0.0813254 , 0.37004135, 0.40967848, 0.        ,
       0.05484076, 0.0531666 , 0.1074642 , 0.1795907 , 0.1292923 ,
       0.18712706, 0.05531004],[0.10923339, 0.09537617, 0.13340969, 0.        , 0.06473318,
       0.22006611, 0.21688513, 0.        , 0.06868857, 0.        ,
       0.06562641, 0.06587622, 0.12979403, 0.06553228, 0.06740596,
       0.06633572, 0.09852632, 0.10195375, 0.1018283 , 0.09928921,
       0.09727998, 0.09495905, 0.44895696, 0.46650671, 0.        ,
       0.06587659, 0.06490361, 0.12288172, 0.21815533, 0.17143012,
       0.23116067, 0.06482608],[0.10907941, 0.09578279, 0.13239412, 0.        , 0.06431225,
       0.20541177, 0.20812647, 0.        , 0.0678065 , 0.        ,
       0.06601099, 0.06199656, 0.12481209, 0.06666601, 0.06596867,
       0.06703394, 0.10066285, 0.09982276, 0.0987174 , 0.10112144,
       0.0999321 , 0.09475784, 0.48258371, 0.43967482, 0.        ,
       0.06749854, 0.06689318, 0.13232167, 0.2211706 , 0.15619433,
       0.23922234, 0.06690506],[0.10823993, 0.09868657, 0.134881  , 0.        , 0.06680808,
       0.23083692, 0.21061301, 0.        , 0.06405995, 0.        ,
       0.06663371, 0.06643096, 0.1342097 , 0.06277002, 0.06845783,
       0.0674123 , 0.09474738, 0.09903463, 0.10061298, 0.09898053,
       0.09549129, 0.10192039, 0.45575099, 0.44825734, 0.        ,
       0.06501316, 0.06807284, 0.13665995, 0.22149886, 0.16095594,
       0.23647275, 0.06701473],[0.11087699, 0.10352432, 0.12809128, 0.        , 0.0646857 ,
       0.21528478, 0.21256985, 0.        , 0.06544955, 0.        ,
       0.06812149, 0.06297054, 0.1324036 , 0.06494964, 0.07025412,
       0.06755732, 0.09759683, 0.10654364, 0.09487297, 0.10175513,
       0.10513549, 0.10270166, 0.45789626, 0.47175069, 0.        ,
       0.06280995, 0.06167033, 0.13260622, 0.22222793, 0.17355213,
       0.21902774, 0.06772517],[0.10764327, 0.10041474, 0.13317513, 0.        , 0.06744131,
       0.22781458, 0.21507432, 0.        , 0.06479474, 0.        ,
       0.06793855, 0.06601381, 0.13222506, 0.06756131, 0.06382145,
       0.06824616, 0.10501338, 0.0972463 , 0.1017638 , 0.09434468,
       0.09196578, 0.10186541, 0.47203792, 0.46763973, 0.        ,
       0.06699032, 0.06767493, 0.12782994, 0.22616092, 0.16851626,
       0.22116584, 0.06443312],[0.10907319, 0.10105107, 0.12936519, 0.        , 0.07004879,
       0.22355246, 0.23104466, 0.        , 0.06534618, 0.        ,
       0.06706149, 0.06609673, 0.13037684, 0.06674949, 0.0644219 ,
       0.0652057 , 0.10021842, 0.09625193, 0.09784265, 0.0955341 ,
       0.10117196, 0.10473099, 0.43400751, 0.44917884, 0.        ,
       0.06797919, 0.06434727, 0.12850798, 0.21006335, 0.16759447,
       0.22559875, 0.0659093 ],[0.09127045, 0.0816713 , 0.10460128, 0.        , 0.05738713,
       0.18868416, 0.18135947, 0.        , 0.05533771, 0.        ,
       0.05698864, 0.05128812, 0.10999459, 0.05172711, 0.05647191,
       0.05288501, 0.07813446, 0.08447207, 0.08582947, 0.08419109,
       0.07574367, 0.08405064, 0.40431012, 0.38376732, 0.        ,
       0.05284522, 0.05593235, 0.11077315, 0.18617406, 0.12942146,
       0.18193767, 0.05311706],[0.09154323, 0.07904916, 0.10796687, 0.        , 0.05416132,
       0.16545782, 0.1740562 , 0.        , 0.05651893, 0.        ,
       0.05070931, 0.05112658, 0.10010093, 0.0585527 , 0.05298583,
       0.05277836, 0.08199797, 0.08248623, 0.07861086, 0.08337854,
       0.08247348, 0.08232177, 0.37159658, 0.37656779, 0.        ,
       0.05195168, 0.05507176, 0.10663119, 0.17530872, 0.13372241,
       0.19397923, 0.05316323],[0.09092204, 0.07565438, 0.10764283, 0.        , 0.05232015,
       0.17817682, 0.17369532, 0.        , 0.05493126, 0.        ,
       0.05337406, 0.05571837, 0.11434687, 0.05037376, 0.0508297 ,
       0.05390081, 0.08074537, 0.08357087, 0.08112392, 0.08557736,
       0.07854002, 0.08510978, 0.38229888, 0.38694106, 0.        ,
       0.05068413, 0.05107051, 0.11202531, 0.18165817, 0.13244743,
       0.18503406, 0.05675694],[0.09127869, 0.08157582, 0.10759699, 0.        , 0.0531985 ,
       0.17931341, 0.17823143, 0.        , 0.05573324, 0.        ,
       0.05140462, 0.04926846, 0.1074433 , 0.05828316, 0.05505154,
       0.05189656, 0.08427005, 0.08185185, 0.0811689 , 0.08043369,
       0.08171419, 0.08612059, 0.37644586, 0.39185845, 0.        ,
       0.05632334, 0.05135051, 0.10764495, 0.1844444 , 0.13973952,
       0.19656182, 0.05375067],[0.08667077, 0.08193928, 0.10306377, 0.        , 0.05720724,
       0.18967185, 0.16981008, 0.        , 0.05549741, 0.        ,
       0.05535437, 0.05422928, 0.11463183, 0.05190293, 0.05688447,
       0.05655357, 0.08166624, 0.08254402, 0.07929057, 0.08445379,
       0.07986056, 0.08600709, 0.38744916, 0.37189319, 0.        ,
       0.05526064, 0.05363921, 0.10977783, 0.1843568 , 0.13766617,
       0.19595933, 0.05408613],[0.0924198 , 0.07863885, 0.11443106, 0.        , 0.05668955,
       0.18221529, 0.17449186, 0.        , 0.05650739, 0.        ,
       0.05333575, 0.05377352, 0.11012986, 0.0568528 , 0.0554207 ,
       0.05579433, 0.07518715, 0.08374855, 0.08155925, 0.08009606,
       0.08543683, 0.07941722, 0.38551849, 0.36562226, 0.        ,
       0.05274702, 0.05397442, 0.10666637, 0.18744583, 0.13840915,
       0.19767155, 0.05127183]])
    list_fu_he_Q = np.array([
        [0.05770079, 0.0386376 , 0.07961815, 0.        , 0.01933042,
       0.09913875, 0.09938865, 0.        , 0.01972525, 0.        ,
       0.03490905, 0.03341251, 0.07729607, 0.00981184, 0.01917516,
       0.0196423 , 0.03955656, 0.0392095 , 0.03935301, 0.03931102,
       0.03830224, 0.0484106 , 0.19302322, 0.19540345, 0.        ,
       0.02401215, 0.0192319 , 0.06700636, 0.57434532, 0.06686607,
       0.09879598, 0.03972427],[0.05848438, 0.03898159, 0.07859603, 0.        , 0.01985785,
       0.0980013 , 0.09890605, 0.        , 0.01935312, 0.        ,
       0.03444088, 0.03488127, 0.07869296, 0.00978149, 0.01960914,
       0.01926402, 0.03850584, 0.0391627 , 0.03852938, 0.03950699,
       0.03916972, 0.04943783, 0.19691531, 0.19588019, 0.        ,
       0.02390708, 0.01925643, 0.06832626, 0.57952118, 0.06862438,
       0.09582045, 0.03950284],[0.05829818, 0.03847009, 0.07746006, 0.        , 0.01964001,
       0.09827333, 0.09807977, 0.        , 0.01939033, 0.        ,
       0.0339472 , 0.03417625, 0.07728414, 0.0098183 , 0.01944063,
       0.01987277, 0.03900441, 0.0391075 , 0.03852023, 0.03865319,
       0.03984016, 0.04935005, 0.19160922, 0.1942226 , 0.        ,
       0.02458334, 0.0193358 , 0.06872649, 0.58357482, 0.06874451,
       0.09722666, 0.03939107],[0.05874399, 0.03922113, 0.07823666, 0.        , 0.01970029,
       0.09826516, 0.09597712, 0.        , 0.01955988, 0.        ,
       0.03375917, 0.03458403, 0.07836939, 0.00952947, 0.01947598,
       0.0196584 , 0.03948668, 0.0385934 , 0.03958148, 0.03884635,
       0.03919777, 0.04901018, 0.19375475, 0.19481178, 0.        ,
       0.02473581, 0.01925209, 0.06837991, 0.5858347 , 0.06784064,
       0.09614146, 0.03947643],[0.058318  , 0.03902235, 0.07760583, 0.        , 0.01948968,
       0.09638865, 0.09743047, 0.        , 0.01916682, 0.        ,
       0.03429976, 0.03425244, 0.07752914, 0.00972594, 0.01956009,
       0.01940553, 0.03904315, 0.03899572, 0.03879841, 0.03913121,
       0.03891029, 0.04880008, 0.19687235, 0.19410254, 0.        ,
       0.02435064, 0.01933377, 0.06790087, 0.59298038, 0.06756922,
       0.09828065, 0.03895104],[0.05886565, 0.0389069 , 0.0766755 , 0.        , 0.01953742,
       0.09673043, 0.09711298, 0.        , 0.01926194, 0.        ,
       0.03394561, 0.03443962, 0.07836002, 0.00966125, 0.01936791,
       0.01963784, 0.03873513, 0.03893895, 0.03843338, 0.0392662 ,
       0.03913756, 0.04804266, 0.19682542, 0.19759952, 0.        ,
       0.02410058, 0.01963695, 0.06924583, 0.58106258, 0.0686588 ,
       0.09855094, 0.03910485],[0.0608656 , 0.03966033, 0.08090541, 0.        , 0.01987009,
       0.10030399, 0.0995117 , 0.        , 0.01985364, 0.        ,
       0.03473962, 0.03511064, 0.07979558, 0.01002998, 0.02017722,
       0.01994176, 0.0404811 , 0.04017716, 0.040338  , 0.03996269,
       0.04021369, 0.04973288, 0.2001659 , 0.19830368, 0.        ,
       0.02477172, 0.02002664, 0.07031528, 0.59789611, 0.06963597,
       0.09934085, 0.03992632],[0.06029589, 0.04068722, 0.08010266, 0.        , 0.02009787,
       0.09945995, 0.10115829, 0.        , 0.02017145, 0.        ,
       0.0348024 , 0.03516542, 0.07962351, 0.00998867, 0.01995003,
       0.01986557, 0.04009369, 0.03993039, 0.04015194, 0.0403701 ,
       0.03953811, 0.04960192, 0.19900417, 0.19869629, 0.        ,
       0.0253159 , 0.01987264, 0.07035339, 0.59760547, 0.06966363,
       0.09873546, 0.04002927],[0.0600634 , 0.04001799, 0.07958901, 0.        , 0.02007881,
       0.10001329, 0.09972057, 0.        , 0.02032643, 0.        ,
       0.03468215, 0.03481382, 0.0807846 , 0.0100519 , 0.0198155 ,
       0.02000504, 0.03970612, 0.03951536, 0.0395835 , 0.03996101,
       0.04016831, 0.05023153, 0.19859961, 0.19851911, 0.        ,
       0.02466783, 0.01998642, 0.07041497, 0.60410121, 0.07020132,
       0.09999327, 0.03985503],[0.05896064, 0.04024598, 0.08030843, 0.        , 0.01987045,
       0.09895213, 0.10025976, 0.        , 0.02016825, 0.        ,
       0.03506624, 0.03498663, 0.08007428, 0.00991895, 0.0201425 ,
       0.02013357, 0.04007537, 0.04039443, 0.03996501, 0.03990854,
       0.04021619, 0.04979957, 0.19988303, 0.20160606, 0.        ,
       0.02515903, 0.02012977, 0.07039142, 0.60155362, 0.06959568,
       0.09915465, 0.04024997],[0.06021915, 0.04011702, 0.08037514, 0.        , 0.01992728,
       0.0988639 , 0.09921634, 0.        , 0.02002864, 0.        ,
       0.03544273, 0.03546554, 0.07987426, 0.01003552, 0.02021856,
       0.01978688, 0.0398567 , 0.04010813, 0.04013247, 0.03983375,
       0.03983858, 0.05011868, 0.20144738, 0.19806967, 0.        ,
       0.02484614, 0.019991  , 0.06962315, 0.60154816, 0.07031208,
       0.09985208, 0.04006276],[0.06038475, 0.03999836, 0.07979986, 0.        , 0.02011159,
       0.09999937, 0.09992791, 0.        , 0.02006239, 0.        ,
       0.03507181, 0.03518454, 0.08136509, 0.01011261, 0.02010283,
       0.02016545, 0.04039043, 0.04001246, 0.04010747, 0.03997207,
       0.04028289, 0.05058563, 0.19957856, 0.19947284, 0.        ,
       0.02481279, 0.02018493, 0.06930876, 0.59884659, 0.0699506 ,
       0.09937578, 0.0400024 ],[0.06222354, 0.04091692, 0.0816209 , 0.        , 0.02057058,
       0.10296091, 0.10286608, 0.        , 0.02052401, 0.        ,
       0.03578345, 0.03662941, 0.08233184, 0.01020946, 0.0205116 ,
       0.02056892, 0.04066913, 0.04097692, 0.04119826, 0.04077966,
       0.04074377, 0.05197798, 0.2055091 , 0.20318766, 0.        ,
       0.02576978, 0.0204313 , 0.07155552, 0.61394804, 0.07084734,
       0.10312675, 0.04126052],[0.06201517, 0.04058577, 0.08075682, 0.        , 0.02026017,
       0.10165258, 0.10240297, 0.        , 0.02058013, 0.        ,
       0.03618006, 0.03587888, 0.08268495, 0.01013742, 0.0208525 ,
       0.02036559, 0.04114051, 0.04079494, 0.04109512, 0.04099457,
       0.04184197, 0.05158121, 0.20525352, 0.20560248, 0.        ,
       0.02520113, 0.02067829, 0.07137399, 0.61347163, 0.07203706,
       0.10274198, 0.04152802],[0.06078975, 0.04119077, 0.08159175, 0.        , 0.02063673,
       0.1017437 , 0.10175822, 0.        , 0.02038166, 0.        ,
       0.03640356, 0.03608235, 0.08238764, 0.01011247, 0.02045547,
       0.02042732, 0.04106213, 0.04114935, 0.04151961, 0.041168  ,
       0.04086287, 0.05147492, 0.202735  , 0.20450661, 0.        ,
       0.0255721 , 0.02030702, 0.07083815, 0.62218039, 0.07245517,
       0.10082865, 0.04087837],[0.06099358, 0.0412132 , 0.08186866, 0.        , 0.02074818,
       0.10281744, 0.10234495, 0.        , 0.02022762, 0.        ,
       0.03580787, 0.03558935, 0.082016  , 0.01019462, 0.02043115,
       0.02034914, 0.04087712, 0.04031102, 0.04085752, 0.04051268,
       0.04087795, 0.05130018, 0.20276598, 0.2037501 , 0.        ,
       0.02564528, 0.02054468, 0.0713807 , 0.62171696, 0.07124159,
       0.10270968, 0.04069057],[0.06192304, 0.04139127, 0.08262468, 0.        , 0.02041734,
       0.10243371, 0.10386463, 0.        , 0.0205728 , 0.        ,
       0.03592729, 0.03580876, 0.08342463, 0.01025814, 0.02081031,
       0.02060989, 0.04149439, 0.04125961, 0.04105277, 0.04051966,
       0.04113254, 0.05107296, 0.20539132, 0.20563656, 0.        ,
       0.02551033, 0.02051572, 0.07280273, 0.61868941, 0.07109291,
       0.1030191 , 0.04086253],[0.06201471, 0.04067612, 0.08265214, 0.        , 0.02027471,
       0.10405863, 0.10250803, 0.        , 0.02051623, 0.        ,
       0.03621187, 0.03597229, 0.08202755, 0.01026447, 0.02044953,
       0.02049738, 0.04107431, 0.04047083, 0.04074388, 0.04139186,
       0.04086406, 0.05086862, 0.20458477, 0.20373212, 0.        ,
       0.02571591, 0.02055992, 0.07299199, 0.61960855, 0.07136336,
       0.10162976, 0.04085091],[0.05972538, 0.0394386 , 0.08081794, 0.        , 0.01989645,
       0.1000223 , 0.10028318, 0.        , 0.01993234, 0.        ,
       0.03472049, 0.03489309, 0.07949926, 0.01009034, 0.019901  ,
       0.02023616, 0.04033165, 0.04053056, 0.03962958, 0.03989638,
       0.0400195 , 0.05036184, 0.19831214, 0.1995455 , 0.        ,
       0.02501591, 0.01982965, 0.07000395, 0.60031139, 0.06957489,
       0.0995668 , 0.03998168],[0.05972915, 0.03985558, 0.08062527, 0.        , 0.02003516,
       0.09995535, 0.09979704, 0.        , 0.02008319, 0.        ,
       0.03514357, 0.03531967, 0.07998435, 0.00990692, 0.02013329,
       0.01994095, 0.04029312, 0.03988288, 0.04002439, 0.03978506,
       0.0399835 , 0.05008922, 0.20051922, 0.19908716, 0.        ,
       0.02488365, 0.02032598, 0.0700877 , 0.59513007, 0.07039759,
       0.10011836, 0.03949281],[0.05999748, 0.04016946, 0.07969745, 0.        , 0.02019384,
       0.10129995, 0.10035721, 0.        , 0.01977857, 0.        ,
       0.03487449, 0.03474383, 0.07998833, 0.0100136 , 0.02005092,
       0.01993377, 0.03995803, 0.03980372, 0.03975913, 0.04011175,
       0.03981855, 0.05023411, 0.19978804, 0.20112883, 0.        ,
       0.02484675, 0.01987838, 0.06984151, 0.60132547, 0.07050692,
       0.09993954, 0.0399612 ],[0.05949098, 0.03988287, 0.07948293, 0.        , 0.02001315,
       0.09961569, 0.10008778, 0.        , 0.01992467, 0.        ,
       0.03501144, 0.03476444, 0.07933506, 0.00996116, 0.02010762,
       0.02008699, 0.04033592, 0.03990518, 0.03947744, 0.04047166,
       0.03988219, 0.05065344, 0.20049085, 0.20235379, 0.        ,
       0.02484392, 0.01989167, 0.06942342, 0.60631541, 0.07014263,
       0.09915386, 0.03991272],[0.06008808, 0.04006284, 0.07961071, 0.        , 0.01987398,
       0.09968643, 0.09972663, 0.        , 0.02003295, 0.        ,
       0.03524769, 0.03502057, 0.08001604, 0.01001524, 0.01999922,
       0.02012172, 0.03966675, 0.04004884, 0.0398898 , 0.0401588 ,
       0.03944081, 0.05015424, 0.2003047 , 0.19809868, 0.        ,
       0.02511128, 0.01980985, 0.0703013 , 0.59765479, 0.06941047,
       0.09985   , 0.03974034],[0.06041843, 0.04057528, 0.07936331, 0.        , 0.01978199,
       0.09962345, 0.1005368 , 0.        , 0.01997856, 0.        ,
       0.03486125, 0.03490559, 0.07896369, 0.00996025, 0.0200351 ,
       0.02017949, 0.04025389, 0.03981501, 0.04000833, 0.03971658,
       0.03942958, 0.05039121, 0.20024023, 0.2000958 , 0.        ,
       0.02503618, 0.02006035, 0.07074941, 0.59622721, 0.06952723,
       0.10061015, 0.04024926]])

    _test_agent_1(env, agent, list_feng_11, list_feng_26, list_fu_he, list_fu_he_Q)
elif kk == 8:
    list_feng_11 = np.array([1.5488557403480891, 1.57933150302968, 1.4376752962718076, 1.3782044556260824, 1.3785954370233169, 1.6247565228619205, 1.5329106719803562, 1.4614476278661634, 1.5902631311856394, 1.5051846156738555, 1.3561255277869584, 1.5273172731865547, 1.549653047085945, 1.3992211904826553, 1.527818377808626, 1.5059427209158147, 1.4657599931909793, 1.6306285658196376, 1.5060140454993387, 1.345600574362991, 1.4012052878228405, 1.5015198321502223, 1.405424562405938, 1.5400429769507777])
    list_feng_26 = np.array([0.8617383005595454, 0.9457455529951362, 1.0367509564932418, 1.106702245474253, 0.938692374168024, 1.0491854548607482, 0.9979799924886213, 1.03819634594521, 0.901377099700603, 0.9978369050992482, 1.0464097289756173, 0.966504588259971, 0.927818499202405, 0.9787144853366994, 0.9806921721588975, 1.040610748910693, 0.9515715921501662, 0.9638858013900542, 1.029122064288517, 1.052807046393738, 1.0661313346267278, 1.0661932208248572, 1.0451661634225418, 1.0317794763698733])
    list_fu_he = np.array([
        [0.06757174, 0.06033339, 0.08936129, 0.        , 0.03857772,
       0.15182122, 0.15487913, 0.        , 0.03625231, 0.        ,
       0.04728422, 0.04620733, 0.08674676, 0.04422626, 0.04099308,
       0.04208565, 0.06079624, 0.06042622, 0.05541277, 0.05724377,
       0.05501083, 0.05796531, 0.26958187, 0.30195851, 0.        ,
       0.04132546, 0.04189196, 0.09383762, 0.13472479, 0.09105503,
       0.14993261, 0.04047599],[0.06956789, 0.05675208, 0.0901542 , 0.        , 0.04174008,
       0.13801349, 0.13790034, 0.        , 0.04658358, 0.        ,
       0.04215934, 0.04545437, 0.08298302, 0.04242918, 0.04403568,
       0.04402915, 0.06834909, 0.0664883 , 0.05603626, 0.06482695,
       0.06276554, 0.05460001, 0.2948129 , 0.27525442, 0.        ,
       0.04032016, 0.04424505, 0.07513097, 0.14801113, 0.1139033 ,
       0.15939822, 0.04189668],[0.06907261, 0.0621973 , 0.08454598, 0.        , 0.0419628 ,
       0.13916734, 0.1370552 , 0.        , 0.0399906 , 0.        ,
       0.04521212, 0.04045715, 0.08978178, 0.04369408, 0.03976366,
       0.04300851, 0.06328124, 0.06234736, 0.06078405, 0.06679141,
       0.06394106, 0.06394599, 0.30774229, 0.3193906 , 0.        ,
       0.04502327, 0.04280576, 0.08291428, 0.14057098, 0.11088438,
       0.14846067, 0.04241545],[0.06966703, 0.06562549, 0.08880933, 0.        , 0.04452225,
       0.15463846, 0.14508879, 0.        , 0.04225218, 0.        ,
       0.03865822, 0.04484924, 0.08158678, 0.04196252, 0.04164963,
       0.04256799, 0.06259093, 0.06358923, 0.06581662, 0.06429791,
       0.06277874, 0.06101251, 0.27937567, 0.27446087, 0.        ,
       0.0413951 , 0.04228841, 0.08191035, 0.1374881 , 0.10272981,
       0.14674024, 0.03991618],[0.07493233, 0.06758096, 0.08075695, 0.        , 0.04459879,
       0.13577826, 0.13713873, 0.        , 0.04311501, 0.        ,
       0.0403594 , 0.04103175, 0.07900295, 0.03968826, 0.0420751 ,
       0.04307319, 0.06349338, 0.06480531, 0.06678276, 0.06707069,
       0.05915577, 0.06694373, 0.28717927, 0.32895458, 0.        ,
       0.04083959, 0.04306526, 0.07883559, 0.14021912, 0.10280616,
       0.14416962, 0.04377987],[0.07501435, 0.06378503, 0.08166021, 0.        , 0.0393607 ,
       0.13329048, 0.13903686, 0.        , 0.04095604, 0.        ,
       0.03946726, 0.04079669, 0.08609961, 0.04304468, 0.04480849,
       0.03780532, 0.05670901, 0.06650503, 0.06163094, 0.06264077,
       0.05662834, 0.06421337, 0.29358407, 0.31466221, 0.        ,
       0.04178676, 0.04648072, 0.08707639, 0.13217525, 0.10600713,
       0.15023172, 0.04521939],[0.09278192, 0.07699878, 0.10749182, 0.        , 0.0550282 ,
       0.18755421, 0.17949877, 0.        , 0.05221276, 0.        ,
       0.0534675 , 0.04943112, 0.10328698, 0.05665029, 0.05504957,
       0.05305849, 0.08123788, 0.08110318, 0.08249747, 0.07810613,
       0.08213313, 0.08487781, 0.38568684, 0.37077937, 0.        ,
       0.05779498, 0.05388657, 0.10465364, 0.18875032, 0.13393736,
       0.17148784, 0.05503308],[0.0969507 , 0.08333274, 0.10351476, 0.        , 0.05728626,
       0.17926223, 0.18563602, 0.        , 0.05778072, 0.        ,
       0.05379969, 0.05243787, 0.11323712, 0.054511  , 0.05279851,
       0.05300865, 0.08472389, 0.07870095, 0.08362241, 0.0796442 ,
       0.07847775, 0.08209215, 0.35483471, 0.36203245, 0.        ,
       0.05251206, 0.05453658, 0.11274516, 0.17172702, 0.13564818,
       0.1901516 , 0.05181514],[0.09118317, 0.08291944, 0.10436833, 0.        , 0.05696286,
       0.18658394, 0.17544054, 0.        , 0.0518693 , 0.        ,
       0.05441134, 0.05238432, 0.10225604, 0.05452103, 0.05113372,
       0.05314594, 0.07891455, 0.08061852, 0.08222339, 0.08047715,
       0.07787565, 0.07890443, 0.3755623 , 0.3560352 , 0.        ,
       0.05274882, 0.05417644, 0.10704513, 0.18614303, 0.13104032,
       0.19731033, 0.05477022],[0.08844594, 0.08214106, 0.10747292, 0.        , 0.05598688,
       0.17307709, 0.19311818, 0.        , 0.05518302, 0.        ,
       0.05615797, 0.05573573, 0.10784884, 0.05478612, 0.05343074,
       0.05494396, 0.07662828, 0.0834605 , 0.08055631, 0.07995952,
       0.07598415, 0.08509948, 0.3931197 , 0.38598987, 0.        ,
       0.05355753, 0.05695133, 0.10830375, 0.17631417, 0.14584735,
       0.18042856, 0.05392813],[0.08955648, 0.07831094, 0.10847091, 0.        , 0.05477364,
       0.17810926, 0.18099517, 0.        , 0.05188892, 0.        ,
       0.05170541, 0.05370331, 0.11366268, 0.05587421, 0.05646539,
       0.05405044, 0.08145797, 0.07700414, 0.07758429, 0.08403878,
       0.08063658, 0.07955075, 0.38073252, 0.38200482, 0.        ,
       0.05708289, 0.05768641, 0.10264059, 0.18977503, 0.12149714,
       0.18522704, 0.05333817],[0.08343076, 0.07745523, 0.10844848, 0.        , 0.05270137,
       0.16933749, 0.17371694, 0.        , 0.05761643, 0.        ,
       0.057374  , 0.05520785, 0.11375406, 0.05440569, 0.05322603,
       0.05141649, 0.0813526 , 0.08474692, 0.08443702, 0.08188665,
       0.07945659, 0.08157371, 0.38175931, 0.358545  , 0.        ,
       0.05494852, 0.05194693, 0.11088567, 0.18936813, 0.14498187,
       0.19196286, 0.05559717],[0.10671496, 0.0959853 , 0.13780042, 0.        , 0.0669237 ,
       0.22731092, 0.22255833, 0.        , 0.06319048, 0.        ,
       0.07094318, 0.06773392, 0.13070589, 0.06712342, 0.06662752,
       0.06752689, 0.09757852, 0.09993903, 0.10013379, 0.09445521,
       0.09723745, 0.10017227, 0.47563223, 0.46779647, 0.        ,
       0.0631512 , 0.06685753, 0.12744396, 0.22059713, 0.15817309,
       0.2377982 , 0.06302993],[0.10943044, 0.0970575 , 0.13889783, 0.        , 0.06224103,
       0.23065463, 0.21392776, 0.        , 0.06404108, 0.        ,
       0.06793076, 0.06731738, 0.13377424, 0.06559112, 0.06475974,
       0.06870112, 0.0961947 , 0.10231766, 0.09531829, 0.09705646,
       0.10360401, 0.09426167, 0.45036944, 0.45556935, 0.        ,
       0.0666262 , 0.06487508, 0.12410112, 0.21155288, 0.15775707,
       0.2321127 , 0.06830645],[0.11220955, 0.09673526, 0.13036862, 0.        , 0.06496487,
       0.22594744, 0.23027012, 0.        , 0.06521351, 0.        ,
       0.06575859, 0.06831404, 0.12632274, 0.06484804, 0.06149757,
       0.06889012, 0.10021222, 0.10006153, 0.09925015, 0.10197394,
       0.09438854, 0.10055338, 0.46857213, 0.47059647, 0.        ,
       0.06500387, 0.06660674, 0.13079638, 0.20772349, 0.16355019,
       0.23181229, 0.06549065],[0.10770607, 0.09925982, 0.12823606, 0.        , 0.06461556,
       0.2300326 , 0.22098647, 0.        , 0.0640472 , 0.        ,
       0.06393368, 0.06803221, 0.13215425, 0.06761543, 0.06820951,
       0.06638911, 0.10010565, 0.10059058, 0.1012365 , 0.10174909,
       0.09870328, 0.09738146, 0.48208529, 0.47628994, 0.        ,
       0.06674094, 0.06658436, 0.13862804, 0.22428634, 0.15953585,
       0.24466332, 0.06590852],[0.11160027, 0.09766037, 0.12793861, 0.        , 0.06638076,
       0.22367475, 0.23106616, 0.        , 0.06640305, 0.        ,
       0.06595084, 0.06245172, 0.13578767, 0.06837748, 0.06926979,
       0.06394077, 0.09397639, 0.09521645, 0.09856926, 0.10069987,
       0.09978849, 0.10543171, 0.45469126, 0.4578849 , 0.        ,
       0.06812588, 0.06817931, 0.1320458 , 0.22689403, 0.16611531,
       0.23385788, 0.06285237],[0.10909391, 0.09817676, 0.13624565, 0.        , 0.06245548,
       0.23031349, 0.2206948 , 0.        , 0.06953055, 0.        ,
       0.06380391, 0.06649719, 0.13091194, 0.06590901, 0.06452073,
       0.06300712, 0.10399023, 0.09904224, 0.09687359, 0.09903481,
       0.1020471 , 0.10142784, 0.45663051, 0.48577536, 0.        ,
       0.06257257, 0.06535739, 0.13644732, 0.21801125, 0.17052228,
       0.23203071, 0.06331283],[0.09020259, 0.07553217, 0.11155295, 0.        , 0.05000135,
       0.17732548, 0.1853446 , 0.        , 0.05282017, 0.        ,
       0.05577341, 0.05308034, 0.11469608, 0.05454462, 0.05274975,
       0.05592317, 0.08289681, 0.07731169, 0.08553696, 0.08008283,
       0.08520473, 0.07853393, 0.34850143, 0.39925871, 0.        ,
       0.05464441, 0.0551924 , 0.10390417, 0.17437125, 0.13326652,
       0.18466483, 0.05469181],[0.09084686, 0.08070224, 0.10678073, 0.        , 0.05381055,
       0.17927172, 0.17952241, 0.        , 0.05171964, 0.        ,
       0.05597272, 0.05233249, 0.11144603, 0.05359232, 0.05188783,
       0.05319029, 0.07740421, 0.08464986, 0.0839551 , 0.08090585,
       0.08329161, 0.078787  , 0.35969967, 0.37929727, 0.        ,
       0.05524915, 0.05786501, 0.11303957, 0.1714288 , 0.13397148,
       0.18963659, 0.05436186],[0.08699544, 0.08046747, 0.11087805, 0.        , 0.0550244 ,
       0.1788161 , 0.18022561, 0.        , 0.05596173, 0.        ,
       0.05798971, 0.052038  , 0.10481043, 0.05453398, 0.05366578,
       0.05253314, 0.07893228, 0.07820941, 0.07881569, 0.07412885,
       0.08181459, 0.07705343, 0.38800251, 0.36726225, 0.        ,
       0.05579233, 0.05299995, 0.10632582, 0.16602739, 0.13547357,
       0.18061903, 0.0556446 ],[0.08749787, 0.08671429, 0.1056432 , 0.        , 0.05175245,
       0.18299391, 0.18871882, 0.        , 0.05312222, 0.        ,
       0.05672912, 0.05356191, 0.10674614, 0.05507739, 0.0530159 ,
       0.05453711, 0.080619  , 0.08109722, 0.08030592, 0.08171614,
       0.08155913, 0.08328699, 0.38924792, 0.37502109, 0.        ,
       0.0571069 , 0.05258474, 0.10881597, 0.17116938, 0.13472745,
       0.17798672, 0.0552977 ],[0.0858429 , 0.07930674, 0.11050837, 0.        , 0.05388165,
       0.1732324 , 0.18660535, 0.        , 0.05242767, 0.        ,
       0.05288853, 0.0529571 , 0.11607957, 0.05075408, 0.05547202,
       0.05715911, 0.08467603, 0.08089869, 0.08118358, 0.07699122,
       0.0736926 , 0.0834888 , 0.38416839, 0.39183298, 0.        ,
       0.05201438, 0.05386261, 0.10834212, 0.17494801, 0.13976826,
       0.18268574, 0.05467167],[0.09012328, 0.08645018, 0.10865746, 0.        , 0.05579366,
       0.18554829, 0.17992158, 0.        , 0.05754516, 0.        ,
       0.056224  , 0.05250876, 0.11114428, 0.05242163, 0.05567794,
       0.0528594 , 0.08006801, 0.08212701, 0.08268951, 0.07871566,
       0.08424759, 0.08430931, 0.3849133 , 0.35998589, 0.        ,
       0.05668196, 0.05587842, 0.10568254, 0.17907176, 0.13430565,
       0.19483175, 0.05590417]])
    list_fu_he_Q = np.array([
        [0.05815543, 0.03997142, 0.07953437, 0.        , 0.0196609 ,
       0.09997066, 0.09532699, 0.        , 0.0190211 , 0.        ,
       0.03473589, 0.03429392, 0.07796641, 0.00956231, 0.01991547,
       0.01907436, 0.03955739, 0.03860482, 0.03921167, 0.03972829,
       0.03951563, 0.04802036, 0.19742875, 0.19219516, 0.        ,
       0.02498201, 0.01969144, 0.06763701, 0.591897  , 0.06677726,
       0.09972586, 0.03859514],[0.05930265, 0.03858883, 0.0775231 , 0.        , 0.01931537,
       0.09745915, 0.09602879, 0.        , 0.01959687, 0.        ,
       0.03439218, 0.03465593, 0.07768846, 0.00981213, 0.01972508,
       0.01976106, 0.03921865, 0.03831871, 0.03836623, 0.03946204,
       0.03856003, 0.04943019, 0.19578557, 0.19104866, 0.        ,
       0.02433404, 0.01935178, 0.06913887, 0.58023064, 0.06817568,
       0.0956623 , 0.03972394],[0.05840657, 0.03918369, 0.07776224, 0.        , 0.01937738,
       0.09874557, 0.09686096, 0.        , 0.01937646, 0.        ,
       0.03438344, 0.03377455, 0.07791386, 0.00962837, 0.01935305,
       0.01947433, 0.03935183, 0.03892839, 0.03861754, 0.03842695,
       0.03936615, 0.04801998, 0.19791138, 0.1933055 , 0.        ,
       0.02444414, 0.01976884, 0.06917358, 0.58225337, 0.06926081,
       0.09620027, 0.03959792],[0.0581155 , 0.03929071, 0.07893978, 0.        , 0.01932896,
       0.09865508, 0.09703243, 0.        , 0.01941851, 0.        ,
       0.0347277 , 0.03395264, 0.07818885, 0.00968539, 0.01945137,
       0.01952995, 0.03891677, 0.0388873 , 0.03905609, 0.0391524 ,
       0.03918775, 0.04968913, 0.19633403, 0.1932508 , 0.        ,
       0.02394265, 0.01956191, 0.06861595, 0.58872722, 0.06797861,
       0.09856852, 0.03873111],[0.0590191 , 0.03929488, 0.07744694, 0.        , 0.01953704,
       0.09670287, 0.09741309, 0.        , 0.01963847, 0.        ,
       0.03423037, 0.03466055, 0.07725679, 0.0097661 , 0.01913442,
       0.01954636, 0.03925752, 0.03884817, 0.03897206, 0.03878907,
       0.0389999 , 0.04913265, 0.19409975, 0.19498546, 0.        ,
       0.02442639, 0.01949946, 0.06851984, 0.58289826, 0.06793861,
       0.09778427, 0.03927143],[0.05804466, 0.03932201, 0.07860362, 0.        , 0.01980023,
       0.09707028, 0.09874468, 0.        , 0.01931322, 0.        ,
       0.03428822, 0.03420616, 0.07813864, 0.00982787, 0.01960519,
       0.01939068, 0.03935184, 0.03949562, 0.03854057, 0.03924037,
       0.0389848 , 0.04940326, 0.19678349, 0.19667102, 0.        ,
       0.02464936, 0.01962188, 0.06846103, 0.58601762, 0.06751288,
       0.09744237, 0.03862605],[0.06032995, 0.04004524, 0.07974691, 0.        , 0.02005276,
       0.100898  , 0.0996786 , 0.        , 0.02017983, 0.        ,
       0.0349364 , 0.03502587, 0.08042014, 0.01006198, 0.01990528,
       0.02010622, 0.04061109, 0.03977937, 0.04020507, 0.04060319,
       0.03956226, 0.04983385, 0.19905679, 0.1993235 , 0.        ,
       0.02507385, 0.02010387, 0.0706269 , 0.60152396, 0.06992571,
       0.09924657, 0.03953414],[0.05962643, 0.03957384, 0.08033352, 0.        , 0.02026912,
       0.10088543, 0.0989797 , 0.        , 0.02004989, 0.        ,
       0.03483308, 0.03496296, 0.07922673, 0.00999485, 0.02006845,
       0.02011206, 0.03968554, 0.03990236, 0.03996129, 0.04009798,
       0.04004905, 0.05070667, 0.20073058, 0.20261861, 0.        ,
       0.02489281, 0.02011297, 0.06985083, 0.60437182, 0.06936555,
       0.09934957, 0.03995404],[0.05970826, 0.04011143, 0.0800528 , 0.        , 0.02029938,
       0.10063127, 0.10018173, 0.        , 0.01996257, 0.        ,
       0.03545881, 0.03524936, 0.08098072, 0.00994109, 0.01976922,
       0.020164  , 0.04014632, 0.04046522, 0.03970623, 0.04027851,
       0.03972467, 0.04960804, 0.20073972, 0.19875801, 0.        ,
       0.02515628, 0.02011903, 0.06990302, 0.60327281, 0.07022602,
       0.10034404, 0.0403518 ],[0.05977659, 0.03991607, 0.07945417, 0.        , 0.01987566,
       0.09941751, 0.09984289, 0.        , 0.01998802, 0.        ,
       0.03489167, 0.03521821, 0.07905262, 0.01004116, 0.020097  ,
       0.02006471, 0.0395984 , 0.0398575 , 0.03963578, 0.04065459,
       0.0399967 , 0.04985836, 0.19998138, 0.20040281, 0.        ,
       0.02535054, 0.0199743 , 0.07098471, 0.59551526, 0.06919875,
       0.09977005, 0.04054744],[0.06032275, 0.03980584, 0.07971964, 0.        , 0.01972679,
       0.09981687, 0.09928849, 0.        , 0.02001161, 0.        ,
       0.03517288, 0.03494498, 0.07954831, 0.00996774, 0.02002759,
       0.01965057, 0.04011048, 0.03992251, 0.04006775, 0.04030509,
       0.04039949, 0.05019232, 0.20142906, 0.19936499, 0.        ,
       0.02523773, 0.02019631, 0.06945162, 0.60380518, 0.07038015,
       0.09988877, 0.03963715],[0.05935624, 0.03988419, 0.07961985, 0.        , 0.01989557,
       0.1005466 , 0.1003493 , 0.        , 0.02003061, 0.        ,
       0.03564284, 0.03518387, 0.08045386, 0.01002471, 0.01999258,
       0.01982941, 0.04026708, 0.03977918, 0.03971448, 0.0399734 ,
       0.04016032, 0.04999775, 0.19962923, 0.19830602, 0.        ,
       0.02496813, 0.02016453, 0.06966563, 0.60108968, 0.06925446,
       0.1006314 , 0.03970902],[0.06106667, 0.04056867, 0.08213325, 0.        , 0.020656  ,
       0.10203743, 0.10339198, 0.        , 0.02047589, 0.        ,
       0.03638853, 0.03585478, 0.08158818, 0.01030442, 0.02064738,
       0.02055541, 0.04123542, 0.04072045, 0.04120388, 0.04027505,
       0.04078202, 0.05101992, 0.20372124, 0.2047907 , 0.        ,
       0.02558456, 0.02058599, 0.07134167, 0.61410543, 0.07080644,
       0.1034692 , 0.04146668],[0.06027586, 0.04083144, 0.08078613, 0.        , 0.02035991,
       0.10206342, 0.10253362, 0.        , 0.02037377, 0.        ,
       0.03633044, 0.03637414, 0.08194719, 0.01029939, 0.02061408,
       0.02059954, 0.04100018, 0.04069362, 0.04114493, 0.0407505 ,
       0.0408078 , 0.05140135, 0.20579227, 0.20165331, 0.        ,
       0.02551086, 0.02032785, 0.07280723, 0.61853348, 0.07102045,
       0.10233134, 0.04162286],[0.06158264, 0.04078429, 0.08193381, 0.        , 0.02055382,
       0.10417823, 0.10314893, 0.        , 0.02027916, 0.        ,
       0.03567484, 0.03629815, 0.08321028, 0.01012355, 0.02064148,
       0.02072002, 0.04070845, 0.04110822, 0.04111882, 0.0409245 ,
       0.04064614, 0.05159463, 0.20655604, 0.20597771, 0.        ,
       0.02565488, 0.02077222, 0.07122024, 0.61555512, 0.07064335,
       0.10269719, 0.04051813],[0.0618693 , 0.04056719, 0.0820409 , 0.        , 0.02058642,
       0.10281221, 0.10099796, 0.        , 0.02060588, 0.        ,
       0.03633993, 0.03619553, 0.08148005, 0.010204  , 0.0207806 ,
       0.02084316, 0.04084523, 0.04139936, 0.04027093, 0.04081307,
       0.040827  , 0.05122552, 0.20433426, 0.20633948, 0.        ,
       0.02543642, 0.02059483, 0.07155683, 0.61175422, 0.07208193,
       0.10194039, 0.04164763],[0.06186104, 0.04146266, 0.0820959 , 0.        , 0.02035284,
       0.10214897, 0.10312006, 0.        , 0.02042795, 0.        ,
       0.0354985 , 0.03552164, 0.08181377, 0.01009082, 0.02082253,
       0.02045428, 0.04039616, 0.04073722, 0.0411795 , 0.04143745,
       0.04102364, 0.05209048, 0.20432872, 0.208244  , 0.        ,
       0.02573746, 0.02060726, 0.07215892, 0.61541628, 0.07135439,
       0.10225296, 0.0409095 ],[0.06235245, 0.0405196 , 0.08227043, 0.        , 0.02045143,
       0.10275602, 0.10116731, 0.        , 0.02067687, 0.        ,
       0.03578448, 0.03593992, 0.08193878, 0.01025191, 0.02055299,
       0.02042698, 0.0404981 , 0.04086101, 0.04093839, 0.04073805,
       0.04071751, 0.051507  , 0.2030118 , 0.20596256, 0.        ,
       0.02553232, 0.02075066, 0.07228193, 0.61864262, 0.07122625,
       0.10167097, 0.04110457],[0.05983545, 0.04019277, 0.08069902, 0.        , 0.02004683,
       0.09950473, 0.10142642, 0.        , 0.02006582, 0.        ,
       0.03486574, 0.0352729 , 0.080717  , 0.01004158, 0.01974116,
       0.02021901, 0.03991069, 0.03990734, 0.03990729, 0.03973007,
       0.03980144, 0.04956621, 0.20114076, 0.20008322, 0.        ,
       0.0248417 , 0.01996689, 0.07013793, 0.59988328, 0.06989004,
       0.09894206, 0.03975939],[0.06066201, 0.04005399, 0.07921911, 0.        , 0.01997205,
       0.09879467, 0.09945596, 0.        , 0.02000933, 0.        ,
       0.03485855, 0.03559667, 0.08054415, 0.00999655, 0.01994041,
       0.02024965, 0.03947835, 0.03993394, 0.04022437, 0.03984633,
       0.0396777 , 0.05051708, 0.20160643, 0.20029884, 0.        ,
       0.02473154, 0.02003933, 0.07045205, 0.59596902, 0.06972045,
       0.09985997, 0.03960805],[0.05949586, 0.03979408, 0.07959449, 0.        , 0.01979743,
       0.1005102 , 0.09979546, 0.        , 0.01990026, 0.        ,
       0.03525192, 0.03487844, 0.07956375, 0.00985564, 0.02002111,
       0.02014074, 0.04049091, 0.04006665, 0.04031472, 0.04030933,
       0.03986304, 0.0500088 , 0.19967948, 0.20297313, 0.        ,
       0.02493229, 0.01991233, 0.06917926, 0.60455121, 0.0704884 ,
       0.09927746, 0.03967093],[0.05971678, 0.04019373, 0.07931674, 0.        , 0.01980842,
       0.09989277, 0.09942423, 0.        , 0.01974356, 0.        ,
       0.03515183, 0.03516123, 0.0797641 , 0.01011742, 0.01995885,
       0.01976615, 0.04003799, 0.04050928, 0.04014604, 0.03986678,
       0.03950156, 0.04984467, 0.19924887, 0.19837417, 0.        ,
       0.02505958, 0.02018198, 0.0702125 , 0.605061  , 0.06977414,
       0.10079856, 0.03988126],[0.05996321, 0.03962104, 0.07908806, 0.        , 0.01999848,
       0.10096819, 0.09981682, 0.        , 0.01985115, 0.        ,
       0.03465235, 0.03515355, 0.0798524 , 0.0100702 , 0.02008988,
       0.0201922 , 0.04003178, 0.04003505, 0.0403039 , 0.04037962,
       0.0396646 , 0.05012858, 0.19837314, 0.1997995 , 0.        ,
       0.02474365, 0.01997176, 0.06992499, 0.59901125, 0.06932974,
       0.10075074, 0.04020203],[0.05976591, 0.03959577, 0.07939863, 0.        , 0.02001669,
       0.1003957 , 0.10043894, 0.        , 0.02025603, 0.        ,
       0.03478911, 0.03496858, 0.08041516, 0.00993992, 0.01999355,
       0.01997909, 0.04015376, 0.03961393, 0.03973053, 0.039637  ,
       0.0399977 , 0.04992482, 0.20116224, 0.19823888, 0.        ,
       0.02528233, 0.01994338, 0.06977376, 0.59538542, 0.07018534,
       0.09937324, 0.03972108]])

    _test_agent_1(env, agent, list_feng_11, list_feng_26, list_fu_he, list_fu_he_Q)
elif kk == 9:
    list_feng_11 = np.array([1.278125255569139, 1.5559365684289825, 1.390423485640884, 1.408891079530008, 1.4593127678678337, 1.5952194716429737, 1.2392907408154223, 1.4695087974293894, 1.4554979683939966, 1.5738062012839293, 1.4676412030173647, 1.566253216927625, 1.4218057181623143, 1.5732697101832867, 1.6079699974710375, 1.5476607248631695, 1.7310161201914425, 1.4718677370725977, 1.6439848355959676, 1.6414044433230488, 1.551113811423549, 1.5046501386100606, 1.4351671781623132, 1.4892636518521687])
    list_feng_26 = np.array([1.0770988119822107, 0.9803303017268516, 1.0901404834981587, 1.0169845301660663, 0.9722095275646352, 1.0822745570072996, 1.0648398714084601, 1.1678805956583993, 0.9951554874393185, 1.0436289109999133, 0.9969117318391065, 1.0623549216057533, 1.0799902740098368, 1.036627719446174, 0.9950686807404567, 0.9490565438847434, 1.0429659698325482, 1.0967526168107433, 0.957713759138083, 1.0255762632086636, 1.1142822958798877, 0.8418483723431425, 1.0295231996411298, 0.9448083235774454])
    list_fu_he = np.array([
        [0.07320981, 0.05495296, 0.09329595, 0.        , 0.04051608,
       0.13046984, 0.12714867, 0.        , 0.04713957, 0.        ,
       0.04376689, 0.04481077, 0.09395372, 0.04678161, 0.04258017,
       0.0406909 , 0.06221894, 0.06199424, 0.06244885, 0.06291186,
       0.06249011, 0.07033847, 0.28917035, 0.29575944, 0.        ,
       0.04090418, 0.04695941, 0.08228398, 0.14227399, 0.11843538,
       0.16696233, 0.04704479],[0.07325911, 0.05686052, 0.08622024, 0.        , 0.04447746,
       0.14240282, 0.13883069, 0.        , 0.04121299, 0.        ,
       0.04562201, 0.04662904, 0.09251713, 0.04522737, 0.03939147,
       0.042635  , 0.07027575, 0.05938727, 0.05890996, 0.05753213,
       0.06728222, 0.06194496, 0.32126914, 0.31718036, 0.        ,
       0.04337266, 0.04303129, 0.08322939, 0.1396689 , 0.10912642,
       0.13771114, 0.04056843],[0.07013208, 0.06346049, 0.08425474, 0.        , 0.04064469,
       0.14061469, 0.14149233, 0.        , 0.04065585, 0.        ,
       0.0409168 , 0.04194797, 0.08870808, 0.04311543, 0.03786035,
       0.04607999, 0.06281334, 0.06375437, 0.06344243, 0.0604958 ,
       0.06533194, 0.06262269, 0.2869342 , 0.28680667, 0.        ,
       0.04374425, 0.04450601, 0.08262814, 0.12545942, 0.1017482 ,
       0.16275226, 0.04323426],[0.06555776, 0.06191616, 0.08289015, 0.        , 0.04351315,
       0.12983074, 0.14170455, 0.        , 0.04147274, 0.        ,
       0.04213576, 0.04251671, 0.08733745, 0.04209831, 0.04117734,
       0.04336726, 0.05913255, 0.06080066, 0.06605695, 0.05981444,
       0.06116622, 0.05925769, 0.29706667, 0.30003797, 0.        ,
       0.04378122, 0.0420258 , 0.08708177, 0.13799685, 0.10951529,
       0.15090104, 0.04290987],[0.06944059, 0.06125345, 0.0827418 , 0.        , 0.0401327 ,
       0.1481144 , 0.13733823, 0.        , 0.04176119, 0.        ,
       0.03740917, 0.03957957, 0.08433279, 0.04271156, 0.04492106,
       0.04343348, 0.05881964, 0.06777199, 0.06872054, 0.06142612,
       0.05917732, 0.06521683, 0.30012528, 0.28590524, 0.        ,
       0.04405324, 0.04357819, 0.0779347 , 0.13921814, 0.10448845,
       0.14775494, 0.04229791],[0.07255109, 0.05994572, 0.08893142, 0.        , 0.03799969,
       0.13319615, 0.1264953 , 0.        , 0.0433105 , 0.        ,
       0.0400119 , 0.04343998, 0.08561792, 0.04474535, 0.04010266,
       0.04244958, 0.06068203, 0.06158607, 0.06668609, 0.06285146,
       0.05720684, 0.06450882, 0.31208656, 0.31020132, 0.        ,
       0.04069842, 0.03987969, 0.08157927, 0.14581139, 0.11341564,
       0.16272808, 0.04453944],[0.09265499, 0.08277757, 0.10782197, 0.        , 0.04882924,
       0.1768776 , 0.19519577, 0.        , 0.05577618, 0.        ,
       0.05135583, 0.05560771, 0.11646531, 0.0554587 , 0.05465354,
       0.05827626, 0.07935052, 0.08378223, 0.07934305, 0.07562109,
       0.0828723 , 0.07815944, 0.38264888, 0.38950496, 0.        ,
       0.05604222, 0.05157435, 0.10752965, 0.1788695 , 0.12577837,
       0.19480663, 0.05421034],[0.08163577, 0.07866063, 0.11254932, 0.        , 0.05365686,
       0.18581608, 0.18303481, 0.        , 0.05334601, 0.        ,
       0.0529486 , 0.05517813, 0.10331224, 0.05380755, 0.05570585,
       0.05388425, 0.08019318, 0.08518776, 0.08206252, 0.08457154,
       0.07682476, 0.08440388, 0.35753261, 0.38552936, 0.        ,
       0.05898128, 0.05418228, 0.10827471, 0.1798628 , 0.13039848,
       0.19561779, 0.05576376],[0.08965346, 0.08023636, 0.10534271, 0.        , 0.05386475,
       0.17403646, 0.18721333, 0.        , 0.05135845, 0.        ,
       0.05651161, 0.05365142, 0.11125865, 0.05543515, 0.05389244,
       0.05518697, 0.07739692, 0.08318085, 0.07659663, 0.07742616,
       0.08145459, 0.07796299, 0.38140452, 0.35002835, 0.        ,
       0.05619124, 0.05280645, 0.10265741, 0.17275909, 0.14470467,
       0.19844836, 0.05325767],[0.09529622, 0.08146444, 0.10822795, 0.        , 0.05480427,
       0.187783  , 0.17776685, 0.        , 0.05751975, 0.        ,
       0.05899877, 0.05419284, 0.1129909 , 0.05680597, 0.05342788,
       0.05635869, 0.07801917, 0.08248907, 0.08493558, 0.0793182 ,
       0.08172027, 0.07626312, 0.37844636, 0.37543844, 0.        ,
       0.05243242, 0.05576631, 0.1127612 , 0.17966556, 0.12577822,
       0.1920904 , 0.05424731],[0.09168767, 0.08127383, 0.11099121, 0.        , 0.0513504 ,
       0.17440647, 0.1861551 , 0.        , 0.05268272, 0.        ,
       0.05474204, 0.05500401, 0.11059968, 0.05686684, 0.05384778,
       0.05440652, 0.08613879, 0.07942279, 0.07609828, 0.08372932,
       0.08154999, 0.08190282, 0.3760713 , 0.38562368, 0.        ,
       0.05440302, 0.05360572, 0.10720594, 0.17103481, 0.13471053,
       0.19115926, 0.0541738 ],[0.09489147, 0.0843725 , 0.10410861, 0.        , 0.05505199,
       0.17781988, 0.17195372, 0.        , 0.05259951, 0.        ,
       0.05434902, 0.05624597, 0.10794289, 0.05154525, 0.05395103,
       0.05403444, 0.08781254, 0.07913557, 0.08321348, 0.07940678,
       0.0811029 , 0.08551096, 0.35816318, 0.37145788, 0.        ,
       0.05330849, 0.05547648, 0.11553566, 0.19101731, 0.13517721,
       0.19693905, 0.05448828],[0.10611719, 0.10154519, 0.12947148, 0.        , 0.06957572,
       0.20869712, 0.21664006, 0.        , 0.06977815, 0.        ,
       0.06578771, 0.06684655, 0.12951291, 0.06582802, 0.06536968,
       0.0687307 , 0.09869899, 0.09786885, 0.09593991, 0.10376343,
       0.09871328, 0.09724434, 0.45459476, 0.45784522, 0.        ,
       0.06636566, 0.06983644, 0.13399332, 0.21521877, 0.164039  ,
       0.23317974, 0.06698957],[0.11164435, 0.0941094 , 0.12860797, 0.        , 0.06640189,
       0.23145982, 0.22622461, 0.        , 0.06467169, 0.        ,
       0.06320639, 0.06284924, 0.13483761, 0.06630476, 0.06416767,
       0.06717266, 0.10087663, 0.09830796, 0.10157974, 0.10173097,
       0.09287544, 0.10295633, 0.45193494, 0.45233421, 0.        ,
       0.06800824, 0.06820466, 0.13592634, 0.21163821, 0.16080074,
       0.2153926 , 0.06619097],[0.10670598, 0.10043444, 0.12724105, 0.        , 0.06461969,
       0.22105108, 0.23097335, 0.        , 0.06705816, 0.        ,
       0.06471245, 0.06661094, 0.13172832, 0.06589445, 0.06661837,
       0.06615289, 0.10199879, 0.10118347, 0.09733226, 0.10078098,
       0.10060961, 0.09947956, 0.43276508, 0.44290115, 0.        ,
       0.06740686, 0.06737072, 0.13407859, 0.22295545, 0.16820503,
       0.23398148, 0.06678134],[0.10727554, 0.10196585, 0.13321011, 0.        , 0.06594115,
       0.22203144, 0.22681085, 0.        , 0.06580799, 0.        ,
       0.06679453, 0.06780807, 0.13451898, 0.06436644, 0.06937558,
       0.06713523, 0.09492846, 0.09668877, 0.09776804, 0.09875514,
       0.10064329, 0.1029357 , 0.4498151 , 0.46513037, 0.        ,
       0.06747795, 0.06724065, 0.13080576, 0.22123372, 0.16310521,
       0.2283664 , 0.06655225],[0.10817818, 0.09842005, 0.1281058 , 0.        , 0.06748163,
       0.2248162 , 0.21499814, 0.        , 0.0671566 , 0.        ,
       0.0649231 , 0.06625864, 0.13181283, 0.06339643, 0.06635299,
       0.0674375 , 0.10565455, 0.09779   , 0.09314517, 0.10062443,
       0.10148557, 0.10190394, 0.44685017, 0.4467208 , 0.        ,
       0.06732988, 0.06739016, 0.12640907, 0.22058721, 0.16667245,
       0.22119849, 0.06268716],[0.10280067, 0.09790202, 0.12642429, 0.        , 0.06637052,
       0.2175774 , 0.21174252, 0.        , 0.06545885, 0.        ,
       0.06622607, 0.06249064, 0.13570444, 0.06642261, 0.06729588,
       0.06852859, 0.09565508, 0.10312347, 0.10227145, 0.100315  ,
       0.09891311, 0.09439866, 0.47103476, 0.46114956, 0.        ,
       0.06689422, 0.06969738, 0.13638465, 0.2370687 , 0.16789376,
       0.23616492, 0.06854048],[0.0900878 , 0.08007505, 0.10518242, 0.        , 0.05528964,
       0.1775694 , 0.17356294, 0.        , 0.05115271, 0.        ,
       0.05532454, 0.05810275, 0.11258464, 0.05163323, 0.05576465,
       0.05529611, 0.07561539, 0.0757035 , 0.08402598, 0.08475429,
       0.08711935, 0.07936899, 0.35916769, 0.37362853, 0.        ,
       0.05147295, 0.05217895, 0.1076024 , 0.17745939, 0.13299228,
       0.19600904, 0.0529407 ],[0.09005753, 0.07684869, 0.10309664, 0.        , 0.05708291,
       0.19050989, 0.16920899, 0.        , 0.05575207, 0.        ,
       0.05501809, 0.05657674, 0.11198944, 0.0531618 , 0.05593757,
       0.05256832, 0.08546766, 0.07890836, 0.08197911, 0.08144575,
       0.07445876, 0.07893431, 0.37739945, 0.37576086, 0.        ,
       0.05477162, 0.05358144, 0.10384931, 0.17209918, 0.12682728,
       0.17620366, 0.05597521],[0.0908519 , 0.08223533, 0.11439476, 0.        , 0.05300022,
       0.19343688, 0.19422563, 0.        , 0.05719542, 0.        ,
       0.05172533, 0.05539123, 0.10445249, 0.05205882, 0.0555978 ,
       0.05546765, 0.0760163 , 0.08132033, 0.08036942, 0.07941559,
       0.0866207 , 0.08511088, 0.38607473, 0.4116505 , 0.        ,
       0.05316179, 0.05246521, 0.11221904, 0.17199617, 0.13634996,
       0.18986074, 0.0570579 ],[0.09415544, 0.08555413, 0.1141542 , 0.        , 0.04992385,
       0.17319524, 0.18099528, 0.        , 0.05625173, 0.        ,
       0.05170537, 0.0559368 , 0.10724844, 0.05340067, 0.05407283,
       0.05678733, 0.07960722, 0.07674483, 0.07907339, 0.08590104,
       0.08445989, 0.08232704, 0.39365633, 0.36368936, 0.        ,
       0.05485435, 0.05419463, 0.11461081, 0.18269592, 0.1365832 ,
       0.18611285, 0.0527752 ],[0.08783178, 0.08387219, 0.11651874, 0.        , 0.05416517,
       0.17373902, 0.17533109, 0.        , 0.05133485, 0.        ,
       0.05430821, 0.05453668, 0.11082817, 0.05396517, 0.05429832,
       0.05741483, 0.07472919, 0.07632291, 0.08156603, 0.082059  ,
       0.08131355, 0.08329202, 0.38501921, 0.38510399, 0.        ,
       0.05411077, 0.05429273, 0.10637671, 0.17838225, 0.13624525,
       0.19349708, 0.05060314],[0.09342683, 0.08121983, 0.10779757, 0.        , 0.05784765,
       0.1729723 , 0.18310624, 0.        , 0.05392035, 0.        ,
       0.05660002, 0.05292984, 0.11460545, 0.05564942, 0.05690162,
       0.05229373, 0.08016551, 0.0806357 , 0.08343715, 0.08522062,
       0.08548289, 0.0748939 , 0.36691769, 0.38453387, 0.        ,
       0.04924738, 0.05583652, 0.11244903, 0.18900217, 0.13060436,
       0.17972986, 0.05304637]])
    list_fu_he_Q = np.array([
        [0.05918774, 0.03945349, 0.07687278, 0.        , 0.01967837,
       0.09709972, 0.09588757, 0.        , 0.01937896, 0.        ,
       0.03436041, 0.03432213, 0.07696475, 0.00962548, 0.01953836,
       0.0195896 , 0.03947381, 0.03897665, 0.03821203, 0.03888744,
       0.03964029, 0.04951332, 0.19583985, 0.19232498, 0.        ,
       0.02447419, 0.01978917, 0.06936214, 0.58720255, 0.06989448,
       0.09805761, 0.03992599],[0.05952   , 0.03936569, 0.07769743, 0.        , 0.01976518,
       0.09657238, 0.09813652, 0.        , 0.01932097, 0.        ,
       0.03452877, 0.03380987, 0.07854387, 0.00965485, 0.01944787,
       0.01988542, 0.03839352, 0.03888549, 0.0382956 , 0.03852918,
       0.03912446, 0.04802369, 0.19299976, 0.1930696 , 0.        ,
       0.02404047, 0.01987583, 0.06766674, 0.59207571, 0.06761856,
       0.09861122, 0.03861711],[0.05840774, 0.03911392, 0.07738604, 0.        , 0.01951007,
       0.09676668, 0.09852572, 0.        , 0.01927035, 0.        ,
       0.03434385, 0.03438975, 0.07727654, 0.00989655, 0.01954582,
       0.01970426, 0.03893765, 0.03875969, 0.03971799, 0.03903138,
       0.03912461, 0.04868307, 0.1928205 , 0.19837475, 0.        ,
       0.02399415, 0.01923617, 0.06814396, 0.58694474, 0.06847437,
       0.09845559, 0.03885752],[0.05877527, 0.03907218, 0.07882632, 0.        , 0.01928955,
       0.09872608, 0.0976591 , 0.        , 0.0192957 , 0.        ,
       0.03395641, 0.03393925, 0.07828072, 0.00967619, 0.01963398,
       0.01941854, 0.03930315, 0.03817387, 0.03844178, 0.03885143,
       0.03882846, 0.04890959, 0.19241058, 0.19658859, 0.        ,
       0.02437565, 0.01955243, 0.06768418, 0.58519383, 0.06717534,
       0.0964535 , 0.03888319],[0.05818342, 0.03909465, 0.07789883, 0.        , 0.01961248,
       0.09759034, 0.09673636, 0.        , 0.01971904, 0.        ,
       0.03444363, 0.03433982, 0.0787125 , 0.00974726, 0.01960309,
       0.01916065, 0.03919535, 0.0392173 , 0.03883943, 0.03858191,
       0.03899148, 0.04848682, 0.19723833, 0.19521549, 0.        ,
       0.0240287 , 0.01967032, 0.06840633, 0.58981196, 0.06890014,
       0.09916001, 0.03945402],[0.05810492, 0.03825909, 0.07753319, 0.        , 0.01954874,
       0.096847  , 0.09754311, 0.        , 0.01961984, 0.        ,
       0.03380214, 0.03386879, 0.07765334, 0.00979467, 0.01964984,
       0.01936021, 0.03923539, 0.03933954, 0.03926226, 0.03912694,
       0.03883166, 0.04854545, 0.19676044, 0.19449905, 0.        ,
       0.02466509, 0.01960686, 0.06889936, 0.58919149, 0.06854203,
       0.09671526, 0.03908641],[0.06035093, 0.04050966, 0.08011524, 0.        , 0.02015798,
       0.10061869, 0.09942496, 0.        , 0.01989093, 0.        ,
       0.03501083, 0.03475448, 0.08019493, 0.01006848, 0.01996327,
       0.02005886, 0.04035249, 0.03991042, 0.04051921, 0.04002109,
       0.04011666, 0.05034644, 0.19912741, 0.20178512, 0.        ,
       0.02520858, 0.01989405, 0.07048764, 0.59900581, 0.06999329,
       0.09940602, 0.04016464],[0.0593905 , 0.03992584, 0.08025968, 0.        , 0.01980584,
       0.10023932, 0.10103559, 0.        , 0.0199601 , 0.        ,
       0.03541911, 0.03491983, 0.07956608, 0.01006218, 0.01998726,
       0.02014319, 0.03984963, 0.04018813, 0.03989631, 0.04013927,
       0.04002498, 0.05061243, 0.19952308, 0.20146103, 0.        ,
       0.02511255, 0.02000322, 0.06919733, 0.60326622, 0.0702907 ,
       0.10017547, 0.03997194],[0.06002199, 0.03965257, 0.07984273, 0.        , 0.019966  ,
       0.1000132 , 0.10090424, 0.        , 0.02014581, 0.        ,
       0.03516737, 0.03522353, 0.08065143, 0.00993936, 0.02008931,
       0.02002985, 0.03986441, 0.03986639, 0.03983206, 0.04056029,
       0.03986685, 0.04960526, 0.19966853, 0.20024086, 0.        ,
       0.0251279 , 0.02005349, 0.06958826, 0.60632489, 0.07057432,
       0.10045443, 0.04042908],[0.05964889, 0.03991088, 0.08024113, 0.        , 0.01993385,
       0.10054284, 0.09997245, 0.        , 0.0199485 , 0.        ,
       0.0351757 , 0.03488451, 0.07939259, 0.0100437 , 0.0199114 ,
       0.02019945, 0.03988058, 0.04012869, 0.0397279 , 0.03970983,
       0.03971154, 0.05038773, 0.19898375, 0.19899054, 0.        ,
       0.02479197, 0.02001503, 0.06992239, 0.59991212, 0.07004506,
       0.10010626, 0.04020773],[0.05990816, 0.03978063, 0.08019085, 0.        , 0.01991874,
       0.09981042, 0.10034787, 0.        , 0.01996083, 0.        ,
       0.03456763, 0.03497451, 0.08017271, 0.01007388, 0.01979956,
       0.02002885, 0.04023196, 0.04019074, 0.03944518, 0.03980643,
       0.04038604, 0.0499823 , 0.20003412, 0.19865638, 0.        ,
       0.02502502, 0.02028245, 0.070625  , 0.5986874 , 0.07020948,
       0.09995023, 0.04000083],[0.05984404, 0.04035233, 0.07900501, 0.        , 0.02021904,
       0.10027785, 0.10006918, 0.        , 0.01990818, 0.        ,
       0.03509465, 0.03506851, 0.08005174, 0.00996787, 0.01990071,
       0.02009668, 0.0399165 , 0.04016762, 0.03988959, 0.03974064,
       0.0399427 , 0.0494472 , 0.20098971, 0.19880193, 0.        ,
       0.02485171, 0.01993357, 0.06930365, 0.60470129, 0.07024899,
       0.09984831, 0.04018415],[0.06057149, 0.04066222, 0.08212271, 0.        , 0.02037167,
       0.10109174, 0.10204433, 0.        , 0.02091008, 0.        ,
       0.0356307 , 0.03563717, 0.08214624, 0.01024374, 0.02050547,
       0.02082323, 0.04047697, 0.04105283, 0.04078505, 0.04114855,
       0.04102651, 0.05079618, 0.20693051, 0.20668989, 0.        ,
       0.02581375, 0.02060762, 0.07189926, 0.61837958, 0.07158368,
       0.10124155, 0.04058596],[0.06111971, 0.04057592, 0.08023948, 0.        , 0.02070807,
       0.10175242, 0.10243857, 0.        , 0.02035947, 0.        ,
       0.03569102, 0.03637644, 0.08237171, 0.01017884, 0.02051298,
       0.02036763, 0.04138233, 0.04061382, 0.04074216, 0.04054507,
       0.04129733, 0.05126634, 0.2059307 , 0.2019358 , 0.        ,
       0.0252034 , 0.02037472, 0.07247225, 0.61339151, 0.07131902,
       0.10324602, 0.04132652],[0.06174129, 0.04090524, 0.0826588 , 0.        , 0.02058519,
       0.10385946, 0.10250247, 0.        , 0.02056324, 0.        ,
       0.03591137, 0.03590455, 0.08176016, 0.01025765, 0.02058225,
       0.02059901, 0.04118101, 0.04052364, 0.04083084, 0.04092824,
       0.04114633, 0.05072984, 0.20286299, 0.20353908, 0.        ,
       0.02563256, 0.02059764, 0.07160802, 0.62010075, 0.07213617,
       0.10158678, 0.04118747],[0.06104737, 0.04106178, 0.08234111, 0.        , 0.02033251,
       0.10263718, 0.10295948, 0.        , 0.02060146, 0.        ,
       0.03616754, 0.0358315 , 0.08211413, 0.01024229, 0.02016797,
       0.02048527, 0.04081259, 0.04163251, 0.04091229, 0.04073868,
       0.04119315, 0.05145654, 0.20657558, 0.20414435, 0.        ,
       0.02572431, 0.02073436, 0.07224057, 0.62285439, 0.07213315,
       0.10271706, 0.04086321],[0.06148591, 0.04089268, 0.08261417, 0.        , 0.02035505,
       0.10334082, 0.10162351, 0.        , 0.02046647, 0.        ,
       0.03592477, 0.03604725, 0.0815693 , 0.01033108, 0.02043277,
       0.02042012, 0.04135048, 0.04127239, 0.04137141, 0.04132299,
       0.04101251, 0.05065177, 0.20852966, 0.20563004, 0.        ,
       0.02521817, 0.02051618, 0.07153569, 0.61749925, 0.07104953,
       0.10374483, 0.04069908],[0.06094157, 0.04167275, 0.08171749, 0.        , 0.02022589,
       0.10162096, 0.10303015, 0.        , 0.0206036 , 0.        ,
       0.03580055, 0.0358561 , 0.08175906, 0.01032678, 0.02043654,
       0.02068262, 0.04154302, 0.04141559, 0.04032263, 0.04094491,
       0.04157638, 0.05202374, 0.20384411, 0.20823362, 0.        ,
       0.02570005, 0.02069019, 0.07131177, 0.61279904, 0.07202643,
       0.1015295 , 0.04100683],[0.0600276 , 0.04051151, 0.08008324, 0.        , 0.01986088,
       0.0994221 , 0.10039018, 0.        , 0.02019421, 0.        ,
       0.03506195, 0.03483247, 0.07958841, 0.00989548, 0.02022343,
       0.01986367, 0.04007027, 0.03973274, 0.0403989 , 0.03996841,
       0.04028495, 0.05023383, 0.19972444, 0.19693711, 0.        ,
       0.02505153, 0.02011558, 0.06945294, 0.60370308, 0.06971042,
       0.09942317, 0.0396723 ],[0.06021328, 0.04030245, 0.08026579, 0.        , 0.01979182,
       0.10101729, 0.09959862, 0.        , 0.01984403, 0.        ,
       0.03529265, 0.03516923, 0.07908134, 0.01001257, 0.01990251,
       0.01989398, 0.04020888, 0.03972535, 0.04005108, 0.04058032,
       0.04012317, 0.05048236, 0.20107331, 0.19989193, 0.        ,
       0.0252312 , 0.02008478, 0.06930222, 0.60385353, 0.07009249,
       0.09894932, 0.04039798],[0.05944683, 0.03943885, 0.08082139, 0.        , 0.02004415,
       0.09954701, 0.0999207 , 0.        , 0.02009742, 0.        ,
       0.03468813, 0.03505366, 0.07992228, 0.0100448 , 0.01992914,
       0.01990675, 0.04012003, 0.03975294, 0.04055815, 0.03983099,
       0.03975295, 0.05025247, 0.20048634, 0.19904694, 0.        ,
       0.02460869, 0.02002643, 0.07059299, 0.60624037, 0.06975842,
       0.09973052, 0.04055549],[0.06016676, 0.03994758, 0.08112014, 0.        , 0.02009738,
       0.10087483, 0.09936454, 0.        , 0.01978627, 0.        ,
       0.03507887, 0.0352523 , 0.07952585, 0.0099729 , 0.02002563,
       0.02008232, 0.0398924 , 0.03939879, 0.0404839 , 0.03944483,
       0.04054137, 0.04969117, 0.19945455, 0.2025067 , 0.        ,
       0.02493471, 0.0197248 , 0.06967089, 0.60080776, 0.06955689,
       0.09950811, 0.04041705],[0.05983295, 0.03985929, 0.080248  , 0.        , 0.02013253,
       0.0997354 , 0.10050495, 0.        , 0.02023422, 0.        ,
       0.03494801, 0.03467113, 0.07923204, 0.01013814, 0.01988196,
       0.01983035, 0.03968279, 0.04022394, 0.04029791, 0.03982048,
       0.03985142, 0.04989948, 0.20220116, 0.19872513, 0.        ,
       0.02517349, 0.0199947 , 0.07082931, 0.60351627, 0.07044645,
       0.10064446, 0.03963565],[0.05967662, 0.03990345, 0.08046948, 0.        , 0.02010693,
       0.10024798, 0.09897825, 0.        , 0.02006287, 0.        ,
       0.03467852, 0.03469552, 0.07988061, 0.01008262, 0.01981273,
       0.02000918, 0.04008002, 0.03980071, 0.04010723, 0.04004621,
       0.04000394, 0.0503903 , 0.19975082, 0.20095946, 0.        ,
       0.02487776, 0.01978992, 0.06982904, 0.60769581, 0.06902761,
       0.09995627, 0.03964597]])

    _test_agent_1(env, agent, list_feng_11, list_feng_26, list_fu_he, list_fu_he_Q)

