from tqdm import tqdm
import numpy as np
import torch
import collections
import random
import matplotlib.pyplot as plt

import pickle


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size, model_path='model.pth'):
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)

    torch.save(agent.actor.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    torch.save(agent.critic_1.state_dict(), 'critic_1_model.pth')
    torch.save(agent.critic_2.state_dict(), 'critic_2_model.pth')
    print(f"Critic model saved")

    return return_list


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


def test_agent(env, agent, model_path='model.pth',model_path_1 = 'critic_1_model.pth', model_path_2 = 'critic_2_model.pth'):
    agent.actor.load_state_dict(torch.load(model_path))
    agent.actor.eval()
    total_cost = 0
    state = env.reset()
    done = False

    hourly_metrics = {
        'total_loss':[],
        'cost': [],
        'sgen_p': [],
        'sgen_q': [],
        'load': [],
        "total_sgen":[],
    }

    while not done:
        action = agent.take_action(state,stochastic=True)
        next_state, reward, done, info = env.step(action)

        cost = -reward
        total_loss=info.get('total_loss',0)
        sgen_p = info.get('sgen_p', 0)
        sgen_q = info.get('sgen_q', 0)
        load_p = info.get("load_p", 0)
        total_load = sum(load_p)
        total_sgen = info.get('total_sgen', 0)
        total_cost += cost
        hourly_metrics['cost'].append(cost)
        hourly_metrics['total_loss'].append(total_loss)
        hourly_metrics["sgen_p"].append(sgen_p)
        hourly_metrics["sgen_q"].append(sgen_q)
        hourly_metrics["load"].append(total_load)
        hourly_metrics["total_sgen"].append(total_sgen)
        state = next_state

    avg_hourly_cost = total_cost / 47

    print("\nTest Results:")
    print(f"Average Hourly Cost: {avg_hourly_cost:.2f}")
    print("Load", hourly_metrics["load"])
    print("Total Generation", hourly_metrics["total_sgen"])
    print("total_loss",hourly_metrics["total_loss"])
    print("sgen_p", hourly_metrics["sgen_p"])
    print("sgen_q", hourly_metrics["sgen_q"])
    print("total_cost", total_cost)

    return {
        'avg_hourly_cost': avg_hourly_cost,
        'hourly_metrics': hourly_metrics
    }


def test_agent_com(env, agent, model_path='model.pth'):
    agent.actor.load_state_dict(torch.load(model_path))
    agent.actor.eval()
    total_cost = 0
    total_gen_cost = 0
    line_overload_counts = np.zeros(len(env.net.line))
    state = env.reset()
    done = False

    hourly_metrics = {
        'cost': [],
        'gen_p': [],
        'wind_used': [],
        'pv_used': [],
        'load': [],
        "total_gen":[],
        'gen_cost': [],
        'renew_utilization': [],
        'line_loading': []
    }

    while not done:
        action = agent.take_action(state)
        next_state, reward, done, info = env.step(action)
        cost = -reward
        gen_p = info.get('gen_p', 0)
        hourly_metrics['cost'].append(cost)
        hourly_metrics["gen_p"].append(gen_p)
        state = next_state

    print("\nTest Results:")

    print("Hourly cost", np.mean(hourly_metrics["cost"]))

    print("Generation of Conventional Units", hourly_metrics["gen_p"])

