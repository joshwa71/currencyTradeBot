import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

import itertools
import argparse
import re
import os
import pickle
import math

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

from sklearn.preprocessing import StandardScaler

def get_btc_data():
    data = pd.read_csv(r'C:\Users\joshu\Documents\RL\tadebot\bitcoin_data.csv')
    price_volume_data = data.drop('Time', axis=1)
    return price_volume_data.values

def get_tradeview_btc_data():
    data = pd.read_csv(r'C:\Users\joshu\Documents\RL\tadebot\tradeview_bitcoin_1d.csv')
    price_volume_data = data.drop(['time', 'open', 'high', 'low'], axis=1)
    return price_volume_data.values

def get_google_data():
    data = pd.read_csv(r'C:\Users\joshu\Documents\RL\tadebot\GOOG.csv')
    price_volume_data = data.drop(['Date', 'Open', 'High', 'Low', 'Adj Close'], axis=1)
    return price_volume_data.values

def get_scaler(env):
    states = []
    for i in range(env.n_step):
        action = np.random.choice(env.action_space)
        state, reward, done, info = env.step(action)
        states.append(state)
        if done:
            break

    scaler = StandardScaler()
    scaler.fit(states)
    return scaler


def make_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

class NeuralNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):

        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(state_dim, 15),
            nn.ReLU(),
            nn.Linear(15, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            #nn.Linear(15, 10),
            #nn.ReLU(),
            nn.Linear(10, action_dim)
        )

    def forward(self, x):
        x = x.to(device='cpu')
        logits = self.linear_relu_stack(x.float())
        return logits


class BitcoinEnv:

    def __init__(self, data, initial_investment=20000):

        self.price_history = data
        self.n_step, self.n_state = self.price_history.shape

        self.initial_investment = initial_investment
        self.cur_step = None
        self.btc_owned = None
        self.btc_price = None
        self.prev_btc_price = None
        self.ma7 = None
        self.ma14 = None
        self.ma25 = None
        self.ma50 = None
        self.ma100 = None
        self.RSI = None
        self.volume = None
        self.volume_ma = None
        self.acc_dist = None
        self.cash_in_hand = None

        self.action_space = [0, 1, 2]
        self.action_list = [0, 1, 2]

        self.state_dim = 13

        self.reset()

    def reset(self):
        self.cur_step = 0
        self.btc_owned = 0
        self.prev_btc_price = 0
        self.cash_in_hand = self.initial_investment
        self.btc_price = self.price_history[self.cur_step][0]

        return self._get_obs()

    def step(self, action):
        assert action in self.action_space

        prev_val = self._get_val()

        self.cur_step += 1

        self.prev_btc_price = self.btc_price
        self.btc_price = self.price_history[self.cur_step][0]
        self.ma7 = self.price_history[self.cur_step][1]
        self.ma14 = self.price_history[self.cur_step][2]
        self.ma25 = self.price_history[self.cur_step][3]
        self.ma50 = self.price_history[self.cur_step][4]
        self.ma100 = self.price_history[self.cur_step][5]
        self.volume = self.price_history[self.cur_step][6]
        self.volume_ma = self.price_history[self.cur_step][7]
        self.RSI = self.price_history[self.cur_step][8]
        self.acc_dist = self.price_history[self.cur_step][9]

        self._trade(action)

        cur_val = self._get_val()

        reward = cur_val - prev_val

        done = self.cur_step == self.n_step - 1

        info = {'cur_val': cur_val}

        return self._get_obs(), reward, done, info

    def _get_obs(self):
        obs = np.empty(self.state_dim)
        obs[0] = self.btc_owned
        obs[1] = self.cash_in_hand
        obs[2] = self.prev_btc_price
        obs[3] = self.btc_price
        obs[4] = self.ma7
        obs[5] = self.ma14
        obs[6] = self.ma25
        obs[7] = self.ma50
        obs[8] = self.ma100
        obs[9] = self.volume
        obs[10] = self.volume_ma
        obs[11] = self.RSI
        obs[12] = self.acc_dist
        return obs

    def _get_val(self):
        return self.btc_owned * self.btc_price + self.cash_in_hand

    def _trade(self, action):
        action_val = action

        if action_val == 1:
            self.cash_in_hand += self.btc_price * self.btc_owned
            self.btc_owned = 0
            
        if action_val == 0:

            could_buy = math.floor(self.cash_in_hand / self.btc_price)

            self.cash_in_hand -= could_buy * self.btc_price
            self.btc_owned += could_buy

class DQNAgent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.00001

        self.final_target = []
        self.final_loss = 0
        self.losses = []
        self.final_input = []
        self.final_predicted_target = []

        self.model = NeuralNetwork(self.state_size, self.action_size).to('cpu')
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model(torch.tensor(state))
        #print('max value = ' + str(torch.argmax(act_values).item()))
        return torch.argmax(act_values).item()

    def train(self, state, action, reward, next_state, done):

        if done:
            target = torch.tensor(reward)
            
        else:
            target = torch.tensor(reward + self.gamma * torch.max(self.model(torch.tensor(next_state))).item())

        target_full = self.model(torch.tensor(state))
        predicted_target = torch.clone(target_full)

        target_full[0][action] = target.item()

        #print('target full = ' + str(target_full))

        loss = self.loss_fn(predicted_target, target_full)
        #print('loss = ' + str(loss))

        self.losses.append(loss.item())

        self.optimizer.zero_grad()
        loss.backward()
        self.final_target = target_full
        self.final_predicted_target = predicted_target
        self.final_loss = loss
        self.final_input = state
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def print_model_weights(self):
        param_list = []
        for param in self.model.parameters():
            param_list.append(param)
        print('model weights = ' + str(param_list))
        print('final loss = ' + str(self.final_loss))
        print('final input = ' + str(self.final_input))
        print('final target = ' + str(self.final_target))
        print('final predicted target = ' + str(self.final_predicted_target))


    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)


def play_one_episode(agent, env, is_train):
    state = env.reset()
    state = scaler.transform([state])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        #print('next state = ' + str(next_state))
        next_state = scaler.transform([next_state])
        if is_train == 'train':
            agent.train(state, action, reward, next_state, done)
        state = next_state

    agent.print_model_weights()

    return info['cur_val']


if __name__ == '__main__':

    models_folder = 'btc_bot_linear_models'
    rewards_folder = 'btc_bot_linear_rewards'

    num_episodes = 100
    batch_size = 32
    initial_investment = 20000
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, help='either "train" or "test"')
    args = parser.parse_args()

    make_directory(models_folder)
    make_directory(rewards_folder)

    data = get_tradeview_btc_data()

    n_timesteps = data.shape[0]

    n_train  = n_timesteps // 2

    train_data = data[:n_train]
    test_data = data[n_train:]

    env = BitcoinEnv(train_data, initial_investment)
    state_size = env.state_dim
    action_size = len(env.action_space)

    agent = DQNAgent(state_size, action_size)

    scaler = get_scaler(env)


    portfolio_value = []
 
    if args.mode =='test':
        with open(f'{models_folder}/scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        env = BitcoinEnv(test_data, initial_investment)

        agent.epsilon = 0

        agent.load(f'{models_folder}/linear.npz')

    for e in range(num_episodes):
        t0 = datetime.now()

        print('\n')
        val = play_one_episode(agent, env, args.mode)

        #if e%1000 == 0:
        #print(e)

        dt = datetime.now() - t0
        print(f"episode: {e + 1}/{num_episodes}, episode end value: {val:.2f}, duration: {dt}")
        portfolio_value.append(val)

    if args.mode == 'train':
        agent.save(f'{models_folder}/linear.npz')

        with open(f'{models_folder}/scaler.pkl', 'wb') as f:
            pickle.dump(scaler, f)

        plt.plot(agent.losses)
        plt.show()

    np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)



