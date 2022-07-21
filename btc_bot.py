import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
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

def get_google_data():
    data = pd.read_csv(r'C:\Users\joshu\Documents\RL\tadebot\GOOG.csv')
    price_volume_data = data.drop(['Date', 'Open', 'High', 'Low', 'Adj Close'], axis=1)
    print(price_volume_data)
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


class LinearModel:

    def __init__(self, input_dim, n_action):
        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
        self.b = np.zeros(n_action)

        self.vb = 0
        self.vW = 0

        self.losses = []

    def predict(self, X):
        assert(len(X.shape) == 2)
        return X.dot(self.W) + self.b

    def sgd(self, X, Y, learning_rate=0.001, momentum=0.9):
        assert(len(X.shape) == 2)

        num_values = np.prod(Y.shape)

        Yhat = self.predict(X)
        gW = 2 * X.T.dot(Yhat - Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values

        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb

        self.W += self.vW
        self.b += self.vb

        mse = np.mean((Yhat - Y)**2)
        self.losses.append(mse)

        #print('self.w = ' + str(self.W))
        #print('self.b = ' + str(self.b))
        #print('Y = ' + str(Y))
        #print('Yhat = ' + str(Yhat))
        #print('\n')

    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 3)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class BitcoinEnv:

    def __init__(self, data, initial_investment=20000):

        self.price_history = data
        self.n_step, self.n_state = self.price_history.shape

        self.initial_investment = initial_investment
        self.cur_step = None
        self.btc_owned = None
        self.btc_price = None
        self.btc_trade_volume = None
        self.prev_btc_price = None
        self.volume = None

        self.action_space = [0, 1, 2]
        self.action_list = [0, 1, 2]

        self.state_dim = 5

        self.reset()

    def reset(self):
        self.cur_step = 0
        self.btc_owned = 0
        self.btc_price = self.price_history[self.cur_step][0]
        self.btc_trade_volume = self.price_history[self.cur_step][1]
        self.prev_btc_price = 0
        self.cash_in_hand = self.initial_investment
        return self._get_obs()

    def step(self, action):
        assert action in self.action_space

        prev_val = self._get_val()

        self.cur_step += 1

        self.prev_btc_price = self.btc_price
        self.btc_price = self.price_history[self.cur_step][0]
        self.btc_trade_volume = self.price_history[self.cur_step][1]

        self._trade(action)

        cur_val = self._get_val()

        reward = cur_val - prev_val

        #print('action taken = ' + str(action))
        #print('prev val = ' + str(prev_val))        
        #print('cur val = ' + str(cur_val))
        #print('reward val = ' + str(reward))
        #print('\n')

        done = self.cur_step == self.n_step - 1

        info = {'cur_val': cur_val}
        obs = self._get_obs()

        return obs, reward, done, info

    def _get_obs(self):
        obs = np.empty(self.state_dim)
        obs[0] = self.btc_owned
        obs[1] = self.btc_price
        obs[2] = self.cash_in_hand
        obs[3] = self.prev_btc_price
        obs[4] = self.btc_trade_volume
        return obs

    def _get_val(self):
        return self.btc_owned * self.btc_price + self.cash_in_hand

    def _trade(self, action):
        action_val = action

        if action_val == 1:
            self.cash_in_hand += self.btc_price * self.btc_owned

            #print('\n')
            #print('btc sold = ' + str(self.btc_owned) + ', cash in hand now ' + str(self.cash_in_hand) )
            #print('\n')

            self.btc_owned = 0
            

        if action_val == 0:

            could_buy = math.floor(self.cash_in_hand / self.btc_price)

            self.cash_in_hand -= could_buy * self.btc_price
            self.btc_owned += could_buy

            #print('btc bought = ' + str(could_buy) + ', cash in hand now ' + str(self.cash_in_hand) )
            #print('\n')

class DQNAgent(object):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999


        self.model = LinearModel(state_size, action_size)

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def train(self, state, action, reward, next_state, done):
        if done:
            target = reward
        else:
            target = reward + self.gamma * np.amax(self.model.predict(next_state), axis=1)

        target_full = self.model.predict(state)
        target_full[0, action] = target

        self.model.sgd(state, target_full)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


def play_one_episode(agent, env, is_train):
    state = env.reset()
    state = scaler.transform([state])
    done = False

    while not done:
        action = agent.act(state)
        next_state, reward, done, info = env.step(action)
        next_state = scaler.transform([next_state])
        if is_train == 'train':
            agent.train(state, action, reward, next_state, done)
        state = next_state

    return info['cur_val']



if __name__ == '__main__':

    models_folder = 'btc_bot_linear_models'
    rewards_folder = 'btc_bot_linear_rewards'

    num_episodes = 1000
    batch_size = 32
    initial_investment = 20000
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--mode', type=str, required=True, help='either "train" or "test"')
    args = parser.parse_args()

    make_directory(models_folder)
    make_directory(rewards_folder)

    data = get_btc_data()
    print(data)
    price_data = data[0]
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

        plt.plot(agent.model.losses)
        plt.show()

    print('print8')
    
    np.save(f'{rewards_folder}/{args.mode}.npy', portfolio_value)



