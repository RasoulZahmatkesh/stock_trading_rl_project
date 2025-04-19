
import gym
import numpy as np
import pandas as pd
import random

class StockTradingEnv(gym.Env):
    def __init__(self, data, window_size=10):
        super(StockTradingEnv, self).__init__()
        
        self.data = data  # Stock price data
        self.window_size = window_size  # Number of past days the model looks at
        self.current_step = 0
        
        # Actions: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = gym.spaces.Discrete(3)
        
        # Observation space: stock prices
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=(self.window_size,), dtype=np.float32)
        
    def reset(self):
        self.current_step = self.window_size
        self.state = self.data['Close'].iloc[self.current_step-self.window_size:self.current_step].values
        return self.state
    
    def step(self, action):
        self.current_step += 1
        done = False
        reward = 0
        
        if self.current_step >= len(self.data) - 1:
            done = True

        current_price = self.data['Close'].iloc[self.current_step]
        prev_price = self.data['Close'].iloc[self.current_step - 1]
        
        if action == 1:  # Buy
            reward = current_price - prev_price  # Profit or loss from buying
        elif action == 2:  # Sell
            reward = prev_price - current_price  # Profit or loss from selling
        
        self.state = self.data['Close'].iloc[self.current_step-self.window_size:self.current_step].values
        return self.state, reward, done, {}

    def render(self):
        profit = self.data['Close'].iloc[self.current_step] - self.data['Close'].iloc[0]
        print(f'Step: {self.current_step}, Profit: {profit}')
    