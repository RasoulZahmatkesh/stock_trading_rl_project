
import pandas as pd
import numpy as np
from environment import StockTradingEnv
from agent import QLearningAgent

# Load data
data = pd.read_csv('AAPL.csv')  # Stock price data for Apple or any other stock
env = StockTradingEnv(data)

# Create Q-Learning agent
agent = QLearningAgent(action_space=env.action_space.n)

# Train model
episodes = 1000
for episode in range(episodes):
    state = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        agent.learn(state, action, reward, next_state, done)
        state = next_state
        total_reward += reward
    
    if episode % 100 == 0:
        print(f'Episode {episode}, Total Reward: {total_reward}')
    