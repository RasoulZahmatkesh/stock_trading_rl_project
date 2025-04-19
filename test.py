
import pandas as pd
from environment import StockTradingEnv
from agent import QLearningAgent

# Load data
data = pd.read_csv('AAPL.csv')  # Stock price data for Apple or any other stock
env = StockTradingEnv(data)

# Create Q-Learning agent
agent = QLearningAgent(action_space=env.action_space.n)

# Load trained model
# In this case, you can load the trained model if it was saved earlier
# agent.q_table = load_model('q_table.pkl')

# Test model
state = env.reset()
done = False
while not done:
    action = agent.act(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    env.render()  # Display results at each step
    