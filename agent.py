
import numpy as np
import random

class QLearningAgent:
    def __init__(self, action_space, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = {}

    def get_state_key(self, state):
        return tuple(state)

    def act(self, state):
        state_key = self.get_state_key(state)
        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space)
        
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(self.action_space))
        else:
            return np.argmax(self.q_table[state_key])

    def learn(self, state, action, reward, next_state, done):
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)

        if state_key not in self.q_table:
            self.q_table[state_key] = np.zeros(self.action_space)

        if next_state_key not in self.q_table:
            self.q_table[next_state_key] = np.zeros(self.action_space)
        
        # Q-learning update rule
        best_next_action = np.argmax(self.q_table[next_state_key])
        target = reward + self.gamma * self.q_table[next_state_key][best_next_action] * (1 - done)
        self.q_table[state_key][action] += self.alpha * (target - self.q_table[state_key][action])
    