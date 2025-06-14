import numpy as np
import random
from collections import defaultdict

from .base import BaseLearningModel

class TabularQLearning(BaseLearningModel):
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, gamma=0.99, epsilon=0.99, epsilon_decay_rate=0.998):
        super().__init__()
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate 
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate

        # Initialize Q-table. Using a defaultdict for sparse state spaces,
        # otherwise a numpy array if states can be directly indexed.
        # For simplicity, assuming states can be used as keys (e.g., tuples for discrete states)
        # If state_space_size is well-defined and states are integers, a numpy array would be better:
        # self.q_table = np.zeros((state_space_size, action_space_size))
        self.q_table = defaultdict(lambda: np.zeros(self.action_space_size))

        self.loss = [] 

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space_size)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state=None, done=False):
        current_q_value = self.q_table[state][action]

        if done:
            target_q_value = reward
        else:
            # Q-learning update rule: Q(s,a) = Q(s,a) + alpha * [reward + gamma * max(Q(s',a')) - Q(s,a)]
            max_next_q = np.max(self.q_table[next_state]) if next_state is not None else 0
            target_q_value = reward + self.gamma * max_next_q

        self.q_table[state][action] += self.learning_rate * (target_q_value - current_q_value)
        self.decay_epsilon()
        # In tabular Q-learning, there isn't a 'loss' in the same sense as a neural network.
        # We can append the change in Q-value or just a placeholder if the API requires 'loss' to be populated.
        self.loss.append(abs(target_q_value - current_q_value))


    def decay_epsilon(self):
        self.epsilon *= self.epsilon_decay_rate