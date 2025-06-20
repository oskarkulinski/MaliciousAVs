import numpy as np
import random
from collections import defaultdict

from .base import BaseLearningModel

class TabularQLearning(BaseLearningModel):
    def __init__(self, number_routes, bin_size=5, cars=100, learning_rate=0.1, gamma=0.99, epsilon=0.99, epsilon_decay_rate=0.9829):
        super().__init__()
        self.number_routes = number_routes
        self.bin_size = bin_size
        self.max_bins = cars// bin_size +1
        self.learning_rate = learning_rate 
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon
        self.epsilon_decay_rate = epsilon_decay_rate

        # Initialize Q-table. Using a defaultdict for sparse state spaces,
        # otherwise a numpy array if states can be directly indexed.
        # For simplicity, assuming states can be used as keys (e.g., tuples for discrete states)
        # If state_space_size is well-defined and states are integers, a numpy array would be better:
        # self.q_table = np.zeros((state_space_size, action_space_size))
        self.q_table = defaultdict(lambda: np.zeros(self.number_routes))

        self.loss = [] 

    def create_feature_state(self, route_cars):
        """
        Create a smaller state space using engineered features
        """
        features = []
        
        # Feature 1: Which route has minimum cars?
        min_cars_route = np.argmin(route_cars)
        features.append(min_cars_route)
        
        # Feature 2: Difference between min and max cars
        car_spread = np.max(route_cars) - np.min(route_cars)
        features.append(car_spread)
        
        return tuple(features)

    def act(self, state):
        state = self.create_feature_state(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.number_routes)
        else:
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state=None, done=False):
        state = self.create_feature_state(state)
        next_state = self.create_feature_state(next_state) if next_state else None
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