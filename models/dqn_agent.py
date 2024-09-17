import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from models.dqn import DQN  # Ensure that DQN is correctly implemented

class DQNAgent:
    def __init__(self):
        self.model = DQN()  # Initialize your DQN model
        self.target_model = DQN()  # Initialize target model for stable updates
        self.optimizer = optim.Adam(self.model.parameters())  # Adam optimizer
        self.memory = deque(maxlen=2000)  # Experience replay buffer
        self.gamma = 0.99  # Discount factor for future rewards
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.epsilon_decay = 0.99974  # Exploration decay rate
        self.batch_size = 64  # Mini-batch size for training
        self.update_target_frequency = 10  # Update target model every 10 episodes
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        valid_actions = [c for c in range(7) if state[0][c] == 0]  # Get valid actions (non-full columns)
        if len(valid_actions) == 0:
            return None  # No valid actions left

        if np.random.rand() <= self.epsilon:
            return random.choice(valid_actions)  # Random action

        state_tensor = torch.FloatTensor(state.flatten()).unsqueeze(0)
        q_values = self.model(state_tensor).detach().cpu().numpy().flatten()

        # Mask out invalid actions
        q_values[~np.isin(range(7), valid_actions)] = -np.inf

        return np.argmax(q_values)  # Return the best action based on Q-values

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state.flatten()).unsqueeze(0)
            next_state = torch.FloatTensor(next_state.flatten()).unsqueeze(0)

            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state)).item()

            q_values = self.model(state)
            target_f = q_values.clone()
            target_f[0][action] = target

            self.optimizer.zero_grad()
            loss = nn.MSELoss()(q_values, target_f)
            loss.backward()
            self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
