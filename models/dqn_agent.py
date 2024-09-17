import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from models.dqn import DQN

class DQNAgent:
    def __init__(self):
        self.model = DQN()  # Initialize the DQN model
        self.target_model = DQN()  # Target model for more stable learning
        self.optimizer = optim.Adam(self.model.parameters())
        self.memory = deque(maxlen=2000)  # Replay memory
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.999
        self.batch_size = 64
        self.update_target_frequency = 10  # Update target network every 10 episodes
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(7)  # Random action (explore)
        state = torch.FloatTensor(state.flatten()).unsqueeze(0)
        q_values = self.model(state)
        return torch.argmax(q_values).item()  # Greedy action (exploit)

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

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
