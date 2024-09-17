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
        self.epsilon_min = 0.1  # Minimum epsilon value
        self.epsilon_decay = 0.9995  # How quickly epsilon decays
        self.batch_size = 64
        self.update_target_frequency = 10  # Update target network every 10 episodes
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experiences in replay memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Choose an action based on epsilon-greedy policy.
        If random number <= epsilon, choose random action (explore),
        otherwise choose greedy action from model (exploit).
        """
        # Get a list of valid actions (columns that are not full)
        valid_actions = [c for c in range(7) if state[0][c] == 0]  # Ensure column is not full
    
        if np.random.rand() <= self.epsilon:
            # Explore: Choose a random valid action
            return random.choice(valid_actions)
    
        # Exploit: Choose the best action (greedy)
        state = torch.FloatTensor(state.flatten()).unsqueeze(0)  # Convert state to tensor
        q_values = self.model(state)  # Predict Q-values
    
        # Choose the action with the highest Q-value, but only among valid actions
        q_values_np = q_values.detach().cpu().numpy().flatten()
        best_action = valid_actions[np.argmax(q_values_np[valid_actions])]
    
        return best_action


    def replay(self):
        """
        Sample random minibatch from memory, compute Q-values, and perform gradient descent.
        """
        if len(self.memory) < self.batch_size:
            return

        # Randomly sample minibatch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        
        # Iterate through each sample in the minibatch
        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state.flatten()).unsqueeze(0)
            next_state = torch.FloatTensor(next_state.flatten()).unsqueeze(0)

            # Compute target Q-value
            target = reward
            if not done:
                target = reward + self.gamma * torch.max(self.target_model(next_state)).item()

            # Get current Q-values for the state
            q_values = self.model(state)

            # Clone the Q-values so we can modify the specific action's Q-value
            target_f = q_values.clone()
            target_f[0][action] = target  # Set the target Q-value for the specific action

            # Perform gradient descent
            self.optimizer.zero_grad()
            loss = nn.MSELoss()(q_values, target_f)
            loss.backward()
            self.optimizer.step()

        # Decay epsilon after each replay step
        self.update_epsilon()

    def update_epsilon(self):
        """
        Decay the exploration rate (epsilon) after each episode.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_model(self):
        """
        Copy weights from the model to the target model for more stable learning.
        """
        self.target_model.load_state_dict(self.model.state_dict())
