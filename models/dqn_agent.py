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
        self.memory = deque(maxlen=10000)  # Experience replay buffer
        self.gamma = 0.99  # Discount factor for future rewards
        self.epsilon = 1.0  # Initial exploration rate
        self.epsilon_min = 0.1  # Minimum exploration rate
        self.epsilon_decay = 0.99999  # Exploration decay rate (set for 10,000 episodes)
        self.batch_size = 64  # Mini-batch size for training
        self.update_target_frequency = 10  # Update target model every 10 episodes
    
    def remember(self, state, action, reward, next_state, done):
        # Save experience in the replay buffer
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Choose an action based on epsilon-greedy policy.
        If random number <= epsilon, choose random action (explore),
        otherwise choose greedy action from model (exploit).
        """
        # Get a list of valid actions (columns that are not full)
        valid_actions = [c for c in range(7) if state[0][c] == 0]  # Ensure column is not full

        if len(valid_actions) == 0:
            # No valid actions: Return a signal to indicate a draw (end the episode)
            return None  # This signals that no more valid actions are available

        if np.random.rand() <= self.epsilon:
            # Explore: Choose a random valid action
            return random.choice(valid_actions)
    
        # Exploit: Choose the best action (greedy)
        state = torch.FloatTensor(state.flatten()).unsqueeze(0)  # Convert state to tensor
        q_values = self.model(state)  # Predict Q-values
    
        # Filter Q-values to consider only valid actions
        q_values_np = q_values.detach().cpu().numpy().flatten()
        valid_q_values = np.array([q_values_np[c] for c in valid_actions])
    
        # Get the action corresponding to the highest valid Q-value
        best_action = valid_actions[np.argmax(valid_q_values)]
    
        return best_action

    def replay(self):
        """
        Sample random minibatch from memory, compute Q-values, and perform gradient descent.
        """
        if len(self.memory) < self.batch_size:
            return

        # Randomly sample minibatch from memory
        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            # Flatten the state and next_state before feeding into the model
            state = torch.FloatTensor(state.flatten()).unsqueeze(0)  # Convert to 1D tensor
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
        Decay the exploration rate (epsilon) after each episode, ensuring it doesn't fall below the minimum value.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min  # Ensure epsilon does not fall below minimum value

    def update_target_model(self):
        # Copy weights from model to target model for stability
        self.target_model.load_state_dict(self.model.state_dict())
