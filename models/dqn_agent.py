import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from models.dqn import DQN  # Ensure that DQN is correctly implemented

class DQNAgent:
    def __init__(self):
        # Set up the device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize your DQN models
        self.model = DQN().to(self.device)
        self.target_model = DQN().to(self.device)

        self.optimizer = optim.Adam(self.model.parameters())
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99999
        self.batch_size = 64
        self.update_target_frequency = 10
        
    def remember(self, state, action, reward, next_state, done):
        # Save experience in the replay buffer
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """
        Choose an action based on epsilon-greedy policy.
        """
        # Get a list of valid actions (columns that are not full)
        valid_actions = [c for c in range(7) if state[0][c] == 0]  # Ensure column is not full

        if not valid_actions:
            # No valid actions: Return a signal to indicate the episode should end
            return None  # Signals that no more valid actions are available

        if np.random.rand() <= self.epsilon:
            # Explore: Choose a random valid action
            return random.choice(valid_actions)
        else:
            # Exploit: Choose the best action (greedy)
            # Convert state to a tensor and move it to the appropriate device
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Shape: [1, 6, 7]

            # Get Q-values from the model
            with torch.no_grad():
                q_values = self.model(state_tensor)  # Output shape: [1, 7]

            # Convert Q-values to a numpy array
            q_values_np = q_values.cpu().numpy()[0]  # Shape: [7]

            # Mask invalid actions by setting their Q-values to a very low value
            masked_q_values = np.full(q_values_np.shape, -np.inf)
            for action in valid_actions:
                masked_q_values[action] = q_values_np[action]

            # Select the action with the highest Q-value among valid actions
            best_action = int(np.argmax(masked_q_values))

            return best_action

    def replay(self):
        """
        Sample random minibatch from memory, compute Q-values, and perform gradient descent.
        """
        if len(self.memory) < self.batch_size:
            return

        # Randomly sample minibatch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        # Unpack the minibatch
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to tensors and move to the appropriate device
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute current Q-values for the actions taken
        q_values = self.model(states).gather(1, actions)  # Shape: [batch_size, 1]

        # Compute next Q-values using the target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)  # Shape: [batch_size, 1]

        # Compute target Q-values
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = nn.MSELoss()(q_values, target_q_values)

        # Perform gradient descent
        self.optimizer.zero_grad()
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
