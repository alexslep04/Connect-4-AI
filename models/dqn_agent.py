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
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-5)  # Start with a small learning rate
        self.memory = deque(maxlen=600000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9999944  # Adjusted for per-episode decay
        self.batch_size = 128
        self.update_target_frequency = 50

        # Initialize loss history for tracking
        self.loss_history = []

    def remember(self, state, action, reward, next_state, done):
        # Save experience in the replay buffer
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, evaluate=False):
        """
        Choose an action based on epsilon-greedy policy.
        If 'evaluate' is True, the agent acts greedily (epsilon=0).
        """
        # Get a list of valid actions (columns that are not full)
        valid_actions = [c for c in range(7) if state[0][c] == 0]

        if not valid_actions:
            # No valid actions: Return None to indicate the episode should end
            return None

        # Set epsilon to 0 during evaluation to act greedily
        epsilon = 0.0 if evaluate else self.epsilon

        if np.random.rand() <= epsilon:
            # Explore: Choose a random valid action
            return random.choice(valid_actions)
        else:
            # Exploit: Choose the best action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)  # Shape: [1, 6, 7]

            with torch.no_grad():
                q_values = self.model(state_tensor)  # Output shape: [1, 7]

            q_values_np = q_values.cpu().numpy()[0]  # Shape: [7]

            # Mask invalid actions by setting their Q-values to negative infinity
            masked_q_values = np.full(q_values_np.shape, -np.inf)
            for action in valid_actions:
                masked_q_values[action] = q_values_np[action]

            best_action = int(np.argmax(masked_q_values))
            return best_action

    def replay(self):
        """
        Sample a minibatch from memory, compute Q-values, and perform gradient descent.
        Returns the loss value for tracking.
        """
        if len(self.memory) < self.batch_size:
            print(f"Skipping replay: Not enough samples in memory (only {len(self.memory)})")
            return None  # Not enough samples to train

        # Randomly sample minibatch from memory
        minibatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*minibatch)

        # Convert to tensors and move to the appropriate device
        states = torch.FloatTensor(np.array(states)).to(self.device)         # Shape: [batch_size, 6, 7]
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)     # Shape: [batch_size, 1]
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)    # Shape: [batch_size, 1]
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)  # Shape: [batch_size, 6, 7]
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)        # Shape: [batch_size, 1]

        # Compute current Q-values for the actions taken
        q_values = self.model(states).gather(1, actions)  # Shape: [batch_size, 1]

        # Compute next Q-values using the target network
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)  # Shape: [batch_size, 1]

        # Set next_q_values to 0 for terminal states (when done is True)
        next_q_values = next_q_values * (1 - dones)

        # Compute target Q-values
        target_q_values = rewards + (self.gamma * next_q_values)

        # Use Huber loss (Smooth L1) to compute loss instead of MSE
        loss = nn.SmoothL1Loss()(q_values, target_q_values)

        # Perform gradient descent
        self.optimizer.zero_grad()
        loss.backward()

        # Apply gradient clipping to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        # Store loss in history
        self.loss_history.append(loss.item())

        # Return loss value for tracking
        return loss.item()


    def update_epsilon(self):
        """
        Decay the exploration rate (epsilon) after each episode, ensuring it doesn't fall below the minimum value.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min

    def update_target_model(self):
        # Copy weights from the main model to the target model
        self.target_model.load_state_dict(self.model.state_dict())
