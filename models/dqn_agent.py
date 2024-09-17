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
        Implements Double DQN where we use the primary model to select actions,
        but the target model to compute the target Q-value.
        """
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        for state, action, reward, next_state, done in minibatch:
            state = torch.FloatTensor(state.flatten()).unsqueeze(0)
            next_state = torch.FloatTensor(next_state.flatten()).unsqueeze(0)

            # Predict Q-values for current state
            q_values = self.model(state)

            # Calculate target Q-value using Double DQN
            with torch.no_grad():
                if done:
                    target = reward
                else:
                    # Double DQN: Use the primary model to select action, but use target model to estimate Q-value
                    best_action_next = torch.argmax(self.model(next_state)).item()
                    target = reward + self.gamma * self.target_model(next_state)[0][best_action_next].item()

            # Clone Q-values and update the action with the target Q-value
            target_f = q_values.clone()
            target_f[0][action] = target

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
        Apply epsilon-annealing for more dynamic exploration.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_min  # Stop decaying once it reaches the minimum value

    def update_target_model(self):
        """
        Copy weights from the model to the target model for more stable learning.
        """
        self.target_model.load_state_dict(self.model.state_dict())
