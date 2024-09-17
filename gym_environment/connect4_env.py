import gym
from gym import spaces
import numpy as np
from pygame_ui.connect4_game import Connect4Game  # Update this to reflect the new folder name


class Connect4Env(gym.Env):
    """Custom Connect4 environment for reinforcement learning."""
    
    def __init__(self):
        super(Connect4Env, self).__init__()
        
        # Define the action space: 7 possible actions (one for each column)
        self.action_space = spaces.Discrete(7)  # 7 columns in Connect 4
        
        # Observation space: The board is a 6x7 grid, with values 0 (empty), 1 (player 1), 2 (player 2)
        self.observation_space = spaces.Box(low=0, high=2, shape=(6, 7), dtype=int)
        
        # Initialize the Connect4 game instance
        self.game = Connect4Game()

    def reset(self):
        """Reset the game to an initial state and return the board."""
        self.game = Connect4Game()  # Reset the Pygame Connect 4 game
        return self.game.board  # Return the initial empty board (6x7 grid)

    def step(self, action):
        """
        Take an action in the game (drop a token in a column).
        Returns:
        - Next state (the updated board)
        - Reward (1 for winning, -1 for losing, 0 otherwise)
        - done (True if the game is over, False otherwise)
        - info (optional, empty for now)
        """
        # Check if the column is full (invalid action)
        if self.game.board[0][action] != 0:
            print(f"Invalid action in column {action}. Giving penalty.")
            return self.game.board, -1, True, {}  # Penalize agent for choosing a full column
    
        # Apply the action (drop a token in the selected column)
        self.game.drop_token(action)
    
        # Check if the game is over
        if self.game.game_over:
            # Assign rewards based on which player won (in Connect 4, the current player is switched after each move)
            reward = 10 if self.game.current_player == 2 else -10  # Player 2 (AI) gets positive reward if they won
            return self.game.board, reward, True, {}  # Return the board, reward, and done flag
    
        # If the game is still ongoing, no immediate reward
        return self.game.board, 0, False, {}


    def render(self, mode='human'):
        """Render the current state of the game using Pygame."""
        self.game.render()

    def close(self):
        """Close the environment (not much needed here)."""
        pass
