import gym
from gym import spaces
import numpy as np
from pygame_ui.connect4_game import Connect4Game  # Ensure this path is correct

class Connect4Env:
    def __init__(self):
        # Initialize the game
        self.game = Connect4Game()  # Instantiate the Connect4Game class

    def reset(self):
        """
        Reset the game to start a new episode.
        """
        self.game = Connect4Game()  # Re-initialize the game for a new episode
        return self.game.board

    def step(self, action):
        """
        Take an action in the game (drop a token in a column).
        Returns:
        - Next state (the updated board)
        - Reward (+10 for winning, -10 for losing, 0 otherwise)
        - done (True if the game is over, False otherwise)
        - info (optional, empty for now)
        """
        # Apply the action (drop a token in the selected column)
        self.game.drop_token(action)

        # Check if the game is over (win/loss condition)
        if self.game.game_over:
            reward = 10 if self.game.current_player == 2 else -10  # AI wins: +10, AI loses: -10
            return self.game.board, reward, True, {}

        # If the game is still ongoing, return the updated board and neutral reward
        return self.game.board, 0, False, {}

    def render(self, mode='human'):
        """Render the current state of the game using Pygame."""
        self.game.render()

    def close(self):
        """Close the environment (not much needed here)."""
        pass
