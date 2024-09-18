import gym
from gym import spaces
import numpy as np
import pygame  # Import pygame if you're using it in this file
from pygame_ui.connect4_game import Connect4Game
import random

class Connect4Env(gym.Env):
    def __init__(self, render_mode=False):
        super(Connect4Env, self).__init__()
        self.render_mode = render_mode

        # Define action and observation spaces
        self.action_space = spaces.Discrete(7)  # 7 columns in Connect 4
        self.observation_space = spaces.Box(low=0, high=2, shape=(6, 7), dtype=np.int8)  # Board shape

        # Initialize the game
        self.game = Connect4Game(render_mode=self.render_mode)

        # Initialize rendering if render_mode is True
        if self.render_mode:
            pygame.init()
            # Additional rendering setup if needed

    def reset(self):
        # Reset the game and return the initial state
        self.game = Connect4Game(render_mode=self.render_mode)
        return np.copy(self.game.board)

    def step(self, action):
        """
        Take an action in the game (drop a token in a column).
        Returns:
        - Next state (the updated board)
        - Reward (+10 for winning, -10 for losing, 0 otherwise)
        - done (True if the game is over, False otherwise)
        - info (optional, empty for now)
        """
        # Agent's move (Player 2)
        agent_player = self.game.current_player  # Should be Player 2 (the agent)
        try:
            self.game.drop_token(action)
        except ValueError:
            # Penalize the agent for invalid moves
            reward = -5
            done = False
            return np.copy(self.game.board), reward, done, {}

        # Check if the agent has won
        if self.game.check_win(agent_player):
            reward = 1  # Agent wins
            done = True
            return np.copy(self.game.board), reward, done, {}

        # Check for a draw
        if self.game.is_board_full():
            reward = -0.2  # Draw
            done = True
            return np.copy(self.game.board), reward, done, {}

        # Opponent's move (Player 1)
        opponent_player = self.game.current_player  # Now it's Player 1's turn
        valid_actions = self.game.get_valid_actions()
        if valid_actions:
            opponent_action = random.choice(valid_actions)
            self.game.drop_token(opponent_action)

            # Check if the opponent has won
            if self.game.check_win(opponent_player):
                reward = -1  # Agent loses
                done = True
                return np.copy(self.game.board), reward, done, {}

            # Check for a draw after opponent's move
            if self.game.is_board_full():
                reward = 0  # Draw
                done = True
                return np.copy(self.game.board), reward, done, {}
        else:
            # No valid actions left; it's a draw
            reward = 0
            done = True
            return np.copy(self.game.board), reward, done, {}

        # Continue the game
        reward = 0
        done = False
        return np.copy(self.game.board), reward, done, {}

    def render(self, mode='human'):
        # Render the game state if rendering is enabled
        if self.render_mode:
            self.game.render()

    def close(self):
        # Close the game environment and clean up resources
        if self.render_mode:
            pygame.quit()
