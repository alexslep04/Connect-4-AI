import matplotlib.pyplot as plt
import numpy as np
from gym_environment.connect4_env import Connect4Env
from models.dqn_agent import DQNAgent  # Ensure the correct import

# Enable interactive mode for live plotting
plt.ion()

# Function to smooth rewards using a moving average
def smooth_rewards(rewards, window=10):
    """Apply a simple moving average to the rewards for smoothing."""
    if len(rewards) < window:
        return rewards
    smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
    return smoothed

if __name__ == "__main__":
    env = Connect4Env()  # Initialize the environment
    agent = DQNAgent()  # Initialize the agent
    episodes = 10000  # Set the number of episodes for training

    all_rewards = []  # To store total rewards for each episode
    dqn_wins = 0  # Track DQN agent wins
    player1_wins = 0  # Track Player 1 wins
    player2_wins = 0  # Track Player 2 wins

    # Create a plot for live updates
    fig, ax = plt.subplots()
    line_rewards, = ax.plot([], label='Smoothed Total Reward')  # Line for rewards
    ax.set_xlim(0, episodes)  # Set x-axis limit dynamically
    ax.set_ylim(-100, 100)  # Set y-axis limit (adjustable based on reward range)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Smoothed Total Reward')
    ax.set_title('Total Reward per Episode')
    ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)  # Add solid baseline at y=0
    ax.legend()  # Show the legend

    window = 10  # Moving average window size for rewards
    
    for e in range(episodes):
        state = env.reset()  # Reset the environment at the start of each episode
        done = False
        total_reward = 0  # Track the total reward for this episode
        dqn_win = False  # Track whether the DQN agent won
        player_win = None  # Track who wins each episode (Player 1 or Player 2)

        # While the episode is still ongoing
        while not done:
            action = agent.act(state)  # Agent chooses action
            next_state, reward, done, _ = env.step(action)  # Take action in the environment

            reward = float(reward)  # Ensure 'reward' is scalar before checking

            # Accumulate total reward
            total_reward += reward

            # Track who won
            if reward == 10:  # Player 2 (DQN agent) wins
                dqn_win = True
                player_win = 2
            elif reward == -10:  # Player 1 wins
                player_win = 1

            # Store experience and train the agent
            agent.remember(state, action, reward, next_state, done)
            agent.replay()

            # Move to the next state
            state = next_state

        # Track win rate
        if dqn_win:
            dqn_wins += 1
        if player_win == 1:
            player1_wins += 1
        elif player_win == 2:
            player2_wins += 1

        # Calculate win rate
        win_rate = dqn_wins / (e + 1)

        # Log the total reward, win rate, and epsilon for the episode
        print(f"Episode {e}/{episodes} - Epsilon: {agent.epsilon:.4f}, Total Reward: {total_reward}, "
              f"Win Rate: {win_rate:.2f}, Player {player_win} wins!")

        # Store the total reward for the episode
        all_rewards.append(total_reward)

        # Update the target model every few episodes for stability
        if e % agent.update_target_frequency == 0:
            agent.update_target_model()

        # Smooth the rewards for plotting
        smoothed_rewards = smooth_rewards(all_rewards, window)

        # Dynamically adjust x-axis to the number of episodes processed so far
        ax.set_xlim(0, len(all_rewards))

        # Update the plot data
        line_rewards.set_xdata(range(len(smoothed_rewards)))
        line_rewards.set_ydata(smoothed_rewards)

        # Update the plot title with win rate and epsilon
        ax.set_title(f'Total Reward per Episode\nWin Rate: {win_rate:.2f} | Epsilon: {agent.epsilon:.4f}')

        plt.draw()
        plt.pause(0.01)  # Short pause to allow updates to be visible

# Final report on wins
print(f"\nFinal Results after {episodes} episodes:")
print(f"Player 1 (Human) Wins: {player1_wins}")
print(f"Player 2 (DQN Agent) Wins: {player2_wins}")
print(f"DQN Win Rate: {win_rate:.2f}")
