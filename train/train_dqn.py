import matplotlib.pyplot as plt
import numpy as np
from gym_environment.connect4_env import Connect4Env
from models.dqn_agent import DQNAgent

import torch  # Add torch import if not already present
import time   # For tracking training time

# Enable interactive mode for live plotting
plt.ion()

# Function to smooth rewards using a moving average
def smooth_rewards(rewards, window=10):
    """Apply a simple moving average to the rewards for smoothing."""
    if len(rewards) < window:
        return rewards
    smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
    return smoothed

# Evaluation function
def evaluate_agent(agent, env, episodes=100):
    """Evaluate the agent's performance without exploration."""
    wins = 0
    total_rewards = []
    agent_epsilon_backup = agent.epsilon  # Backup the current epsilon
    agent.epsilon = 0.0  # Disable exploration during evaluation

    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            state = next_state
        total_rewards.append(reward)
        if reward == 1:
            wins += 1

    agent.epsilon = agent_epsilon_backup  # Restore the original epsilon
    win_rate = wins / episodes
    average_reward = np.mean(total_rewards)
    return win_rate, average_reward

if __name__ == "__main__":
    env = Connect4Env(render_mode=False)  # Initialize the environment
    agent = DQNAgent()  # Initialize the agent
    episodes = 15000  # Set the number of episodes for training

    all_rewards = []  # To store total rewards for each episode
    all_losses = []   # To store loss values
    evaluation_win_rates = []  # To store evaluation win rates
    evaluation_episodes = []   # To keep track of episodes at which evaluations occur
    dqn_wins = 0  # Track DQN agent wins
    player1_wins = 0  # Track Player 1 wins
    player2_wins = 0  # Track Player 2 wins

    # Create a plot for live updates with padding between subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig.subplots_adjust(hspace=0.4)  # Adjust the space between the two subplots

    line_rewards, = ax1.plot([], label='Smoothed Total Reward')  # Line for rewards
    line_losses, = ax2.plot([], label='Smoothed Loss', color='orange')  # Line for losses

    ax1.set_xlim(0, episodes)
    ax1.set_ylim(-50, 50)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Smoothed Total Reward')
    ax1.set_title('Total Reward per Episode', pad=20)  # Add padding to the title
    ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)
    ax1.legend()

    ax2.set_xlim(0, episodes)
    ax2.set_ylim(-2, 2)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Smoothed Loss')
    ax2.set_title('Training Loss per Episode', pad=20)  # Add padding to the title
    ax2.legend()

    window = 100  # Moving average window size for rewards and losses

    # Initialize global cumulative reward
    cumulative_total_reward = 0  # Tracks rewards across all episodes

    # Start training
start_time = time.time()
for e in range(episodes):
    state = env.reset()  # Reset the environment at the start of each episode
    done = False
    total_reward = 0  # Track the total reward for this episode
    dqn_win = False  # Track whether the DQN agent won
    player_win = None  # Track who wins each episode (Player 1 or Player 2)

    episode_losses = []  # To track losses within the episode

    # While the episode is still ongoing
    while not done:
        action = agent.act(state)  # Agent chooses action
        next_state, reward, done, _ = env.step(action)  # Take action in the environment

        reward = float(reward)  # Ensure 'reward' is scalar before checking

        # Accumulate total reward for this episode
        total_reward += reward

        # Track who won
        if reward == 1:  # Player 2 (DQN agent) wins
            dqn_win = True
            player_win = 2
        elif reward == -1:  # Player 1 wins
            player_win = 1
        else:
            player_win = None

        # Store experience and train the agent
        agent.remember(state, action, reward, next_state, done)
        loss = agent.replay()  # Capture the loss value
        if loss is not None:
            episode_losses.append(loss)
            print(f"Loss for episode {e + 1}: {loss}")
        else:
            print(f"Replay buffer size: {len(agent.memory)}/{agent.batch_size}")



        # Move to the next state
        state = next_state

    # Update epsilon after the episode
    agent.update_epsilon()

    # Accumulate the reward over all episodes (cumulative tracking)
    cumulative_total_reward += total_reward

    # Track win rate
    if dqn_win:
        dqn_wins += 1
    if player_win == 1:
        player1_wins += 1
    elif player_win == 2:
        player2_wins += 1

    # Calculate win rate
    win_rate = dqn_wins / (e + 1)

    # Log the total reward, win rate, cumulative reward, and epsilon for the episode
    print(f"Episode {e + 1}/{episodes} - Epsilon: {agent.epsilon:.4f}, "
          f"Episode Reward: {total_reward}, "
          f"Cumulative Total Reward: {cumulative_total_reward}, "
          f"Win Rate: {win_rate:.2f}, Player {player_win} wins!")

    # Store the total reward and average loss for the episode
    all_rewards.append(total_reward)
    if episode_losses:
        average_loss = np.mean(episode_losses)
        all_losses.append(average_loss)
    else:
        all_losses.append(None)  # No loss computed if replay didn't occur

    # Update the target model every few episodes for stability
    if e % agent.update_target_frequency == 0:
        agent.update_target_model()

    # Perform evaluation every 500 episodes
    if (e + 1) % 500 == 0:
        eval_win_rate, eval_avg_reward = evaluate_agent(agent, env, episodes=100)
        evaluation_win_rates.append(eval_win_rate)
        evaluation_episodes.append(e + 1)
        print(f"Evaluation after {e + 1} episodes: Win Rate = {eval_win_rate:.2f}, "
              f"Average Reward = {eval_avg_reward:.2f}")

    # Smooth the rewards and losses for plotting
    smoothed_rewards = smooth_rewards(all_rewards, window)
    smoothed_losses = smooth_rewards([l for l in all_losses if l is not None], window)

    # Dynamically adjust x-axis to the number of episodes processed so far
    ax1.set_xlim(0, len(all_rewards))
    ax2.set_xlim(0, len(all_losses))

    # Dynamically adjust y-axis for the loss graph
    if len(smoothed_losses) > 0:
        min_loss = min(smoothed_losses)
        max_loss = max(smoothed_losses)
        ax2.set_ylim(min_loss - 0.1 * abs(min_loss), max_loss + 0.1 * abs(max_loss))  # Add padding for clarity

    # Update the plot data
    line_rewards.set_xdata(range(len(smoothed_rewards)))
    line_rewards.set_ydata(smoothed_rewards)
    line_losses.set_xdata(range(len(smoothed_losses)))
    line_losses.set_ydata(smoothed_losses)

    # Update the plot titles with win rate and epsilon
    ax1.set_title(f'Total Reward per Episode\nWin Rate: {win_rate:.2f} | Epsilon: {agent.epsilon:.4f}', pad=20)
    ax2.set_title(f'Training Loss per Episode', pad=20)

    plt.draw()
    plt.pause(0.01)  # Short pause to allow updates to be visible


# End of training
total_training_time = time.time() - start_time
print(f"\nTotal Training Time: {total_training_time:.2f} seconds")

# Final report on wins
print(f"\nFinal Results after {episodes} episodes:")
print(f"Player 1 (Opponent) Wins: {player1_wins}")
print(f"Player 2 (DQN Agent) Wins: {player2_wins}")
print(f"DQN Win Rate: {win_rate:.2f}")

# Plot evaluation win rates
plt.figure()
plt.plot(evaluation_episodes, evaluation_win_rates, marker='o')
plt.xlabel('Episode')
plt.ylabel('Evaluation Win Rate')
plt.title('Evaluation Win Rate Over Time')
plt.show()

# Close the environment
env.close()
