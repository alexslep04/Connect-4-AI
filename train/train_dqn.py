import matplotlib.pyplot as plt
import numpy as np
from gym_environment.connect4_env import Connect4Env
from models.dqn_agent import DQNAgent

# Enable interactive mode
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
    episodes = 1000  # Set the number of episodes for training

    all_rewards = []  # To store total rewards for each episode
    all_invalid_actions = []  # To store invalid actions per episode

    # Create a plot for live updates
    fig, ax = plt.subplots()
    line_rewards, = ax.plot([], label='Smoothed Total Reward')  # Line for rewards
    line_invalid_actions, = ax.plot([], label='Invalid Actions', color='r')  # Line for invalid actions
    ax.set_xlim(0, episodes)  # Set x-axis limit
    ax.set_ylim(-100, 100)  # Set y-axis limit (you can adjust based on expected range)
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Total Reward / Invalid Actions')
    plt.title('Total Reward and Invalid Actions per Episode')
    plt.legend()  # Show the legend

    window = 10  # Moving average window size for rewards
    
    for e in range(episodes):
        state = env.reset()  # Reset the environment at the start of each episode
        done = False
        total_reward = 0  # Track the total reward for this episode

    while not done:
        action = agent.act(state)  # Agent chooses action
        next_state, reward, done, _ = env.step(action)  # Take action in the environment

        agent.remember(state, action, reward, next_state, done)  # Store experience
        agent.replay()  # Train on a minibatch from memory

        state = next_state  # Move to the next state
        total_reward += reward  # Accumulate total reward

    # Log the total reward for the episode
    print(f"Episode {e}/{episodes} - Epsilon: {agent.epsilon:.4f}, Total Reward: {total_reward}")

    # Update the target model every few episodes for stability
    if e % agent.update_target_frequency == 0:
        agent.update_target_model()

        # Smooth the rewards for plotting
        smoothed_rewards = smooth_rewards(all_rewards, window)

        # Update the plot after each episode
        line_rewards.set_xdata(range(len(smoothed_rewards)))
        line_rewards.set_ydata(smoothed_rewards)
        line_invalid_actions.set_xdata(range(len(all_invalid_actions)))
        line_invalid_actions.set_ydata(all_invalid_actions)

        ax.relim()  # Recalculate limits
        ax.autoscale_view()  # Rescale the view

        # Redraw the plot
        plt.draw()
        plt.pause(0.01)  # Short pause to allow updates to be visible

    # Disable interactive mode and show the final plot
    plt.ioff()
    plt.show()
