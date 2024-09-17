import matplotlib
import matplotlib.pyplot as plt
import numpy as np  # Needed for smoothing
from gym_environment.connect4_env import Connect4Env
from models.dqn_agent import DQNAgent

# Enable interactive mode
plt.ion()

# Function to smooth rewards using moving average
def smooth_rewards(rewards, window=10):
    smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
    return smoothed

if __name__ == "__main__":
    env = Connect4Env()  # Initialize the environment
    agent = DQNAgent()  # Initialize the agent
    episodes = 1000  # Set the number of episodes for training

    all_rewards = []  # To store total rewards for each episode

    fig, ax = plt.subplots()  # Create a plot for live updates
    line, = ax.plot([])  # Initialize the plot line
    ax.set_xlim(0, episodes)  # Set x-axis limit
    ax.set_ylim(-100, 100)  # Set y-axis limit (you can adjust based on expected reward range)
    plt.xlabel('Episode')
    plt.ylabel('Smoothed Total Reward')
    plt.title('Smoothed Total Reward per Episode')

    window = 10  # Moving average window size
    for e in range(episodes):
        state = env.reset()  # Reset the environment at the start of each episode
        done = False
        total_reward = 0  # Track the total reward for this episode

        while not done:
            # The agent chooses an action
            action = agent.act(state)

            # Take the action in the environment
            next_state, reward, done, _ = env.step(action)

            # Store the experience in the replay memory
            agent.remember(state, action, reward, next_state, done)

            # Train the agent by replaying memories
            agent.replay()

            # Move to the next state
            state = next_state

            # Update total reward
            total_reward += reward

        # Store the total reward for this episode
        all_rewards.append(total_reward)

        # Smooth rewards after we have at least "window" number of rewards
        if len(all_rewards) >= window:
            smoothed_rewards = smooth_rewards(all_rewards, window=window)
            line.set_ydata(smoothed_rewards)  # Update the plot line with new data
            line.set_xdata(range(len(smoothed_rewards)))  # Set x-axis values
            ax.relim()  # Recalculate limits
            ax.autoscale_view()  # Rescale the view
            plt.draw()  # Redraw the plot
            plt.pause(0.01)  # Pause for a brief moment to update the plot

        # Print progress
        print(f"Episode {e}/{episodes} - Epsilon: {agent.epsilon:.2f}, Total Reward: {total_reward}")

    # Final display of the plot after training completes
    plt.ioff()  # Disable interactive mode
    plt.show()  # Show the final plot after all episodes are done
