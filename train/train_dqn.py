import matplotlib
matplotlib.use('TkAgg')  # Ensure the backend supports displaying plots
import matplotlib.pyplot as plt
from gym_environment.connect4_env import Connect4Env
from models.dqn_agent import DQNAgent

if __name__ == "__main__":
    env = Connect4Env()  # Initialize the environment
    agent = DQNAgent()  # Initialize the agent
    episodes = 1000  # Set the number of episodes for training

    all_rewards = []  # To store total rewards for each episode

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

        # Every few episodes, update the target model to improve stability
        if e % agent.update_target_frequency == 0:
            agent.update_target_model()

        # Print progress
        print(f"Episode {e}/{episodes} - Epsilon: {agent.epsilon:.2f}, Total Reward: {total_reward}")

    # After training, plot the total rewards
    plt.plot(all_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Total Reward per Episode')

    # Ensure the plot pops up
    plt.show()  # This should display the plot in a new window
