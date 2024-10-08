from gym_environment.connect4_env import Connect4Env

if __name__ == "__main__":
    env = Connect4Env(render_mode=False)
    state = env.reset()
    
    done = False
    while not done:
        # Sample a random action (just for testing)
        action = env.action_space.sample()  # Randomly pick a column
        
        # Take the action
        next_state, reward, done, _ = env.step(action)
        
        print(f"Action: {action}, Reward: {reward}, Done: {done}")
    
    env.close()
