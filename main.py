import torch
import numpy as np
import pygame
from gym_environment.connect4_env import Connect4Env
from pygame_ui.connect4_game import Connect4Game, SQUARE_SIZE  # Import SQUARE_SIZE
from models.dqn_agent import DQN

# Load the trained model
def load_model(path, device):
    model = DQN().to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()  # Set the model to evaluation mode (important for inference)
    return model

# Get the agent's action based on the current game state
def agent_act(model, state, device):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)  # Shape: [1, 6, 7]
    with torch.no_grad():
        q_values = model(state_tensor)  # Output shape: [1, 7]
    valid_actions = [c for c in range(7) if state[0][c] == 0]  # Get valid actions (non-full columns)
    
    # Choose the action with the highest Q-value among valid actions
    q_values_np = q_values.cpu().numpy()[0]
    best_action = max(valid_actions, key=lambda c: q_values_np[c])
    return best_action

def human_vs_agent():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained DQN agent model
    model_path = 'dqn_model_50k_updated.pth'  # Update with your model path if different
    model = load_model(model_path, device)

    env = Connect4Env(render_mode=True)  # Enable rendering for visualization
    state = env.reset()
    done = False
    current_player = 1  # Human starts as Player 1, agent is Player 2

    print("You are Player 1 (Red), the DQN Agent is Player 2 (Yellow).")

    while not done:
        env.render()

        # Handle Pygame events (to prevent freezing)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            # Human's turn - Player 1 (Human)
            if event.type == pygame.MOUSEBUTTONDOWN and current_player == 1 and not done:  
                posx = event.pos[0]
                action = posx // SQUARE_SIZE  # Get the column from mouse click

                valid_actions = [c for c in range(7) if state[0][c] == 0]
                if action not in valid_actions:
                    print(f"Invalid move. Column {action} is full or not allowed.")
                    continue

                # Perform the action for Player 1 (Human)
                next_state, reward, done, _ = env.step(action)
                state = next_state

                print(f"Human (Player 1) clicked column {action}")

                if done:
                    env.render()
                    if reward == 1:
                        print("Player 1 (You) win!")
                    elif reward == -1:
                        print("Player 2 (Agent) wins!")
                    else:
                        print("It's a draw!")
                    break

                
    env.close()

if __name__ == "__main__":
    human_vs_agent()
