import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        
        # Input layer: 6x7 grid (for Connect 4 board)
        self.fc1 = nn.Linear(6 * 7, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 7)  # Output 7 values (one for each column)

    def forward(self, x):
        # Flatten the input: (batch_size, 6, 7) -> (batch_size, 6 * 7)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Output is Q-values for each column
