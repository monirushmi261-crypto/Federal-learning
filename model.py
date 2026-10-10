import torch
import torch.nn as nn
import torch.nn.functional as F

class ClassifierNN(nn.Module):
    def __init__(self, input_dim=10):
        super(ClassifierNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, 16)
        self.out = nn.Linear(16, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
