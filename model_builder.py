import torch
from torch import nn
import numpy as np

device = 'cpu'

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,  output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out