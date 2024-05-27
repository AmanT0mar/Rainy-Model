import torch
from torch import nn
from torch.autograd import Variable

device = 'cpu'

class LSTMModel(nn.Module):
    """
        Creates an LSTM Neural Network.


    Args:
    input_size: An integer indicating size of input units.
    hidden_size: An integer indicating number of hidden units between layers.
    num_layers: An integer indicating number of layers of LSTM.
    output_size: An integer indicating number of output units.
    """

    def __init__(self, input_size, hidden_size, num_layers,  output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

        self.relu = nn.ReLU()

        self.fc_1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)

        self.fc = nn.Linear(in_features=hidden_size, out_features=output_size)

    def forward(self, x):
        batch_size = x.size(0)

        h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))
        c0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size).to(device))

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc_1(out[:, -1, :])

        out = self.relu(out)

        out = self.fc(out)

        return self.relu(out)