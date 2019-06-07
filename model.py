import torch
import torch.nn as nn

class SafetyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, lstm_layers, batch_size):
        super(SafetyModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, lstm_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.batch_size = batch_size
        self.lstm_layers = lstm_layers
        self.hidden_dim = hidden_dim

    def init_hidden(self):
        return (torch.randn(self.lstm_layers, self.batch_size, self.hidden_dim),
                torch.randn(self.lstm_layers, self.batch_size, self.hidden_dim))

    def forward(self, sequence, sequence_lengths):
        self.hidden = self.init_hidden()
        a = nn.utils.rnn.pack_padded_sequence(sequence, sequence_lengths, batch_first=True, enforce_sorted=False)
        b, self.hidden = self.lstm(a, self.hidden)
        c, _ = nn.utils.rnn.pad_packed_sequence(b, batch_first=True)
        d = torch.diagonal(c[:, sequence_lengths - 1, :], 0).transpose(0, 1)
        e = self.linear(d)
        f = self.sigmoid(e)
        g = f.view(self.batch_size)
        return g
