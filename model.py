import torch
import torch.nn as nn
from attention_model import AttentionModel

class SafetyModel(nn.Module):
    def __init__(self, input_dim, second_hidden_size, minute_hidden_size, lstm_layers, batch_size, bidirectional=True):
        super(SafetyModel, self).__init__()
        self.batch_size = batch_size
        self.lstm_layers = lstm_layers
        self.second_hidden_size = second_hidden_size
        self.minute_hidden_size = minute_hidden_size
        self.second_att_net = AttentionModel(input_dim, second_hidden_size, bidirectional)
        if bidirectional:
            second_hidden_size *= 2
        self.minute_att_net = AttentionModel(second_hidden_size, minute_hidden_size, bidirectional)
        if bidirectional:
            minute_hidden_size *= 2
        self.fc = nn.Linear(minute_hidden_size, 2)
        self.init_hidden()

    def init_hidden(self):
        self.second_hidden_state = (torch.zeros(2, self.batch_size, self.second_hidden_size), torch.zeros(2, self.batch_size, self.second_hidden_size))
        self.minute_hidden_state = (torch.zeros(2, self.batch_size, self.minute_hidden_size), torch.zeros(2, self.batch_size, self.minute_hidden_size))

    def forward(self, sequence):
        output_list = []
        sequence = sequence.permute(1, 0, 2, 3)
        for minute in sequence:
            output, self.second_hidden_state = self.second_att_net(minute.permute(1, 0, 2), self.second_hidden_state)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, self.minute_hidden_state = self.minute_att_net(output, self.minute_hidden_state)
        output = output.squeeze(0)
        output = self.fc(output)
        return output
