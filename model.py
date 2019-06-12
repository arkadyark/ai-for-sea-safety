import torch
import torch.nn as nn
from attention_model import AttentionModel

class SafetyModel(nn.Module):
    def __init__(self, input_dim, second_hidden_size, minute_hidden_size, rnn_layers, batch_size, bidirectional, use_lstm):
        super(SafetyModel, self).__init__()
        self.batch_size = batch_size
        self.rnn_layers = rnn_layers
        self.use_lstm = use_lstm
        self.second_hidden_size = second_hidden_size
        self.minute_hidden_size = minute_hidden_size
        self.second_att_net = AttentionModel(input_dim, second_hidden_size, bidirectional, use_lstm)
        if bidirectional:
            second_hidden_size *= 2
        self.minute_att_net = AttentionModel(second_hidden_size, minute_hidden_size, bidirectional, use_lstm)
        if bidirectional:
            minute_hidden_size *= 2
        self.fc = nn.Linear(minute_hidden_size, 2)
        self.init_hidden()

    def init_hidden(self):
        if self.use_lstm:
            self.second_hidden_state = (torch.zeros(2, self.batch_size, self.second_hidden_size), torch.zeros(2, self.batch_size, self.second_hidden_size))
            self.minute_hidden_state = (torch.zeros(2, self.batch_size, self.minute_hidden_size), torch.zeros(2, self.batch_size, self.minute_hidden_size))
            if torch.cuda.is_available():
                self.second_hidden_state = (self.second_hidden_state[0].cuda(), self.second_hidden_state[1].cuda())
                self.minute_hidden_state = (self.minute_hidden_state[0].cuda(), self.minute_hidden_state[1].cuda())
        else:
            self.second_hidden_state = torch.zeros(2, self.batch_size, self.second_hidden_size)
            self.minute_hidden_state = torch.zeros(2, self.batch_size, self.minute_hidden_size)
            if torch.cuda.is_available():
                self.second_hidden_state = self.second_hidden_state.cuda()
                self.minute_hidden_state = self.minute_hidden_state.cuda()

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
