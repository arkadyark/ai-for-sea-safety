import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModel(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional, use_lstm):
        super(AttentionModel, self).__init__()
        if use_lstm:
            self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=bidirectional)
        else:
            self.rnn = nn.GRU(input_size, hidden_size, bidirectional=bidirectional)
        if bidirectional:
            hidden_size *= 2
        self.word_weight = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.word_bias = nn.Parameter(torch.randn(1, hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(hidden_size, 1))
        self.word_weight.data.normal_(0, 0.05)
        self.word_bias.data.normal_(0, 0.05)
        self.context_weight.data.normal_(0, 0.05)

    def forward(self, sequence, hidden_state):
        f_output, h_output = self.rnn(sequence, hidden_state)
        word_weight = self.word_weight.unsqueeze(0).expand(f_output.shape[0], -1, -1)
        output = torch.bmm(f_output, word_weight)
        word_bias = self.word_bias.expand(output.shape[0], output.shape[1], -1)
        output = output + word_bias
        output = torch.tanh(output)
        context_weight = self.context_weight.unsqueeze(0).expand(output.shape[0], -1, -1)
        output = torch.tanh(torch.bmm(output, context_weight)).squeeze().permute(1, 0)
        output = F.softmax(output, -1)
        output = f_output * output.permute(1, 0).unsqueeze(2).expand_as(f_output)
        output = torch.sum(output, 0).unsqueeze(0)

        return output, h_output
