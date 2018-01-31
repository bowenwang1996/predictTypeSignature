import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size) #bidirectional
        self.rnn = nn.LSTM(hidden_size, hidden_size/2, bidirectional=True)

    def forward(self, input, hidden):
        input_len = input.size()[0]
        embedded = self.embedding(input).view(input_len, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.rnn(output, hidden)
        return output, hidden

    def initHidden(self):
        return (Variable(torch.zeros(2, 1, self.hidden_size/2))
                , Variable(torch.zeros(2, 1, self.hidden_size/2)))

class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, n_layers=1):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        input_len = input.size()[0]
        embedded = self.embedding(input).view(input_len, 1, -1)
        output = embedded
        for i in range(self.n_layers):
            output, hidden = self.rnn(embedded, hidden)
        output = F.log_softmax(self.out(output[0]))
        return output, hidden

class AttnDecoder(nn.Module):
    def __init__(self, output_size, hidden_size, n_layers=1):
        super(AttnDecoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.rnn = nn.LSTM(hidden_size, hidden_size)
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.attn_combine = nn.Linear(2 * hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    '''
    hidden: (1 * 1 * hidden_size, 1 * 1 * hidden_size)
    encoder_outputs: L * hidden_size
    '''
    def forward(self, input, hidden, encoder_outputs):
        input_len = input.size(0)
        embedded = self.embedding(input).view(input_len, 1, -1)
        attn_scores = torch.matmul(encoder_outputs, self.attn(embedded.view(-1)))
        attn_scores = F.softmax(attn_scores)
        context_vector = torch.matmul(attn_scores, encoder_outputs)
        output = self.attn_combine(torch.cat((embedded.view(-1), context_vector), 0))
        output = F.tanh(output)
        for i in range(self.n_layers):
            output, hidden = self.rnn(embedded, hidden)
        output = F.log_softmax(self.out(output[0]))
        return output, hidden
