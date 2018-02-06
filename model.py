import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout_p=0.0):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size/2, num_layers=n_layers, batch_first=True, dropout=dropout_p, bidirectional=True)

    '''
    input: B * T, sorted in decreasing length
    output: B * T * H
    '''
    def forward(self, input, lengths, hidden):
        max_len = input.size(1)
        batch_size = input.size(0)
        embed_input = self.embedding(input).view(batch_size, max_len, -1)
        output = pack_padded_sequence(embed_input, lengths, batch_first=True)
        output, hidden = self.lstm(output, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output, hidden

    def initHidden(self, batch_size):
        hidden = (Variable(torch.zeros(2 * self.n_layers, batch_size, self.hidden_size/2)),
                  Variable(torch.zeros(2 * self.n_layers, batch_size, self.hidden_size/2))
                  )
        return hidden
                                                                    
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
    def __init__(self, output_size, hidden_size, n_layers=1, dropout_p=0.0):
        super(AttnDecoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True, num_layers=n_layers)
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(dropout_p)
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    '''
    input : B * T (T should be 1 since we use an outer loop to process target)
    encoder_outputs: B * T * H
    '''
    def forward(self, input, hidden, encoder_outputs):
        input_len = input.size(1)
        batch_size = input.size(0)
        embedded = self.embedding(input).view(batch_size, input_len, -1)
        transformed_input = self.attn(embedded.view(batch_size, -1))
        attn_scores = torch.bmm(encoder_outputs, transformed_input.unsqueeze(2)) # B*T*1
        attn_scores = F.softmax(attn_scores, dim=1).transpose(1, 2)
        context = torch.bmm(attn_scores, encoder_outputs).squeeze(1)
        output, hidden = self.lstm(embedded, hidden)
        output = self.attn_combine(torch.cat((output.view(batch_size, -1), context), 1))
        output = F.tanh(output) #B*H
        output = F.log_softmax(self.out(output), dim=1)
        return output, hidden
        
    def initHidden(self, batch_size):
        hidden = (Variable(torch.zeros(1, batch_size, self.hidden_size)),
                  Variable(torch.zeros(1, batch_size, self.hidden_size))
                  )
        return hidden
