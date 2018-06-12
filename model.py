import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import Counter

#import time
from utils import *

use_cuda = torch.cuda.is_available()
#use_cuda = False

class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout_p=0.0):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size//2, num_layers=n_layers, batch_first=True, dropout=dropout_p, bidirectional=True)

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
        hidden = (Variable(torch.zeros(2 * self.n_layers, batch_size, self.hidden_size//2)),
                  Variable(torch.zeros(2 * self.n_layers, batch_size, self.hidden_size//2))
                  )
        return hidden

class ContextEncoder(Encoder):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout_p=0.0):
        super(ContextEncoder, self).__init__(input_size, embed_size, hidden_size, n_layers, dropout_p)
    def forward(self, input, lengths, sort_index, inv_sort_index, hidden):
        batch_size = input.size(0)
        input_len = input.size(1)
        embedded = self.embedding(input).view(batch_size, input_len, - 1)
        output = pack_padded_sequence(embedded[sort_index], lengths, batch_first=True)
        output, hidden = self.lstm(output, hidden)
        output, _ = pad_packed_sequence(output, batch_first=True)
        return output[inv_sort_index], hidden

class Decoder(nn.Module):
    def __init__(self, output_size, embed_size, hidden_size, n_layers=1):
        super(Decoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size)
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
    def __init__(self, output_size, embed_size, hidden_size, n_layers=1, dropout_p=0.0):
        super(AttnDecoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(output_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True, num_layers=n_layers)
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(dropout_p)
        self.attn = Attn(hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        '''
        input : B * T (T should be 1 since we use an outer loop to process target)
        encoder_outputs: B * T * H
        '''

        input_len = input.size(1)
        batch_size = input.size(0)
        embedded = self.embedding(input).view(batch_size, input_len, -1)
        '''
        transformed_input = self.attn(embedded.view(batch_size, -1))
        attn_scores = torch.bmm(encoder_outputs, transformed_input.unsqueeze(2)) # B*T*1
        attn_scores = F.softmax(attn_scores, dim=1).transpose(1, 2)
        context = torch.bmm(attn_scores, encoder_outputs).squeeze(1)
        '''
        output, hidden = self.lstm(embedded, hidden)
        output = self.attn(output, encoder_outputs)
        '''
        output = self.attn_combine(torch.cat((output.view(batch_size, -1), context), 1))
        output = F.tanh(output) #B*H
        '''
        output = F.log_softmax(self.out(output), dim=1)
        return output, hidden

    def initHidden(self, batch_size):
        hidden = (Variable(torch.zeros(1, batch_size, self.hidden_size)),
                  Variable(torch.zeros(1, batch_size, self.hidden_size))
                  )
        return hidden

class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        self.attn = nn.Linear(hidden_size, hidden_size)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, decoder_out, encoder_outputs, score_only=False, context_only=False):
        '''
        decoder_out: B * 1 * H
        encoder_outputs: B * T * H
        returns: transformed decoder output and attention scores
        '''
        decoder_out = self.attn(decoder_out)
        attn_scores = torch.bmm(encoder_outputs, decoder_out.transpose(1, 2))
        attn_scores = F.softmax(attn_scores, dim=1).transpose(1, 2)
        if score_only:
            return attn_scores.squeeze(1)
        context = torch.bmm(attn_scores, encoder_outputs).squeeze(1) #B*H
        if context_only:
            return context
        output = F.tanh(self.attn_combine(torch.cat((decoder_out.squeeze(1), context), 1)))
        return output

# according to https://discuss.pytorch.org/t/does-nn-sigmoid-have-bias-parameter/10561/2
class SigmoidBias(nn.Module):
    def __init__(self, output_size):
        super(SigmoidBias, self).__init__()
        self.bias = nn.Parameter(torch.Tensor(output_size))
        self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):
        output = input + self.bias.unsqueeze(0).expand_as(input)
        output = F.sigmoid(output)
        return output

class ContextAttnDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1, dropout_p=0.0, max_oov=50):
        super(ContextAttnDecoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.max_oov = max_oov
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True, num_layers=n_layers)
        self.attn = Attn(hidden_size)
        self.context_attn = Attn(hidden_size)
        self.gen_prob = nn.Linear(3 * hidden_size + embed_size, 1)
        self.sigmoid = SigmoidBias(1)
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden_size, vocab_size)


    def forward(self, input, hidden, encoder_outputs, context_encoder_outputs, context_input):
        batch_size = input.size(0)
        input_len = input.size(1)
        context_input_len = context_input.size(1)
        embedded = self.embedding(input).view(batch_size, input_len, -1) # B * 1 * H
        output, hidden = self.rnn(embedded, hidden)
        context = self.attn(output, encoder_outputs, context_only=True)
        context_attn_scores = self.context_attn(output, context_encoder_outputs, score_only=True)
        context_context = torch.bmm(context_attn_scores.unsqueeze(1), context_encoder_outputs).squeeze(1)
        p_gen = self.sigmoid(self.gen_prob(torch.cat((context, context_context, output.view(batch_size, -1), embedded.view(batch_size, -1)), 1)))

        context_length = (context_input > 0).long().sum(1).unsqueeze(1)
        p_gen = p_gen.clone().masked_fill_(context_length == 0, 1)

        p_vocab = F.softmax(self.out(output.squeeze(1)), dim=1) # B * O
        oov_var = Variable(torch.zeros(batch_size, self.max_oov))
        if use_cuda:
            oov_var = oov_var.cuda()
        p_vocab = torch.cat((p_vocab, oov_var), 1)

        batch_indices = torch.arange(start=0, end=batch_size).long()
        batch_indices = batch_indices.expand(context_input_len, batch_size).transpose(1, 0).contiguous().view(-1)

        p_copy = Variable(torch.zeros(batch_size, self.vocab_size + self.max_oov))
        if use_cuda:
            p_copy = p_copy.cuda()
            batch_indices = batch_indices.cuda()
        word_indices = context_input.view(-1)
        linearized_indices = Variable(batch_indices) * (self.vocab_size + self.max_oov) + word_indices
        value_to_add = context_attn_scores.view(-1)
        p_copy.put_(linearized_indices, value_to_add, accumulate=True)
        output_prob = p_gen * p_vocab + (1 - p_gen) * p_copy

        return torch.log(output_prob.clamp(min=1e-10)), hidden
