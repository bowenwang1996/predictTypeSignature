import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from collections import Counter

from utils import *
from batch import Batch
from prepare_data import start_token, end_token
from beam import Beam

use_cuda = torch.cuda.is_available()

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
        super().__init__(input_size, embed_size, hidden_size, n_layers, dropout_p)
        self.embedding = nn.Embedding(input_size, embed_size, padding_idx=0)

    def embed(self, input):
        l = input.size(2)
        embedded = self.embedding(input).sum(dim=2)/l
        embedded = F.tanh(embedded)
        return embedded
    
    def forward(self, input, lengths, sort_index, inv_sort_index, hidden):
        batch_size = input.size(0)
        input_len = input.size(1)
        embedded = self.embed(input).view(batch_size, input_len, -1)
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
    def __init__(self, vocab_size, type_vocab_size, embed_size, hidden_size,
                 n_layers=1, dropout_p=0.0, max_oov=50):
        super(ContextAttnDecoder, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.type_vocab_size = type_vocab_size
        self.max_oov = max_oov
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.rnn = nn.LSTM(embed_size, hidden_size, batch_first=True, num_layers=n_layers)
        self.attn = Attn(hidden_size)
        self.context_attn = Attn(hidden_size)
        self.gen_prob = nn.Linear(3 * hidden_size + embed_size, 1)
        self.sigmoid = SigmoidBias(1)
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden_size, type_vocab_size)

    def embed(self, input):
        l = input.size(2)
        embedded = self.embedding(input)
        embedded = F.tanh(embedded.sum(dim=2)/l)
        return embedded
        

    def forward(self, input, hidden, encoder_outputs, context_encoder_outputs, context_input):
        batch_size = input.size(0)
        input_len = input.size(1)
        context_input_len = context_input.size(1)
        embedded = self.embed(input).view(batch_size, input_len, -1)
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

        p_copy = Variable(torch.zeros(batch_size, self.type_vocab_size + self.max_oov))
        if use_cuda:
            p_copy = p_copy.cuda()
            batch_indices = batch_indices.cuda()
        word_indices = context_input.view(-1)
        linearized_indices = Variable(batch_indices) * (self.type_vocab_size + self.max_oov) + word_indices
        value_to_add = context_attn_scores.view(-1)
        p_copy.put_(linearized_indices, value_to_add, accumulate=True)
        output_prob = p_gen * p_vocab + (1 - p_gen) * p_copy

        return torch.log(output_prob.clamp(min=1e-10)), hidden

class Model(nn.Module):

    def __init__(self, input_vocab, target_vocab, 
                 embed_size, hidden_size, train_batch_size,
                 eval_batch_size, criterion,
                 n_layers=1, dropout_p=0.0):
        super().__init__()
        input_vocab_size = input_vocab.n_word
        target_vocab_size = target_vocab.n_word
        target_type_vocab_size = target_vocab.n_type
        self.encoder = Encoder(input_vocab_size, embed_size, hidden_size,
                               n_layers=n_layers, dropout_p=dropout_p)
        self.decoder = ContextAttnDecoder(target_vocab_size, target_type_vocab_size,
                                          embed_size, hidden_size,
                                          n_layers=n_layers, dropout_p=dropout_p)
        self.context_encoder = ContextEncoder(target_vocab_size, embed_size,
                                              hidden_size, n_layers=n_layers,
                                              dropout_p=dropout_p)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.batch = Batch(train_batch_size, input_vocab, target_vocab)
        self.criterion = criterion
        self.combiner = nn.Linear(hidden_size * 2, hidden_size)

    def combine(self, hidden1, hidden2):
        combined = (F.tanh(self.combiner(torch.cat((hidden1[0], hidden2[0]), 2))),
                    F.tanh(self.combiner(torch.cat((hidden1[1], hidden2[0]), 2)))
                    )
        return combined

    def get_token_idx(self, idx, trainInfo):
        if idx < self.batch.target_vocab.n_type:
            token = self.batch.target_vocab.idx_to_type[idx]
            indices = self.batch.target_vocab.type_to_tokens[token]
        elif idx in trainInfo.idx_oov_dict:
            token = trainInfo.idx_oov_dict[idx]
            indices = trainInfo.oov_type_dict[token]
        else:
            indices = self.target_vocab.compute_embed_idx(token)
        if not indices:
            print(idx)
            print(token)
            raise ValueError("indices cannot be empty")
        return indices
        

    def forward(self, trainInfo, is_dev=False, is_test=False, max_length=30):
        batch_size = self.batch.batch_size
        encoder_hidden = self.encoder.initHidden(batch_size)
        context_encoder_hidden = self.context_encoder.initHidden(batch_size)
        if use_cuda:
            encoder_hidden = (encoder_hidden[0].cuda(), encoder_hidden[1].cuda())
            context_encoder_hidden = (context_encoder_hidden[0].cuda(),
                                      context_encoder_hidden[1].cuda()
                                      )
        target_len = trainInfo.target_variable.size(1)

        encoder_outputs, encoder_hidden = self.encoder(trainInfo.input_variable,
                                                       trainInfo.input_lengths,
                                                       encoder_hidden)
        #context_variable = self.batch.unk_batch(trainInfo.context_variable)  # for feeding to context encoder
        context_encoder_outputs, context_encoder_hidden = self.context_encoder(trainInfo.context_variable, trainInfo.context_lengths, trainInfo.context_sort_index, trainInfo.context_inv_index, context_encoder_hidden)

        encoder_hidden = (encoder_hidden[0].view(1, batch_size, -1),
                          encoder_hidden[1].view(1, batch_size, -1)
                          )
        context_encoder_hidden = (context_encoder_hidden[0].view(1, batch_size, -1),
                                  context_encoder_hidden[1].view(1, batch_size, -1)
                                  )
        decoder_hidden = self.combine(encoder_hidden, context_encoder_hidden)
        '''
        if is_test or is_dev:
            beam_size = 1
            beams = [ Beam(beam_size,
                           start_token,
                           start_token,
                           end_token,
                           cuda=use_cuda)
                      for _ in range(batch_size)
                      ]
            decoder_hiddens = [decoder_hidden for _ in range(beam_size)]
            for i in range(max_length):
                if all([b.done() for b in beams]):
                    break
                decoder_in = torch.cat([b.get_current_state() for b in beams], 0)\
                                  .view(batch_size, -1)\
                                  .transpose(0, 1)\
                                  .unsqueeze(2)
                decoder_in = Variable(self.batch.unk_batch(decoder_in))
                word_probs = []
                for j in range(beam_size):
                    decoder_out, decoder_hidden = self.decoder(decoder_in[j],
                                                               decoder_hiddens[j],
                                                               encoder_outputs,
                                                               context_encoder_outputs,
                                                               trainInfo.context_type_variable)
                    decoder_hiddens[j] = decoder_hidden
                    word_probs.append(decoder_out)
                word_probs = torch.cat(word_probs, 0).data
                for j, b in enumerate(beams):
                    b.advance(word_probs[j:beam_size*batch_size:batch_size, :])
            decoded_tokens = []
            for b in beams:
                _, ks = b.sort_finished(minimum=b.n_best)
                hyps = []
                for i, (times, k) in enumerate(ks[:b.n_best]):
                    hyp = b.get_hyp(times, k)
                    hyps.append(hyp)
                decoded_tokens.append(hyps[0])
            if is_test:
                return decoded_tokens
        '''
        if is_test or is_dev:
            decoder_in = Variable(torch.LongTensor(batch_size, 1, 1).fill_(start_token))
            # -1 for not taken
            decoded_tokens = torch.LongTensor(batch_size, max_length).fill_(-1)
            eos_tensor = torch.LongTensor(batch_size).fill_(-1)
            if use_cuda:
                decoder_in = decoder_in.cuda()
                decoded_tokens = decoded_tokens.cuda()
                eos_tensor = eos_tensor.cuda()
            for i in range(max_length):
                decoder_out, decoder_hidden = self.decoder(decoder_in, decoder_hidden, encoder_outputs, context_encoder_outputs, trainInfo.context_type_variable)
                _, topi = decoder_out.data.topk(1, dim=1)
                pred = topi[:, 0]
                decoded_tokens[:, i] = pred
                decoded_tokens[:, i].masked_fill_(eos_tensor > -1, -1)
                end_mask = pred == end_token
                eos_tensor.masked_fill_(end_mask & (eos_tensor == -1), i+1)
                next_in = [[self.get_token_idx(x, trainInfo)] for x in topi.squeeze(1).tolist()]
                pad_nested_to_len(next_in, 1)
                decoder_in = Variable(torch.LongTensor(next_in))
                    
                if use_cuda:
                    decoder_in = decoder_in.cuda()
            if is_test:
                return decoded_tokens
        if is_dev or not is_test:
            loss = 0.0
            decoder_input = Variable(torch.LongTensor(batch_size, 1, 1).fill_(start_token))
            length_tensor = torch.LongTensor(trainInfo.target_lengths)
            # at the begining, no sequence has ended so we use -1 as its end index
            eos_tensor = torch.LongTensor(batch_size).fill_(-1)
            if use_cuda:
                decoder_input = decoder_input.cuda()
                length_tensor = length_tensor.cuda()
                eos_tensor = eos_tensor.cuda()

            for i in range(target_len):
                decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                              decoder_hidden,
                                                              encoder_outputs,
                                                              context_encoder_outputs,
                                                              trainInfo.context_type_variable)
                cur_loss = self.criterion(decoder_output, trainInfo.target_type_variable[:, i])
                loss_mask = (length_tensor > i) & (eos_tensor == -1)
                loss_mask = Variable(loss_mask).cuda() if use_cuda else Variable(loss_mask)
                loss += torch.masked_select(cur_loss, loss_mask).sum()
                _, topi = decoder_output.data.topk(1, dim=1)
                next_in = [[self.get_token_idx(x, trainInfo)] for x in topi.squeeze(1).tolist()]
                pad_nested_to_len(next_in, 1)
                end_mask = topi.squeeze(1) == end_token
                end_mask = end_mask & (eos_tensor == -1)
                eos_tensor.masked_fill_(end_mask, i)
                decoder_input = Variable(torch.LongTensor(next_in))
                if use_cuda:
                    decoder_input = decoder_input.cuda()
            if not is_dev:
                return loss
            return loss, decoded_tokens
