import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

#import time
from utils import *
from prepare_data import arrow_token, start_token, unk_token
from type_signatures import Tree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        hidden = (torch.zeros(2 * self.n_layers, batch_size, self.hidden_size//2, requires_grad=True).to(device),
                  torch.zeros(2 * self.n_layers, batch_size, self.hidden_size//2, requires_grad=True).to(device)
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
        output = F.log_softmax(self.out(output[0]), dim=1)
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
        hidden = (torch.zeros(1, batch_size, self.hidden_size, requires_grad=True),
                  torch.zeros(1, batch_size, self.hidden_size, requires_grad=True)
                  )
        return hidden

class Attn(nn.Module):
    def __init__(self, hidden_size, use_coverage=False):
        super(Attn, self).__init__()
        if use_coverage:
            self.attn = nn.Linear(hidden_size, hidden_size + 1)
        else:
            self.attn = nn.Linear(hidden_size, hidden_size)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, decoder_out, encoder_outputs, coverage=None, score_only=False, context_only=False):
        '''
        decoder_out: B * 1 * H
        encoder_outputs: B * T * H
        returns: transformed decoder output and attention scores
        '''
        decoder_out = self.attn(decoder_out)
        if coverage is not None:
            concat_tensor = torch.cat((encoder_outputs, coverage.unsqueeze(2)), 2)
            attn_scores = torch.bmm(concat_tensor, decoder_out.transpose(1, 2))
        else:
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
        self.context_attn = Attn(hidden_size, use_coverage=True)
        self.gen_prob = nn.Linear(3 * hidden_size + embed_size, 1)
        self.sigmoid = SigmoidBias(1)
        self.dropout_p = dropout_p
        self.dropout = nn.Dropout(dropout_p)
        self.out = nn.Linear(hidden_size, vocab_size)


    def forward(self, input, hidden, encoder_outputs, context_encoder_outputs, context_input, coverage):
        batch_size = input.size(0)
        input_len = input.size(1)
        context_input_len = context_input.size(1)
        embedded = self.embedding(input).view(batch_size, input_len, -1) # B * 1 * H
        output, hidden = self.rnn(embedded, hidden)
        context = self.attn(output, encoder_outputs, context_only=True)
        context_attn_scores = self.context_attn(output, context_encoder_outputs, coverage=coverage, score_only=True)
        context_context = torch.bmm(context_attn_scores.unsqueeze(1), context_encoder_outputs).squeeze(1)
        p_gen = self.sigmoid(self.gen_prob(torch.cat((context, context_context, output.view(batch_size, -1), embedded.view(batch_size, -1)), 1)))

        context_length = (context_input > 0).long().sum(1).unsqueeze(1)
        p_gen = p_gen.clone().masked_fill_(context_length == 0, 1)

        p_vocab = F.softmax(self.out(output.squeeze(1)), dim=1) # B * O
        oov_var = torch.zeros(batch_size, self.max_oov, requires_grad=True).to(device)
        p_vocab = torch.cat((p_vocab, oov_var), 1)

        batch_indices = torch.arange(start=0, end=batch_size).long()
        batch_indices = batch_indices.expand(context_input_len, batch_size).transpose(1, 0).contiguous().view(-1).to(device)

        p_copy = torch.zeros(batch_size, self.vocab_size + self.max_oov, requires_grad=True).to(device)
        word_indices = context_input.view(-1)
        linearized_indices = batch_indices * (self.vocab_size + self.max_oov) + word_indices
        value_to_add = context_attn_scores.view(-1)
        p_copy.put_(linearized_indices, value_to_add, accumulate=True)
        output_prob = p_gen * p_vocab + (1 - p_gen) * p_copy

        return torch.log(output_prob.clamp(min=1e-10)), hidden, context_attn_scores

class BaseType(nn.Module):
    '''
    submodule for handling base types
    '''
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1, dropout_p=0.0):
        super(BaseType, self).__init__()
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=n_layers, dropout=dropout_p, batch_first=True)
        self.parent_proj = nn.Linear(hidden_size, hidden_size)
        self.frat_proj = nn.Linear(hidden_size, hidden_size)
        self.attn = Attn(hidden_size)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, parent_hidden, frat_hidden, encoder_outputs):
        '''
        input should already be embedded here (done in Type module)
        '''
        hidden = (F.tanh(self.parent_proj(parent_hidden[0]) + self.frat_proj(frat_hidden[0])),
                  F.tanh(self.parent_proj(parent_hidden[1]) + self.frat_proj(frat_hidden[1])))
        output, hidden = self.rnn(input, hidden)
        output = self.attn(output, encoder_outputs)
        output = self.out(output.squeeze(1))
        output = F.log_softmax(output, dim=1)
        return output, hidden

class ArrowType(nn.Module):
    '''
    submodule for handling arrow types
    '''
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1, dropout_p=0.0):
        #self.arrow_embedding = nn.Parameter(torch.Tensor(embed_size))
        #self.arrow_embedding.data.uniform_(-0.1, 0.1)
        super(ArrowType, self).__init__()
        self.parent_proj = nn.Linear(hidden_size, hidden_size)
        self.frat_proj = nn.Linear(hidden_size, hidden_size)
        self.attn = Attn(hidden_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=n_layers, dropout=dropout_p, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, parent_hidden, frat_hidden, encoder_outputs):
        hidden = (F.tanh(self.parent_proj(parent_hidden[0]) + self.frat_proj(frat_hidden[0])),
                  F.tanh(self.parent_proj(parent_hidden[1]) + self.frat_proj(frat_hidden[1])))
        output, hidden = self.rnn(input, hidden)
        output = self.attn(output, encoder_outputs)
        output = self.out(output.squeeze(1))
        output = F.log_softmax(output, dim=1)
        return hidden

class Type(nn.Module):
    '''
    module for handling type generation
    '''
    def __init__(self,
                 vocab_size, embed_size, hidden_size,
                 kind_dict, topo_loss_factor=1,
                 n_layers=1, dropout_p=0.0, weight=None):
        super(Type, self).__init__()
        self.num_modules = 2
        self.module_selector = nn.Sequential(
                                    nn.Linear(hidden_size, hidden_size//2),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size//2, self.num_modules),
                                    nn.LogSoftmax(dim=0)
                                   )
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.base_module = BaseType(vocab_size, embed_size, hidden_size)
        self.arrow_module = ArrowType(vocab_size, embed_size, hidden_size)
        self.kind_dict = kind_dict
        self.kind_dict[unk_token] = 0
        self.kind_zero = [x for x in kind_dict if kind_dict[x] == 0]
        self.actual_token_pos = arrow_token + 1  # where actual token starts
        self.topo_crit = nn.NLLLoss()
        self.crit = nn.NLLLoss(weight=weight)
        self.topo_loss_factor = topo_loss_factor

    def forward(self, input, parent_hidden, frat_hidden, encoder_outputs,
                is_train=False, reference=None, rec_depth=None,
                wrap_result=True):
        '''
        if is_train, the module returns loss and hidden states.
        Otherwise, it returns the actual tokens, along with hidden states
        '''
        batch_size = input.size(0)
        input_len = input.size(1)
        embed = self.embedding(input).view(batch_size, input_len, -1)
        # assuming batch_size = 1 here, need to be changed later
        choice_probs = self.module_selector(parent_hidden[0].view(-1))
        is_baseType = (choice_probs[0] > choice_probs[1]).item()
        if is_train:
            # compute topology loss
            target = 1 if reference.node == arrow_token else 0
            target = singleton_variable(target, 0).to(device)
            loss = self.topo_loss_factor * self.topo_crit(choice_probs.unsqueeze(0), target)
            is_baseType = reference.node != arrow_token
        else:
            if rec_depth == 0:
                is_baseType = True
        if is_baseType:
            output, frat_hidden = self.base_module(embed, parent_hidden, frat_hidden, encoder_outputs)
            if rec_depth == 0:
                # reach depth 0, can only generate type of kind *
                _, indices = torch.topk(output[:, self.kind_zero], 1)
                # a hack here
                data_constructor = torch.full((1, 1), self.kind_zero[indices[0][0].item()], dtype=torch.long, requires_grad=True).to(device)
            else:
                _, data_constructor = torch.topk(output[:, self.actual_token_pos:], 1)
                data_constructor = data_constructor + self.actual_token_pos
            if is_train:
                constructor_token = reference.node if type(reference.node) is int else reference.node[0]
                target = singleton_variable(constructor_token, 0).to(device)
                loss += self.crit(output, target)
                kind = self.kind_dict[constructor_token]
                data_constructor = singleton_variable(constructor_token, batch_size).to(device)
            else:
                kind = self.kind_dict[data_constructor[0][0].item()]
                # data_constructor.requires_grad_()
            results = [data_constructor.data[0][0].item()]
            for i in range(kind):
                if is_train:
                    if type(reference.node[i+1]) is int:
                        cur_reference = Tree(reference.node[i+1])
                    else:
                        cur_reference = reference.node[i+1]
                    cur_loss, parent_hidden, frat_hidden = self(data_constructor, parent_hidden, frat_hidden, encoder_outputs, is_train, cur_reference)
                    loss += cur_loss
                else:
                    cur_result, parent_hidden, frat_hidden = self(data_constructor, parent_hidden, frat_hidden, encoder_outputs, rec_depth=rec_depth-1, wrap_result=False)
                    results.append(cur_result)
            if is_train:
                return loss, parent_hidden, frat_hidden
            if not wrap_result and len(results) == 1:
                return results[0], parent_hidden, frat_hidden
            return Tree.singleton(results), parent_hidden, frat_hidden
        else:
            parent_hidden = self.arrow_module(embed, parent_hidden, frat_hidden, encoder_outputs)
            arrow_tensor = singleton_variable(arrow_token, batch_size).to(device)
            if is_train:
                left_loss, _, frat_hidden = self(arrow_tensor, parent_hidden, frat_hidden, encoder_outputs, is_train, reference.left)
                right_loss, _, frat_hidden = self(arrow_tensor, parent_hidden, frat_hidden, encoder_outputs, is_train, reference.right)
                loss += left_loss + right_loss
                return loss, parent_hidden, frat_hidden
            else:
                left_result, _, frat_hidden = self(arrow_tensor, parent_hidden, frat_hidden, encoder_outputs, rec_depth=rec_depth-1)
                right_result, _, frat_hidden = self(arrow_tensor, parent_hidden, frat_hidden, encoder_outputs, rec_depth=rec_depth-1)
                return Tree.from_children(arrow_token, left_result, right_result), parent_hidden, frat_hidden

class Model(nn.Module):
    '''
    final model for structured Prediction
    '''
    def __init__(self, input_vocab_size, target_vocab_size,
                 embed_size, hidden_size,
                 kind_dict,
                 n_layers=1, dropout_p=0.0,
                 topo_loss_factor=1, rec_depth=6,
                 weight=None):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = Encoder(input_vocab_size, embed_size, hidden_size,
                               n_layers=n_layers, dropout_p=dropout_p)
        self.decoder = Type(target_vocab_size, embed_size, hidden_size,
                            kind_dict, n_layers, dropout_p, weight=weight)
        self.rec_depth = rec_depth

    def forward(self, input_seq, lengths, is_train=False, reference=None):
        '''
        input_seq: a sequence of tokens representing the names
        '''
        batch_size = input_seq.size(0)
        hidden = self.encoder.initHidden(batch_size)
        encoder_outputs, encoder_hidden = self.encoder(input_seq, lengths, hidden)
        parent_hidden = (encoder_hidden[0].view(1, batch_size, -1),
                         encoder_hidden[1].view(1, batch_size, -1))
        frat_hidden = (torch.zeros(1, batch_size, self.hidden_size, requires_grad=True).to(device),
                       torch.zeros(1, batch_size, self.hidden_size, requires_grad=True).to(device)
                       )
        decoder_in = torch.full((batch_size, 1), start_token, dtype=torch.long, requires_grad=True).to(device)
        if is_train:
            loss, _, _ = self.decoder(decoder_in, parent_hidden, frat_hidden, encoder_outputs, is_train, reference)
            return loss
        else:
            results, _, _ = self.decoder(decoder_in, parent_hidden, frat_hidden, encoder_outputs, rec_depth=self.rec_depth)
            return results
