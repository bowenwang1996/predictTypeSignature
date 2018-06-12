import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import *
from prepare_data import arrow_token, start_token, unk_token
from type_signatures import Tree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#device="cpu"

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
        hidden = (torch.zeros(2 * self.n_layers, batch_size, self.hidden_size//2).to(device),
                  torch.zeros(2 * self.n_layers, batch_size, self.hidden_size//2).to(device)
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
        hidden = (torch.zeros(1, batch_size, self.hidden_size),
                  torch.zeros(1, batch_size, self.hidden_size)
                  )
        return hidden

class Attn(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=None):
        super(Attn, self).__init__()
        self.attn = nn.Linear(input_size, hidden_size)
        if output_size:
            self.attn_combine = nn.Linear(hidden_size * 2, output_size)
        else:
            self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input, encoder_outputs, coverage=None, score_only=False, context_only=False):
        '''
        input: B * 1 * L
        encoder_outputs: B * T * H
        returns: transformed decoder output and attention scores
        '''
        input = self.attn(input)
        attn_scores = torch.bmm(encoder_outputs, input.transpose(1, 2))
        attn_scores = F.softmax(attn_scores, dim=1).transpose(1, 2)
        if score_only:
            return attn_scores.squeeze(1)
        context = torch.bmm(attn_scores, encoder_outputs).squeeze(1) #B*H
        if context_only:
            return context
        output = F.tanh(self.attn_combine(torch.cat((input.squeeze(1), context), 1)))
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
        oov_var = torch.zeros(batch_size, self.max_oov).to(device)
        p_vocab = torch.cat((p_vocab, oov_var), 1)

        batch_indices = torch.arange(start=0, end=batch_size).long()
        batch_indices = batch_indices.expand(context_input_len, batch_size).transpose(1, 0).contiguous().view(-1).to(device)

        p_copy = torch.zeros(batch_size, self.vocab_size + self.max_oov).to(device)
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
        #self.parent_proj = nn.Linear(hidden_size, hidden_size)
        #self.frat_proj = nn.Linear(hidden_size, hidden_size)
        self.attn = CopyAttn(hidden_size, hidden_size)
        self.gen_prob = nn.Sequential(
                            nn.Linear(3 * hidden_size + embed_size, 1),
                            nn.Sigmoid()
                            )
        self.out = nn.Linear(hidden_size, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input, hidden, encoder_outputs, context_info, max_oov=20):
        '''
        input should already be embedded here (done in Type module)
        '''
        # hidden = (F.tanh(self.parent_proj(parent_hidden[0]) + self.frat_proj(frat_hidden[0])),
        #          F.tanh(self.parent_proj(parent_hidden[1]) + self.frat_proj(frat_hidden[1])))
        output, hidden = self.rnn(input, hidden)
        if context_info is not None:
            cat_hidden = torch.cat((output,
                                   context_info.name_hidden[0],
                                   context_info.type_hidden[0],
                                   input), 2)
            gen_p = self.gen_prob(cat_hidden).view(-1)
            oov_var = torch.zeros(max_oov).to(device)
            output_prob = F.softmax(self.out(output).view(-1), dim=0)
            p_vocab = torch.cat((output_prob, oov_var), 0)
            context_attn_scores = self.attn(output, context_info.type_hiddens, score_only=True)
            p_copy = torch.zeros(self.vocab_size + max_oov).to(device)
            p_copy = p_copy.clone()
            p_copy.put_(context_info.type_indices, context_attn_scores.view(-1), accumulate=True)
            prob = gen_p * p_vocab + (1 - gen_p) * p_copy
            return torch.log(prob.clamp(min=1e-10).unsqueeze(0)), hidden
        else:
            output = self.attn(output, encoder_outputs)
            output = self.out(output.squeeze(1))
            output = F.log_softmax(output, dim=1)
            return output, hidden

class ArrowType(nn.Module):
    '''
    submodule for handling arrow types
    '''
    def __init__(self, vocab_size, embed_size, hidden_size, n_layers=1, dropout_p=0.0):
        super(ArrowType, self).__init__()
        #self.parent_proj = nn.Linear(hidden_size, hidden_size)
        #self.frat_proj = nn.Linear(hidden_size, hidden_size)
        self.attn = Attn(embed_size, hidden_size, output_size=embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size, num_layers=n_layers, dropout=dropout_p, batch_first=True)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, input, hidden, encoder_outputs):
        # hidden = (F.tanh(self.parent_proj(parent_hidden[0]) + self.frat_proj(frat_hidden[0])),
        #          F.tanh(self.parent_proj(parent_hidden[1]) + self.frat_proj(frat_hidden[1])))
        input = self.attn(input, encoder_outputs).unsqueeze(0)
        output, hidden = self.rnn(input, hidden)
        return hidden

class CopyAttn(nn.Module):
    '''
    attention for copying, separated from Attn for experiments
    '''
    def __init__(self, input_size, hidden_size, output_size=None):
        super().__init__()
        self.attn = nn.Linear(input_size, hidden_size)
        if output_size:
            self.attn_combine = nn.Linear(hidden_size * 2, output_size)
        else:
            self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input, encoder_outputs, score_only=False):
        '''
        input: B * 1 * T
        encoder_outputs: B * T * H
        returns: normalized attention scores
        '''
        input = self.attn(input)
        attn_scores = torch.bmm(encoder_outputs, input.transpose(1, 2))
        attn_scores = F.softmax(attn_scores, dim=1).transpose(1, 2)
        if score_only:
            return attn_scores
        context = torch.bmm(attn_scores, encoder_outputs).squeeze(1)  # B*H
        output = F.tanh(self.attn_combine(torch.cat((input.squeeze(1), context), 1)))
        return output

class Type(nn.Module):
    '''
    module for handling type generation
    '''
    def __init__(self,
                 vocab_size, embed_size, hidden_size,
                 kind_dict, topo_loss_factor=1,
                 n_layers=1, dropout_p=0.0, weight=None,
                 max_oov=20):
        super(Type, self).__init__()
        self.num_modules = 2
        self.module_selector = nn.Sequential(
                                    nn.Linear(hidden_size + embed_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, self.num_modules),
                                    nn.LogSoftmax(dim=0)
                                   )
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.base_module = BaseType(vocab_size, embed_size, hidden_size)
        self.left_arrow_module = ArrowType(vocab_size, embed_size, hidden_size)
        self.right_arrow_module = ArrowType(vocab_size, embed_size, hidden_size)
        self.context_attn = CopyAttn(hidden_size, hidden_size)
        self.hidden_combiner = Combiner(hidden_size, hidden_num=2)
        self.kind_dict = kind_dict
        self.kind_dict[unk_token] = 0
        self.kind_zero = [x for x in kind_dict if kind_dict[x] == 0]
        self.actual_token_pos = arrow_token + 1  # where actual token starts
        self.topo_crit = nn.NLLLoss()
        self.crit = nn.NLLLoss(weight=weight)
        self.topo_loss_factor = topo_loss_factor
        self.max_oov = max_oov
        self.vocab_size = vocab_size

    def get_kind_zero(self, kind_dict):
        return [x for x in kind_dict if kind_dict[x] == 0]

    def forward(self, input, hidden, encoder_outputs, context_info,
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
        choice_probs = self.module_selector(torch.cat((hidden[0].view(-1), embed.view(-1))))
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
            output, hidden = self.base_module(embed, hidden, encoder_outputs, context_info, max_oov=self.max_oov)
            if rec_depth == 0:
                # reach depth 0, can only generate type of kind *
                if context_info is not None:
                    oov_kind_zero = self.get_kind_zero(context_info.oov_kind_dict)
                else:
                    oov_kind_zero = []
                kind_zero = self.kind_zero + oov_kind_zero
                _, indices = torch.topk(output[:, kind_zero], 1)
                # a hack here
                data_constructor = torch.full((1, 1), kind_zero[indices[0][0].item()], dtype=torch.long).to(device)
            else:
                _, data_constructor = torch.topk(output[:, self.actual_token_pos:], 1)
                data_constructor = data_constructor + self.actual_token_pos
            if is_train:
                token = reference.node if type(reference.node) is int else reference.node[0]
                target = singleton_variable(token, 0).to(device)
                loss += self.crit(output, target)
                kind = self.kind_dict[token]
                data_constructor = singleton_variable(token, batch_size).to(device)
            else:
                token = data_constructor[0][0].item()
                if self.kind_dict.get(token) is not None:
                    kind = self.kind_dict[token]
                else:
                    kind = context_info.oov_kind_dict[token]
            results = [data_constructor.data[0][0].item()]
            for i in range(kind):
                if token >= self.vocab_size:
                    data_constructor = singleton_variable(unk_token, 1).to(device)
                if is_train:
                    if type(reference.node[i+1]) is int:
                        cur_reference = Tree(reference.node[i+1])
                    else:
                        cur_reference = reference.node[i+1]
                    cur_loss, next_in, hidden = self(data_constructor, hidden, encoder_outputs, context_info, is_train, cur_reference)
                    data_constructor = singleton_variable(next_in, 1).to(device)
                    loss += cur_loss
                else:
                    cur_result, hidden = self(data_constructor, hidden, encoder_outputs, context_info, rec_depth=rec_depth-1, wrap_result=False)
                    if isinstance(cur_result, Tree):
                        next_in = cur_result.get_last()
                    else:
                        next_in = cur_result
                    if next_in >= self.vocab_size:
                        next_in = unk_token
                    data_constructor = singleton_variable(next_in, 1).to(device)
                    results.append(cur_result)
            if is_train:
                return loss, token, hidden
            if not wrap_result and len(results) == 1:
                return results[0], hidden
            return Tree.singleton(results), hidden
        else:
            left_hidden = self.left_arrow_module(embed, hidden, encoder_outputs)
            #right_hidden = self.right_arrow_module(embed, hidden, encoder_outputs)
            arrow_tensor = singleton_variable(arrow_token, batch_size).to(device)
            if is_train:
                left_loss, _, left_hidden = self(arrow_tensor, left_hidden, encoder_outputs, context_info, is_train, reference.left)
                right_hidden = self.hidden_combiner(left_hidden, hidden)
                right_loss, _, hidden = self(arrow_tensor, right_hidden, encoder_outputs, context_info, is_train, reference.right)
                loss += left_loss + right_loss
                return loss, reference.get_last(), hidden
            else:
                left_result, left_hidden = self(arrow_tensor, left_hidden, encoder_outputs, context_info, rec_depth=rec_depth-1)
                right_hidden = self.hidden_combiner(left_hidden, hidden)
                right_result, hidden = self(arrow_tensor, right_hidden, encoder_outputs, context_info, rec_depth=rec_depth-1)
                return Tree.from_children(arrow_token, left_result, right_result), hidden

class ContextType(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size,
                 kind_dict, n_layers=1, dropout_p=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.LSTM(embed_size, hidden_size,
                           num_layers=n_layers, dropout=dropout_p,
                           batch_first=True)
        self.vocab_size = vocab_size

    def forward_(self, input, hidden, hiddens):
        if type(input.node) is int:
            embedded = self.embedding(singleton_variable(input.node, 0).to(device))
            output, hidden = self.rnn(embedded.view(1, 1, -1), hidden)
            if input.node != arrow_token:
                hiddens.append(output)
        else:
            assert(type(input.node) is list)
            for item in input.node:
                if type(item) is int:
                    embedded = self.embedding(singleton_variable(item, 0).to(device))
                    output, hidden = self.rnn(embedded.view(1, 1, -1), hidden)
                    if item != arrow_token:
                        hiddens.append(output)
                else:
                    self.forward_(item, hidden, hiddens)
        if input.left is not None:
            self.forward_(input.left, hidden, hiddens)
        if input.right is not None:
            _, hidden = self.forward_(input.right, hidden, hiddens)
        return hiddens, hidden


    def forward(self, input):
        '''
        input is the tree of indices for a type
        '''
        hidden = self.initHidden()
        # node_map = input.traversal()
        # node_num = len(node_map)
        hiddens, last_hidden = self.forward_(input, hidden, [])
        return hiddens, last_hidden

    def initHidden(self):
        hidden = (torch.zeros(1, 1, self.hidden_size).to(device),
                  torch.zeros(1, 1, self.hidden_size).to(device)
                  )
        return hidden

class Combiner(nn.Module):

    def __init__(self, hidden_size, hidden_num=3):
        super().__init__()
        self.weight = nn.Linear(hidden_size * hidden_num, hidden_size)
        '''
        self.w_in = nn.Linear(hidden_size, hidden_size)
        self.w_con_name = nn.Linear(hidden_size, hidden_size)
        self.w_con_type = nn.Linear(hidden_size, hidden_size)
        '''

    def forward_(self, *args):
        return F.tanh(self.weight(torch.cat(args, 2)))
        # return F.tanh(self.w_in(h1) + self.w_con_name(h2) + self.w_con_type(h3))

    def forward(self, *args):
        hidden = (self.forward_(*[x[0] for x in args]),
                  self.forward_(*[x[1] for x in args])
                  )
        return hidden

class ContextInfo():
    '''
    a class for holding relevant context info to pass to decoder
    '''

    def __init__(self, name_hidden, type_hiddens, type_hidden, type_indices, oov_kind_dict):
        assert(len(type_hiddens) == len(type_indices))
        self.name_hidden = name_hidden
        self.type_hiddens = torch.cat(type_hiddens, 1)
        self.type_hidden = type_hidden
        self.type_indices = torch.tensor(type_indices, dtype=torch.long).to(device)
        self.oov_kind_dict = oov_kind_dict

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
        self.context_type_encoder = ContextType(target_vocab_size, embed_size,
                                                hidden_size, kind_dict,
                                                n_layers, dropout_p)
        self.context_name_encoder = Encoder(input_vocab_size, embed_size,
                                            hidden_size, n_layers, dropout_p)
        self.combiner = Combiner(hidden_size)
        self.rec_depth = rec_depth

    def forward(self, input_seq, lengths, context_info,
                is_train=False, reference=None):
        '''
        input_seq: a sequence of tokens representing the names
        '''

        # process input name
        batch_size = input_seq.size(0)
        hidden = self.encoder.initHidden(batch_size)
        encoder_outputs, encoder_hidden = self.encoder(input_seq, lengths, hidden)
        hidden = (encoder_hidden[0].view(1, batch_size, -1),
                  encoder_hidden[1].view(1, batch_size, -1))

        if context_info is not None:
            context_num = context_info.num

            # process context names
            context_name_hidden = self.context_name_encoder.initHidden(context_num)
            context_name_encoder_outputs, context_name_hidden = self.context_name_encoder(context_info.names, context_info.name_lengths, context_name_hidden)
            context_name_hidden = (context_name_hidden[0].view(context_num, 1, -1),
                                   context_name_hidden[1].view(context_num, 1, -1))
            context_name_hidden = ((context_name_hidden[0].sum(dim=0)/context_num).unsqueeze(0),
                                   (context_name_hidden[1].sum(dim=0)/context_num).unsqueeze(0))

            # process context types
            context_type_hiddens = []
            context_type_last_hiddens = []
            for tree in context_info.sigs:
                type_hiddens, type_hidden = self.context_type_encoder(tree)
                context_type_hiddens += type_hiddens
                context_type_last_hiddens.append(type_hidden)
            context_type_hidden = (sum([x[0] for x in context_type_last_hiddens])/context_num,
                                   sum([x[1] for x in context_type_last_hiddens])/context_num)

            context_info = ContextInfo(context_name_hidden, context_type_hiddens,
                                       context_type_hidden, context_info.indices,
                                       context_info.oov_kind_dict)
            hidden = self.combiner(hidden, context_name_hidden, context_type_hidden)
        decoder_in = torch.full((batch_size, 1), start_token, dtype=torch.long).to(device)
        if is_train:
            loss, _, _ = self.decoder(decoder_in, hidden, encoder_outputs, context_info, is_train, reference)
            return loss
        else:
            results, _ = self.decoder(decoder_in, hidden, encoder_outputs, context_info, rec_depth=self.rec_depth)
            return results
