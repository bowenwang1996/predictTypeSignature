from prepare_data import *
import identifier_segmentor
from utils import *

import torch
from torch.autograd import Variable
import copy

use_cuda = torch.cuda.is_available()
#use_cuda = False

class TrainInfo():
    '''
    this class is purely used to hold data. I don't want to rewrite a function that has 20 arguments
    '''
    def __init__(self, input_var, input_lengths,
                 target_var, target_type_var, target_lengths,
                 context_var, context_type_var, context_lengths,
                 oov_idx_dict, idx_oov_dict, oov_type_dict):
        self.input_variable = input_var
        self.input_lengths = input_lengths
        self.target_variable = target_var
        self.target_type_variable = target_type_var
        self.target_lengths = target_lengths
        self.context_variable = context_var
        self.context_type_variable = context_type_var
        self.context_lengths = sorted(context_lengths, reverse=True)
        self.context_sort_index = get_sort_index(context_lengths)
        self.context_inv_index = inv(self.context_sort_index)
        self.oov_idx_dict = oov_idx_dict
        self.idx_oov_dict = idx_oov_dict
        self.oov_type_dict = oov_type_dict

class Batch():
    def __init__(self, batch_size, input_vocab, target_vocab, use_context=True):
        self.batch_size = batch_size
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
        self.use_context = use_context

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def batchify(self, data):
        data_len = len(data)
        num_batches = data_len // self.batch_size
        batches = []
        for i in range(num_batches):
            batches.append(data[self.batch_size*i:self.batch_size*(i+1)])
        return batches

    def indexFromName(self, name):
        tokens = []
        for ident in name:
            tokens += identifier_segmentor.segment(ident)
        indices = [self.input_vocab.lookup(x.lower()) for x in tokens]
        indices.append(end_token)
        return indices

    def variableFromName(self, name):
        indices = self.indexFromName(name)
        name_var = Variable(torch.LongTensor(indices))
        if use_cuda:
            name_var = name_var.cuda()
        return name_var

    def indexFromType(self, token):
        if token in self.target_vocab.type_to_tokens:
            indices = self.target_vocab.type_to_tokens[token]
        else:
            indices = self.target_vocab.compute_embed_idx(token)
        return indices

    def typeIndexFromSignature(self, sig, oov_idx_dict=None):
        tokens = sig.split()
        indices = []
        for token in tokens:
            if token[0].isupper():
                token = token.split('.')[-1]
            if token in self.target_vocab.type_to_idx:
                indices.append(self.target_vocab.type_to_idx[token])
            elif token in oov_idx_dict:
                indices.append(oov_idx_dict[token])
            else:
                indices.append(unk_token)
        return indices
        
    def indexFromSignature(self, sig):
        tokens = sig.split()
        indices = [self.indexFromType(x) for x in tokens]
        indices.append([end_token])
        return indices

    def variableFromSignature(self, sig):
        indices = self.indexFromSignature(sig)
        sig_var = Variable(torch.LongTensor(indices))
        if use_cuda:
            sig_var = sig_var.cuda()
        return sig_var

    # unlike the method above, this method needs to take into account of oov words
    def indexFromSignatures(self, sigs, oov_idx_dict, idx_oov_dict, oov_type_dict):
        indices = []
        type_indices = []
        for sig in sigs:
            tokens = sig.split()
            for token in tokens:
                if token[0].isupper():
                    token = token.split(".")[-1]
                cur_indices = self.indexFromType(token.lower())
                indices.append(cur_indices)

                if token in self.target_vocab.type_to_idx:
                    idx = self.target_vocab.type_to_idx[token]
                    type_indices.append(idx)
                elif token in oov_idx_dict:
                    idx = oov_idx_dict[token]
                    type_indices.append(idx)
                else:
                    idx = self.target_vocab.n_type + len(oov_idx_dict)
                    oov_idx_dict[token] = idx
                    idx_oov_dict[idx] = token
                    oov_type_dict[token] = copy.deepcopy(cur_indices)
            indices.append([end_token])
        if not indices:
            indices.append([end_token])
            type_indices.append(end_token)
        return indices, type_indices

    '''
    def variableFromSignatures(self, sigs, oov_idx_dict, idx_oov_dict):
        indices = self.indexFromSignatures(sigs, oov_idx_dict, idx_oov_dict)
        var = Variable(torch.LongTensor(indices))
        if use_cuda:
            var = var.cuda()
        return var
    '''
    def variableFromBatch(self, batch):
        if self.use_context:
            oov_idx_dict = {}
            idx_oov_dict = {}
            oov_type_dict = {}
            context_batch = []
            context_type_batch = []
            for x in batch:
                indices, type_indices = self.indexFromSignatures(x[2], oov_idx_dict, idx_oov_dict, oov_type_dict)
                context_batch.append(indices)
                context_type_batch.append(type_indices)

            context_lengths = list(map(len, context_batch))
            max_context_len = max(context_lengths)
            pad_nested_to_len(context_batch, max_context_len)
            context_type_batch = pad_to_len(context_type_batch, max_context_len)
            context_variable = Variable(torch.LongTensor(context_batch))
            context_type_variable = Variable(torch.LongTensor(context_type_batch))
            if use_cuda:
                context_variable = context_variable.cuda()
                context_type_variable = context_type_variable.cuda()
        else:
            oov_idx_dict = None
        input_target_batch = [(self.indexFromName(p[0]), self.indexFromSignature(p[1]), self.typeIndexFromSignature(p[1], oov_idx_dict)) for p in batch]
        input_sort_index = get_sort_index(list(map(lambda p: len(p[0]), input_target_batch)))
        input_target_batch = sorted(input_target_batch, key=lambda p:len(p[0]), reverse=True)
        input_lengths = list(map(lambda p: len(p[0]), input_target_batch))
        target_lengths = list(map(lambda p: len(p[1]), input_target_batch))
        max_input_len = max(input_lengths)
        max_target_len = max(target_lengths)
        input_batch = list(map(lambda p: p[0] + (max_input_len - len(p[0])) * [0], input_target_batch))
        target_batch = [p[1] for p in input_target_batch]
        target_type_batch = pad_to_len([p[2] for p in input_target_batch], max_target_len)
        pad_nested_to_len(target_batch, max_target_len)

        input_batch = torch.LongTensor(input_batch)
        target_batch = torch.LongTensor(target_batch)
        target_type_batch = torch.LongTensor(target_type_batch)

        input_variable = Variable(input_batch)
        target_variable = Variable(target_batch)
        target_type_variable = Variable(target_type_batch)
        
        if use_cuda:
            input_variable = input_variable.cuda()
            target_variable = target_variable.cuda()
            target_type_variable = target_type_variable.cuda()

        if self.use_context:
            context_variable = context_variable[input_sort_index]
            context_type_variable = context_type_variable[input_sort_index]
            context_lengths = [context_lengths[i] for i in input_sort_index]
            return TrainInfo(input_variable, input_lengths,
                             target_variable, target_type_variable, target_lengths,
                             context_variable, context_type_variable, context_lengths,
                             oov_idx_dict, idx_oov_dict, oov_type_dict)
        return TrainInfo(input_variable, input_lengths,
                         target_variable, target_lengths, None,
                         None, None, None,
                         None, None, None)

    def unk_batch(self, batch):
        # here batch is a B * seq_len tensor

        vocab_batch = (batch < self.target_vocab.n_word).long()
        unk_batch = (batch >= self.target_vocab.n_word).long()
        return vocab_batch * batch + unk_batch * unk_token
