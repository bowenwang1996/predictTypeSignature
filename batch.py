from prepare_data import *
import identifier_segmentor
from utils import *

import torch
from torch.autograd import Variable

use_cuda = torch.cuda.is_available()
#use_cuda = False

class TrainInfo():
    '''
    this class is purely used to hold data. I don't want to rewrite a function that has 20 arguments
    '''
    def __init__(self, input_var, input_lengths, target_var, target_lengths, context_var, context_lengths, oov_idx_dict, idx_oov_dict):
        self.input_variable = input_var
        self.input_lengths = input_lengths
        self.target_variable = target_var
        self.target_lengths = target_lengths
        self.context_variable = context_var
        self.context_lengths = sorted(context_lengths, reverse=True)
        self.context_sort_index = get_sort_index(context_lengths)
        self.context_inv_index = inv(self.context_sort_index)
        self.oov_idx_dict = oov_idx_dict
        self.idx_oov_dict = idx_oov_dict

class Batch():
    def __init__(self, batch_size, input_vocab, target_vocab, use_context=True):
        self.batch_size = batch_size
        self.input_vocab = input_vocab
        self.target_vocab = target_vocab
        self.use_context = use_context

    def batchify(self, data):
        data_len = len(data)
        num_batches = data_len / self.batch_size
        batches = []
        for i in range(num_batches):
            batches.append(data[self.batch_size*i:self.batch_size*(i+1)])
        return batches

    def indexFromName(self, name):
        tokens = []
        for ident in name:
            tokens += identifier_segmentor.segment(ident)
        indices = map(lambda x: self.input_vocab.lookup(x.lower()), tokens)
        indices.append(end_token)
        return indices

    def variableFromName(self, name):
        indices = self.indexFromName(name)
        name_var = Variable(torch.LongTensor(indices))
        if use_cuda:
            name_var = name_var.cuda()
        return name_var

    def indexFromSignature(self, sig):
        tokens = sig.split()
        def foo(x):
            if x[0].isupper():
                token = x.split('.')[-1]
            else:
                token = x
            return self.target_vocab.lookup(token)
        indices = map(foo, tokens)
        indices.append(end_token)
        return indices

    def variableFromSignature(self, sig):
        indices = self.indexFromSignature(sig)
        sig_var = Variable(torch.LongTensor(indices))
        if use_cuda:
            sig_var = sig_var.cuda()
        return sig_var

    # unlike the method above, this method needs to take into account of oov words
    def indexFromSignatures(self, sigs, oov_idx_dict, idx_oov_dict):
        tokens = sum(map(lambda x: x.split(), sigs), [])
        indices = []
        for sig in sigs:
            tokens = sig.split()
            for token in tokens:
                if token[0].isupper():
                    token = token.split(".")[-1]
                if token in self.target_vocab.token_to_idx:
                    indices.append(self.target_vocab.token_to_idx[token])
                elif token in oov_idx_dict:
                    indices.append(oov_idx_dict[token])
                else:
                    l = len(oov_idx_dict)
                    oov_idx_dict[token] = self.target_vocab.n_word + l
                    idx_oov_dict[self.target_vocab.n_word+l] = token
                    indices.append(oov_idx_dict[token])
            indices.append(end_token)
        if not indices:
            indices.append(start_token)
        return indices

    def variableFromSignatures(self, sigs, oov_idx_dict, idx_oov_dict):
        indices = self.indexFromSignatures(sigs, oov_idx_dict, idx_oov_dict)
        var = Variable(torch.LongTensor(indices))
        if use_cuda:
            var = var.cuda()
        return var

    def variableFromBatch(self, batch):
        input_target_batch = map(lambda p: (self.indexFromName(p[0]), self.indexFromSignature(p[1])), batch)
        input_sort_index = get_sort_index(map(lambda p: len(p[0]), input_target_batch))
        input_target_batch = sorted(input_target_batch, key=lambda p:len(p[0]), reverse=True)
        input_lengths = map(lambda p: len(p[0]), input_target_batch)
        target_lengths = map(lambda p: len(p[1]), input_target_batch)
        max_input_len = max(input_lengths)
        max_target_len = max(target_lengths)
        input_batch = map(lambda p: p[0] + (max_input_len - len(p[0])) * [0], input_target_batch)
        target_batch = map(lambda p: p[1] + (max_target_len - len(p[1])) * [0], input_target_batch)

        input_batch = torch.LongTensor(input_batch)
        target_batch = torch.LongTensor(target_batch)

        input_variable = Variable(input_batch)
        target_variable = Variable(target_batch)
        if use_cuda:
            input_variable = input_variable.cuda()
            target_variable = target_variable.cuda()

        if self.use_context:
            oov_idx_dict = {}
            idx_oov_dict = {}
            context_batch = []
            for x in batch:
                indices = self.indexFromSignatures(x[2], oov_idx_dict, idx_oov_dict)
                context_batch.append(indices)

            context_lengths = map(len, context_batch)
            max_context_len = max(context_lengths)
            context_batch = pad_to_len(context_batch, max_context_len)
            context_variable = Variable(torch.LongTensor(context_batch))
            if use_cuda:
                context_variable = context_variable.cuda()
            context_variable = context_variable[input_sort_index]
            context_lengths = [context_lengths[i] for i in input_sort_index]
            return TrainInfo(input_variable, input_lengths, target_variable, target_lengths, context_variable, context_lengths, oov_idx_dict, idx_oov_dict)
        return TrainInfo(input_variable, input_lengths, target_variable, target_lengths, None, None, None, None)

    def unk_batch(self, batch):
        # here batch is a B * seq_len tensor

        vocab_batch = (batch < self.target_vocab.n_word).long()
        unk_batch = (batch >= self.target_vocab.n_word).long()
        return vocab_batch * batch + unk_batch * unk_token
