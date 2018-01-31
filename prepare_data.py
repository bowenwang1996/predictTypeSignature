import identifier_segmentor
from type_signatures import UNPARSE_ARROW
import random

SPECIAL_SYMBOLS = ["(", ")", "[", "]", ",", UNPARSE_ARROW, ":->", "<-->"] 
start_token = 0
end_token = 1
unk_token = 2

class Lang:
    def __init__(self, is_input):
        self.token_to_idx = {}
        self.idx_to_token = {0: "<TYPE>", 1: "</TYPE>", 2: "<UNK>"}
        self.word_to_count = {}
        self.n_word = 3
        self.is_input = is_input

    def add_token(self, token):
        if token in self.token_to_idx:
            self.word_to_count[token] += 1
        else:
            self.word_to_count[token] = 1
            self.token_to_idx[token] = self.n_word
            self.idx_to_token[self.n_word] = token
            self.n_word += 1

    def lookup(self, token):
        if token in self.token_to_idx:
            return self.token_to_idx[token]
        else:
            return unk_token

def segment(ident):
    if ident in SPECIAL_SYMBOLS:
        return [ident]
    else:
        return identifier_segmentor.segment(ident)

def readSigTokens(filename):
    with open(filename, 'r') as f:
        tokens = f.read().split('\n')
    lang = Lang(False)
    for token in tokens:
        # if it is a typename, we only need the unqualified part
        if token[0].isalpha():
            lang.add_token(token.split('.')[-1])
        elif not token[0].isalpha():
            lang.add_token(token)
    for token in SPECIAL_SYMBOLS:
        lang.add_token(token)
    return lang

def processLine(line):
    [num, annot] = line.split('\t')
    [input_name, sig] = annot.split('::')
    input_name = input_name.strip()
    sig = sig.strip()
    return num, input_name, sig

def prepareData(filename, use_context=False, num_context_sig=3):
    with open(filename, 'r') as f:
        lines = f.read().split('\n')
    input_lang = Lang(True)
    data = []
    if use_context:
        context_sigs = []
        for line in lines:
            num, input_name, sig = processLine(line)
            cur_context = context_sigs[-num_context_sig:]
            sigs = [ p[1] for p in cur_context if p[0] == num]
            data.append((input_name, sig, sigs))
            context_sigs.append([num, sig])
            for token in identifier_segmentor.segment(input_name):
                input_lang.add_token(token)
    else:
        for line in lines:
            _, input_name, sig = processLine(line)
            for token in identifier_segmentor.segment(input_name):
                input_lang.add_token(token)
            data.append((input_name, sig))
    random.shuffle(data)
    return input_lang, data
