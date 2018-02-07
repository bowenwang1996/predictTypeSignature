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

    # this methods is only for input vocab
    def add_name(self, name):
        assert(self.is_input)
        for token in identifier_segmentor.segment(name):
            self.add_token(token.lower())
            
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

# takes the input line and returns the file number, the function name, and the corresponding signature
def processLine(line):
    [num, annot] = line.split('\t')
    [input_name, sig] = annot.split('::', 1)
    input_name = input_name.strip()
    sig = sig.strip()
    return int(num), input_name, sig

# processes plain data that only consist of function names and signatures
def prepareData(filename, use_context=False, num_context_sig=3):
    with open(filename, 'r') as f:
        lines = filter(lambda l: len(l) > 0, f.read().split('\n'))
    input_lang = Lang(True)
    data = []
    if use_context:
        context_sigs = []
        for line in lines:
            num, input_name, sig = processLine(line)
            cur_context = context_sigs[-num_context_sig:]
            sigs = [ p[1] for p in cur_context if p[0] == num]
            data.append(([input_name], sig, sigs))
            context_sigs.append([num, sig])
            for token in identifier_segmentor.segment(input_name):
                input_lang.add_token(token)
    else:
        for line in lines:
            _, input_name, sig = processLine(line)
            for token in identifier_segmentor.segment(input_name):
                input_lang.add_token(token.lower())
            data.append(([input_name], sig))
    random.shuffle(data)
    return input_lang, data

def processLineWithFileName(line):
    [num, path, name_and_sig] = line.split('\t')
    filename = path.split('/')[-1].split('.')[0]
    [name, sig] = name_and_sig.split('::', 1)
    return int(num), filename, name.strip(), sig.strip()

# processes data with qualified name (full path to the file) and signatures
def prepareDataWithFileName(filename, use_context=False, num_context_sig=3):
    with open(filename, 'r') as f:
        lines = filter(lambda l: len(l) > 0, f.read().split('\n'))
    input_lang = Lang(True)
    data = []
    if use_context:
        context_sigs = []
        for line in lines:
            num, fname, input_name, sig = processLineWithFileName(line)
            cur_context = context_sigs[-num_context_sig:]
            sigs = [ p[1] for p in cur_context if p[0] == num]
            data.append(([fname, input_name], sig, sigs))
            context_sigs.append([num, sig])
            input_lang.add_name(input_name)
            input_lang.add_name(fname)
    else:
        for line in lines:
            _, fname, input_name, sig = processLineWithFileName(line)
            input_lang.add_name(input_name)
            input_lang.add_name(fname)
            # this is somewhat hacky. Might need to write a class just for name
            data.append(([fname, input_name], sig))
    random.shuffle(data)
    return input_lang, data
