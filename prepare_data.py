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
        self.token_to_count = {}
        self.n_word = 3
        self.is_input = is_input

    def add_token(self, token):
        if token in self.token_to_idx:
            self.token_to_count[token] += 1
        else:
            self.token_to_count[token] = 1
            self.token_to_idx[token] = self.n_word
            self.idx_to_token[self.n_word] = token
            self.n_word += 1

    # this methods is only for input vocab
    def add_name(self, name):
        assert(self.is_input)
        for token in identifier_segmentor.segment(name):
            self.add_token(token.lower())

    def add_sig(self, sig):
        #assert(not self.is_input)
        # if it is a typename, we only need the unqualified part
        for token in sig.split():
            if token[0].isalpha():
                self.add_token(token.split('.')[-1])
            else:
                self.add_token(token)

    def merge_output(self, output_lang):
        assert(self.is_input)
        token_to_idx = {}
        idx_to_token = {0: "<TYPE>", 1: "</TYPE>", 2: "<UNK>"}
        for token in output_lang.token_to_idx:
            token_to_idx[token] = output_lang.token_to_idx[token]
            idx_to_token[output_lang.token_to_idx[token]] = token
        for token in self.token_to_idx:
            if token not in token_to_idx:
                token_to_idx[token] = output_lang.n_word - 3 + self.token_to_idx[token]
                idx_to_token[output_lang.n_word - 3 + self.token_to_idx[token]] = token
                self.n_word += 1
        self.token_to_idx = token_to_idx
        self.idx_to_token = idx_to_token
            
    def lookup(self, token):
        if token in self.token_to_idx:
            return self.token_to_idx[token]
        else:
            return unk_token

    # remove tokens that have frequency below a certain threshold
    def trim_tokens(self, threshold=5):
        for token in self.token_to_count.keys():
            if self.token_to_count[token] < threshold and self.token_to_idx[token] > 2 and random.random() < 0.5:
                self.idx_to_token.pop(self.token_to_idx[token])
                self.token_to_idx.pop(token)
                self.token_to_count.pop(token)
                self.n_word -= 1
        idx_dict = {}
        for idx, key in enumerate(self.idx_to_token.keys()):
            idx_dict[idx] = self.idx_to_token[key]
        token_dict = {}
        for idx in idx_dict:
            token_dict[idx_dict[idx]] = idx
        self.idx_to_token = idx_dict
        self.token_to_idx = token_dict

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
    output_lang = Lang(False)
    data = []
    if use_context:
        context_sigs = []
        for line in lines:
            num, input_name, sig = processLine(line)
            cur_context = context_sigs[-num_context_sig:]
            sigs = [ p[1] for p in cur_context if p[0] == num]
            data.append(([input_name], sig, sigs))
            context_sigs.append([num, sig])
            input_lang.add_name(input_name)
            output_lang.add_sig(sig)
    else:
        for line in lines:
            _, input_name, sig = processLine(line)
            input_lang.add_name(input_name)
            output_lang.add_sig(sig)
            data.append(([input_name], sig))
    random.shuffle(data)
    return input_lang, output_lang, data

def processLineWithFileName(line, full_path=False):
    [num, path, name_and_sig] = line.split('\t')
    filename = path.split('/')
    if full_path:
        filename[-1] = filename[-1].split('.')[0]
    else:
        filename = [path.split('/')[-1].split('.')[0]]
    [name, sig] = name_and_sig.split('::', 1)
    return int(num), filename, name.strip(), sig.strip()

# processes data with qualified name (full path to the file) and signatures
def prepareDataWithFileName(filename, full_path=False, use_context=False, num_context_sig=3):
    with open(filename, 'r') as f:
        lines = filter(lambda l: len(l) > 0, f.read().split('\n'))
    input_lang = Lang(True)
    output_lang = Lang(False)
    data = []
    if use_context:
        context_sigs = []
        for line in lines:
            num, fname, input_name, sig = processLineWithFileName(line, full_path)
            cur_context = context_sigs[-num_context_sig:]
            context = [ (p[1], p[2]) for p in cur_context if p[0] == num]
            name = fname + [input_name]
            data.append((name, sig, context))
            context_sigs.append([num, name, sig])
            for ident in name:
                input_lang.add_name(ident)
            output_lang.add_sig(sig)
        input_lang.merge_output(output_lang)
    else:
        for line in lines:
            _, fname, input_name, sig = processLineWithFileName(line, full_path)
            name = fname + [input_name]
            for ident in name:
                input_lang.add_name(ident)
            output_lang.add_sig(sig)
            # this is somewhat hacky. Might need to write a class just for name
            data.append((name, sig))
    random.shuffle(data)
    return input_lang, output_lang, data
