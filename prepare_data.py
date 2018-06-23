import identifier_segmentor
from type_signatures import UNPARSE_ARROW, parse_sig, sig_to_normalized_sexp, Tree
import random

SPECIAL_SYMBOLS = ["(", ")", "[", "]", ",", UNPARSE_ARROW, ":->", "<-->"]
start_token = 0
#end_token = 1
unk_token = 1
arrow_token = 2

class Lang:
    def __init__(self, is_input):
        self.token_to_idx = {"<TYPE>": 0, "<UNK>": 1, "->": 2}
        self.idx_to_token = {0: "<TYPE>", 1: "<UNK>", 2: "->"}
        self.token_to_count = {}
        self.n_word = 3
        self.is_input = is_input
        self.kind_dict = {}

    def add_token(self, token, kind=None):
        if not self.is_input:
            assert(kind is not None)
            # serialize by type + "#" + count
            token = token + "#" + str(kind)
            token_num = self.token_to_idx[token] if token in self.token_to_idx else self.n_word
            self.kind_dict[token_num] = kind
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

    def process_type(self, type):
        if type[0].isalpha():
            type = type.split('.')[-1]
        return type

    def add_sig_(self, tree):
        if type(tree.node) is str and tree.node != "->":
            token = self.process_type(tree.node)
            self.add_token(token, kind=0)
        elif type(tree.node) is list:
            token = self.process_type(tree.node[0])
            self.add_token(token, kind=len(tree.node)-1)
            for item in tree.node[1:]:
                self.add_sig(item)
        if tree.left is not None:
            self.add_sig_(tree.left)
        if tree.right is not None:
            self.add_sig_(tree.right)

    def add_sig(self, sig):
        assert(not self.is_input)
        if type(sig) is str:
            tree = Tree.from_str(sig)
        else:
            assert(isinstance(sig, Tree))
            tree = sig
        self.add_sig_(tree)

    def lookup(self, token):
        if token in self.token_to_idx:
            return self.token_to_idx[token]
        else:
            return unk_token

    # remove tokens that have frequency below a certain threshold
    def trim_tokens(self, threshold=5):
        for token in self.token_to_count.keys():
            if self.token_to_count[token] < threshold\
              and self.token_to_idx[token] > 2\
              and random.random() < 0.5:
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
            triple = ([input_name], sig, [sigs])
            data.append(triple)
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
    dir_name = filename[-2]
    if full_path:
        filename[-1] = filename[-1].split('.')[0]
    else:
        filename = [path.split('/')[-1].split('.')[0]]
    [name, rest] = name_and_sig.split(None, 1)
    [colon, sig] = rest.split(None, 1)
    assert(colon == "::")
    return int(num), filename, dir_name, name.strip(), sig.strip()

# processes data with qualified name (full path to the file) and signatures
def prepareDataWithFileName(filename, full_path=False, use_context=False, num_context_sig=3):
    with open(filename, 'r') as f:
        lines = filter(lambda l: len(l) > 0, f.read().split('\n'))
    input_lang = Lang(True)
    output_lang = Lang(False)
    data = []
    if use_context:
        context_sigs = []
        context_names = []
        for line in lines:
            num, fname, dir_name, input_name, sig = processLineWithFileName(line, full_path)
            cur_context = context_sigs[-num_context_sig:]
            cur_name_context = context_names[-num_context_sig:]
            sigs = [p[2] for p in cur_context if p[0] == num]
            names = [p[2] for p in cur_name_context if p[0] == num]
            name = fname + [input_name]
            datum = (name, sig, names, sigs)
            data.append(datum)
            context_sigs.append([num, dir_name, sig])
            context_names.append([num, dir_name, name])
            for ident in name:
                input_lang.add_name(ident)
            output_lang.add_sig(sig)
    else:
        for line in lines:
            _, fname, _, input_name, sig = processLineWithFileName(line, full_path)
            name = fname + [input_name]
            for ident in name:
                input_lang.add_name(ident)
            output_lang.add_sig(sig)
            data.append((name, sig))
    #random.shuffle(data)
    return input_lang, output_lang, data
