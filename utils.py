import time
import math
import torch
from torch.autograd import Variable

# thanks to https://stackoverflow.com/questions/6422700/how-to-get-indices-of-a-sorted-array-in-python
def get_sort_index(arr):
    return [i[0] for i in sorted(enumerate(arr), key=lambda x:x[1], reverse=True)]

# invert a permutation. Thanks to https://stackoverflow.com/questions/9185768/inverting-permutations-in-python
def inv(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

def asMinutes(s):
    m = math.floor(s/60)
    s -= m * 60
    return "%dm %ds" % (m, s)

def timeSince(since):
    now = time.time()
    s = now - since
    return "%s" % asMinutes(s)

# pad each list in the batch to length l
def pad_to_len(batch, l):
    return [x + (l - len(x)) * [0] for x in batch]

def process_sig(sig):
    new_sig = ""
    for token in sig.split():
        if token[0].isalpha():
            new_sig += token.split(".")[-1] + " "
        else:
            new_sig += token + " "
    return new_sig.rstrip()

def singleton_variable(token, batch_size):
    if batch_size == 0:
        var = torch.tensor([token], dtype=torch.long)
    else:
        var = torch.full((batch_size, 1), token, dtype=torch.long)
    return var
