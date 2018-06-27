import time
import math

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
    return list(map(lambda x: x + (l - len(x)) * [0], batch))

def pad_nested_to_len(target_batch, l):
    # in place operation
    
    max_len = 0
    for indices in target_batch:
        for elem in indices:
            if max_len < len(elem):
                max_len = len(elem)

    for indices in target_batch:
        for elem in indices:
            elem += (max_len - len(elem)) * [0]
        indices += (l - len(indices)) * [max_len * [0]]

def process_sig(sig):
    new_sig = ""
    for token in sig.split():
        if token[0].isalpha():
            new_sig += token.split(".")[-1] + " "
        else:
            new_sig += token + " "
    return new_sig.rstrip()
