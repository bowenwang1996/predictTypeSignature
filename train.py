import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import math
import random
import argparse

from identifier_segmentor import segment
from prepare_data import readSigTokens, prepareData, start_token, end_token, unk_token, prepareDataWithFileName
from model import Encoder, Decoder, AttnDecoder
from utils import *

use_cuda = torch.cuda.is_available()
parser = argparse.ArgumentParser(description="train model")

parser.add_argument("--train_data", metavar="TRAIN DATA",
                    default="data/train_simple_sigs_parsable_normalized.txt",
                    )

parser.add_argument("--dev_data", metavar="DEV DATA",
                    default="data/dev_simple_sigs_parsable_normalized.txt",
                    )

parser.add_argument("--test_data", metavar="TEST DATA",
                    default="data/test_simple_sigs_parsable_normalized.txt",
                    )

parser.add_argument("--type_token_data", metavar="TYPE TOKEN",
                    default="data/simple_types_vocab.txt"
                    )

parser.add_argument("--use_qualified_name", default=0, type=int,
                    help="0 for not using qualified name, 1 for using qualified name"
                    )
parser.add_argument("--use_full_path", default=0, type=int)

parser.add_argument("--train_data_qualified", metavar="QUALIFED TRAIN DATA",
                    default="data/data_with_file_names/train_simple_sigs_parsable_normalized.txt")

parser.add_argument("--dev_data_qualified", metavar="QUALIFED DEV DATA",
                    default="data/data_with_file_names/dev_simple_sigs_parsable_normalized.txt")

parser.add_argument("--test_data_qualified", metavar="QUALIFED TEST DATA",
                    default="data/data_with_file_names/test_simple_sigs_parsable_normalized.txt")

parser.add_argument("--encoder_state_file", default="encoder_state.pth")
parser.add_argument("--decoder_state_file", default="decoder_state.pth")
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--eval_batch_size", default=8, type=int)
parser.add_argument("--hidden_size", default=256, type=int)
parser.add_argument("--grad_clip", default=4.0, type=float)
parser.add_argument("--num_epoch", default=50, type=int)
parser.add_argument("--dump_result", default=0, type=int)
parser.add_argument("--dev_result", default="results/dev_result.csv")
parser.add_argument("--test_results", default="results/test_result.csv")

def indexFromSignature(sig, lang):
    tokens = sig.split()
    def foo(x):
        if x[0].isupper():
            token = x.split('.')[-1]
        else:
            token = x
        return lang.lookup(token)
    indices = list(map(foo, tokens))
    indices.append(end_token)
    return indices

def indexFromSignatures(sigs, lang):
    '''
    this function is used for getting indices of context signatures.
    Therefore no need to add end_token
    '''
    indices = []
    for sig in sigs:
        indices += indexFromSignature(sig, lang)[:-1]
    if len(indices) == 0:
        indices.append(unk_token) # a hack to avoid 0 length indices
    return indices

def variableFromSignature(sig, lang):
    indices = indexFromSignature(sig, lang)
    var = Variable(torch.LongTensor(indices))
    if use_cuda:
        var = var.cuda()
    return var

def variableFromSignatures(sigs, lang):
    indices = indexFromSignatures(sigs, lang)
    var = Variable(torch.LongTensor(indices))
    if use_cuda:
        var = var.cuda()
    return var

def indexFromName(name, lang):
    tokens = []
    for ident in name:
        tokens += segment(ident)
    indices = list(map(lambda x: lang.lookup(x.lower()), tokens))
    indices.append(end_token)
    return indices

def variableFromName(name, lang):
    indices = indexFromName(name, lang)
    var = Variable(torch.LongTensor(indices))
    if use_cuda:
        var = var.cuda()
    return var

def batchify(data, batch_size):
    data_len = len(data)
    num_batches = data_len / batch_size
    batches = []
    for i in range(num_batches):
        batches.append(data[batch_size*i:batch_size*(i+1)])
    return batches

def variableFromBatch(batch, input_lang, output_lang):
    batch_size = len(batch)
    batch = list(map(lambda p: (indexFromName(p[0], input_lang), indexFromSignature(p[1], output_lang)), batch))
    batch = sorted(batch, key=lambda p:len(p[0]), reverse=True)
    max_input_len = len(batch[0][0])
    max_output_len = max(list(map(lambda p: len(p[1]), batch)))
    input_lengths = map(lambda p: len(p[0]), batch)
    output_lengths = list(map(lambda p: len(p[1]), batch))
    batch = list(map(lambda p: (p[0] + (max_input_len - len(p[0])) * [0], p[1] + (max_output_len - len(p[1])) * [0]), batch))
    input_batch = torch.LongTensor(batch_size, max_input_len)
    output_batch = torch.LongTensor(batch_size, max_output_len)
    for i in range(batch_size):
        input_batch[i] = torch.LongTensor(batch[i][0])
        output_batch[i] = torch.LongTensor(batch[i][1])
    input_variable = Variable(input_batch)
    output_variable = Variable(output_batch)
    if use_cuda:
        input_variable = input_variable.cuda()
        output_variable = output_variable.cuda()
    return input_variable, input_lengths, output_variable, output_lengths


# I'm aware that writing a function with 10 arguments is bad style
def step(input_variable, input_lengths, target_variable, target_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, is_train):
    if is_train:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()
        
    batch_size = input_variable.size(0)
    encoder_hidden = encoder.initHidden(batch_size)
    if use_cuda:
        encoder_hidden = (encoder_hidden[0].cuda(), encoder_hidden[1].cuda())
    input_len = input_variable.size()[1]
    target_len = target_variable.size()[1]

    encoder_outputs, encoder_hidden = encoder(input_variable, input_lengths, encoder_hidden)
    decoder_hidden = (encoder_hidden[0].view(1, batch_size, -1), encoder_hidden[1].view(1, batch_size, -1))
    loss = 0.0
    decoder_input = Variable(torch.LongTensor(batch_size, 1).fill_(start_token))
    length_tensor = torch.LongTensor(target_lengths)
    # at the begining, no sequence has ended so we use -1 as its end index
    eos_tensor = torch.LongTensor(batch_size).fill_(-1)
    if use_cuda:
        decoder_input = decoder_input.cuda()
        length_tensor = length_tensor.cuda()
        eos_tensor = eos_tensor.cuda()
    for i in range(target_len):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
        cur_loss = criterion(decoder_output, target_variable[:, i])
        loss_mask = (length_tensor > i) & (eos_tensor == -1)
        loss_mask = Variable(loss_mask).cuda() if use_cuda else Variable(loss_mask)
        loss += torch.masked_select(cur_loss, loss_mask).sum()
        _, topi = decoder_output.data.topk(1, dim=1)
        next_in = topi
        end_mask = next_in.squeeze(1) == end_token
        end_mask = end_mask & (eos_tensor == -1)
        eos_tensor.masked_fill_(end_mask, i)
        decoder_input = Variable(next_in)
        if use_cuda:
            decoder_input = decoder_input.cuda()

    if is_train:
        loss.backward()

        clip = 4
        torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
        
        encoder_optimizer.step()
        decoder_optimizer.step()
    return loss.data[0]/batch_size

def train_step(input_variable, input_lengths, output_variable, output_lengths,
               encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    return step(input_variable, input_lengths, output_variable, output_lengths,
                encoder, decoder, encoder_optimizer, decoder_optimizer,
                criterion, True)

def eval_step(input_variable, input_lengths, output_variable, output_lengths, encoder, decoder, criterion):
    return step(input_variable, input_lengths, output_variable, output_lengths,
                encoder, decoder, None, None,
                criterion, False)

def generate_step(input_variable, input_lengths, encoder, decoder, max_length = 30):
    encoder.eval()
    decoder.eval()

    batch_size = input_variable.size(0)
    encoder_hidden = encoder.initHidden(batch_size)
    if use_cuda:
        encoder_hidden = (encoder_hidden[0].cuda(), encoder_hidden[1].cuda())

    encoder_outputs, encoder_hidden = encoder(input_variable, input_lengths, encoder_hidden)
    decoder_hidden = (encoder_hidden[0].view(1, batch_size, -1), encoder_hidden[1].view(1, batch_size, -1))
    decoder_in = Variable(torch.LongTensor(batch_size, 1).fill_(start_token))
    # -1 for not taken
    decoded_tokens = torch.LongTensor(batch_size, max_length).fill_(-1)
    eos_tensor = torch.LongTensor(batch_size).fill_(-1)
    if use_cuda:
        decoder_in = decoder_in.cuda()
        decoded_tokens = decoded_tokens.cuda()
        eos_tensor = eos_tensor.cuda()
    for i in range(max_length):
        decoder_out, decoder_hidden = decoder(decoder_in, decoder_hidden, encoder_outputs)
        _, topi = decoder_out.data.topk(1, dim=1)
        pred = topi[:, 0]
        decoded_tokens[:, i] = pred
        decoded_tokens[:, i].masked_fill_(eos_tensor > -1, -1)
        end_mask = pred == end_token
        eos_tensor.masked_fill_(end_mask & (eos_tensor == -1), i+1)
        decoder_in = Variable(pred.unsqueeze(1))
    return decoded_tokens, eos_tensor

def train(data, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    epoch_loss = 0.0
    start = time.time()
    for i, pair in enumerate(data):
        input_variable, input_lengths, output_variable, output_lengths = pair
        epoch_loss += train_step(input_variable, input_lengths, output_variable, output_lengths, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        if (i+1) % 1000 == 0:
            print("checkpoint{} avg loss: {:.4f}".format((i+1)/1000, epoch_loss/(i+1)))
            print("time since start: {}".format(timeSince(start)))
    epoch_loss /= len(data)
    print("epoch total training time:{}".format(timeSince(start)))
    return epoch_loss

def eval(data, input_lang, output_lang, encoder, decoder, criterion, is_test=False):
    num_correct = 0
    loss = 0
    data_len = len(data)
    batch_size = None
    for batch in data:
        input_variable, input_lengths, output_variable, output_lengths = variableFromBatch(batch, input_lang, output_lang)
        decoded_tokens, eos_tensor = generate_step(input_variable, input_lengths, encoder, decoder)
        batch_size = input_variable.size(0)
        batch = sorted(batch, key=lambda p: len(indexFromName(p[0], input_lang)), reverse=True)
        for i in range(batch_size):
            _, sig = batch[i]
            predict_sig = tokensToString(decoded_tokens[i, :eos_tensor[i]], output_lang)
            if predict_sig == process_sig(sig):
                num_correct += 1
        if not is_test:
            loss += eval_step(input_variable, input_lengths, output_variable, output_lengths, encoder, decoder, criterion)
    accuracy = float(num_correct)/(data_len * batch_size)
    if is_test:
        return accuracy
    else:
        return loss/data_len, accuracy

def eval_test(data, input_lang, output_lang, encoder, decoder):
    return eval(data, input_lang, output_lang, encoder, decoder, None, is_test=True)

def tokensToString(tokens, lang):
    if tokens[-1] == end_token:
        tokens = tokens[:-1]
    s = ""
    for token in tokens:
        s += lang.idx_to_token[token] + " "
    return s.rstrip()

# data should be in original format (i.e. string)
def randomEval(data, encoder, decoder, input_lang, output_lang):
    name, sig = random.choice(data)
    input_variable = variableFromName(name, input_lang)
    input_lengths = [len(input_variable)]
    decoded_tokens, eos_tensor = generate_step(input_variable.unsqueeze(0), input_lengths, encoder, decoder)
    decoded_tokens = decoded_tokens.squeeze(0)[:eos_tensor[0]].tolist()
    predict_sig = tokensToString(decoded_tokens, output_lang)
    print(name)
    print(sig)
    print(predict_sig)

# data should be in original format (i.e. string)
def resultDump(data, encoder, decoder, input_lang, output_lang):
    results = []
    for name, sig in data:
        input_variable = variableFromName(name, input_lang)
        input_lengths = [len(input_variable)]
        target_indices = indexFromSignature(sig, output_lang)
        decoded_tokens, eos_tensor = generate_step(input_variable.unsqueeze(0), input_lengths, encoder, decoder)
        decoded_tokens = decoded_tokens.squeeze(0)[:eos_tensor[0]].tolist()
        if decoded_tokens == target_indices:
            results.append((name, sig))
    return results

def main(arg):
    if arg.use_qualified_name == 1:
        use_full_path = arg.use_full_path == 1
        input_lang, output_lang, train_data = prepareDataWithFileName(arg.train_data_qualified, use_full_path)
        _, _, original_dev_data = prepareDataWithFileName(arg.dev_data_qualified, use_full_path)
        _, _, original_test_data = prepareDataWithFileName(arg.test_data_qualified, use_full_path)
    else:
        input_lang, output_lang, train_data = prepareData(arg.train_data)
        _, _, original_dev_data = prepareData(arg.dev_data)
        _, _, original_test_data = prepareData(arg.test_data)

    train_data = list(map(lambda p: variableFromBatch(p, input_lang, output_lang), batchify(train_data, arg.batch_size)))
    dev_data = batchify(original_dev_data, arg.eval_batch_size)
    #dev_data = map(lambda p: variableFromBatch(p, input_lang, output_lang), batchify(original_dev_data, arg.eval_batch_size))

    dump = arg.dump_result == 1
    encoder = Encoder(input_lang.n_word, arg.hidden_size)
    decoder = AttnDecoder(output_lang.n_word, arg.hidden_size)
    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=2e-4)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=2e-4)
    criterion = nn.NLLLoss(reduce=False)
    
    best_accuracy = 0
    best_model = (encoder.state_dict(), decoder.state_dict())
    print("Start training...")
    for epoch in range(arg.num_epoch):
        try:
            epoch_loss = 0
            print("epoch {}/{}".format(epoch+1, arg.num_epoch))
            epoch_loss = train(train_data, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            print("train loss: {:.4f}".format(epoch_loss))
            dev_loss, accuracy = eval(dev_data, input_lang, output_lang, encoder, decoder, criterion)
            print("dev loss: {:.4f} accuracy: {:.4f}".format(dev_loss, accuracy))
            randomEval(original_dev_data, encoder, decoder, input_lang, output_lang)
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = (encoder.state_dict(), decoder.state_dict())
                torch.save(best_model[0], arg.encoder_state_file)
                torch.save(best_model[1], arg.decoder_state_file)
        except KeyboardInterrupt:
          print("Keyboard Interruption.")
          break
    print("best accuracy: {:.4f}".format(best_accuracy))
    print("Start testing...")
    #test_data = map(lambda p: variableFromBatch(p, input_lang, output_lang), batchify(original_test_data, arg.eval_batch_size))
    test_data = batchify(original_test_data, arg.eval_batch_size)
    encoder.load_state_dict(torch.load(arg.encoder_state_file))
    decoder.load_state_dict(torch.load(arg.decoder_state_file))
    test_accuracy = eval_test(test_data, input_lang, output_lang, encoder, decoder)
    print("test accuracy: {:.4f}".format(test_accuracy))
    if dump:
        import csv
        dev_results = resultDump(dev_data, encoder, decoder, input_lang, output_lang)
        test_results = resultDump(test_data, encoder, decoder, input_lang, output_lang)
        def write_results(results, filename):
            with open(filename, "w+") as csvfile:
                writer = csv.writer(csvfile)
                header = None
                name = results[0][0]
                if len(name) == 1:
                    header = ["Name", "Signature"]
                elif len(name) == 2:
                    header = ["Module Name", "Name", "Signature"]
                if header is not None:
                    writer.writerow(header)
                for name, sig in results:
                    row = name + [sig]
                    writer.writerow(row)
        write_results(dev_results, arg.dev_result)
        write_results(test_results, arg.test_results)
    
if __name__ == "__main__":
    arg = parser.parse_args()
    main(arg)
    
