import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import time
import math
import random
import argparse
import copy
import subprocess

from identifier_segmentor import segment
from prepare_data import readSigTokens, prepareData, start_token, unk_token, arrow_token, prepareDataWithFileName
from model import Encoder, Decoder, AttnDecoder, Model
from utils import *
from type_signatures import Tree

use_cuda = torch.cuda.is_available()
parser = argparse.ArgumentParser(description="train model")

parser.add_argument("--train_data", metavar="TRAIN DATA",
                    default="data/new_data/train_simple_sigs_parsable_normalized.txt",
                    )

parser.add_argument("--dev_data", metavar="DEV DATA",
                    default="data/new_data/dev_simple_sigs_parsable_normalized.txt",
                    )

parser.add_argument("--test_data", metavar="TEST DATA",
                    default="data/new_data/test_simple_sigs_parsable_normalized.txt",
                    )

parser.add_argument("--use_qualified_name", default=1, type=int,
                    help="0 for not using qualified name, 1 for using qualified name"
                    )
parser.add_argument("--use_full_path", default=0, type=int)

parser.add_argument("--train_data_qualified", metavar="QUALIFED TRAIN DATA",
                    default="data/new_data/train_simple_sigs_parsable_normalized.txt")

parser.add_argument("--dev_data_qualified", metavar="QUALIFED DEV DATA",
                    default="data/new_data/dev_simple_sigs_parsable_normalized.txt")

parser.add_argument("--test_data_qualified", metavar="QUALIFED TEST DATA",
                    default="data/new_data/test_simple_sigs_parsable_normalized.txt")

parser.add_argument("--model_state_file", default="model_state.pth")
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--eval_batch_size", default=8, type=int)
parser.add_argument("--hidden_size", default=256, type=int)
parser.add_argument("--embed_size", default=128, type=int)
parser.add_argument("--grad_clip", default=4.0, type=float)
parser.add_argument("--rec_depth", default=6, type=int)
parser.add_argument("--topo_loss_factor", default=1.0, type=float)
parser.add_argument("--num_epoch", default=50, type=int)
parser.add_argument("--dump_result", default=0, type=int)
parser.add_argument("--dev_result", default="results/dev_result.csv")


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
    indices = map(lambda x: lang.lookup(x.lower()), tokens)
    return indices

def variableFromName(name, lang):
    indices = indexFromName(name, lang)
    var = Variable(torch.LongTensor(indices)).unsqueeze(0)
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
    batch = map(lambda p: (indexFromName(p[0], input_lang), indexFromSignature(p[1], output_lang)), batch)
    batch = sorted(batch, key=lambda p:len(p[0]), reverse=True)
    max_input_len = len(batch[0][0])
    max_output_len = max(map(lambda p: len(p[1]), batch))
    input_lengths = map(lambda p: len(p[0]), batch)
    output_lengths = map(lambda p: len(p[1]), batch)
    batch = map(lambda p: (p[0] + (max_input_len - len(p[0])) * [0], p[1] + (max_output_len - len(p[1])) * [0]), batch)
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

def train(data, model, optimizer, dev_data=None, input_lang=None, output_lang=None):
    epoch_loss = 0.0
    start = time.time()
    model.train()
    for i, datum in enumerate(data):
        input_variable, sig_tree = datum
        input_len = input_variable.size(1)
        loss = model(input_variable, [input_len], is_train=True, reference=sig_tree)
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), arg.grad_clip)
        optimizer.step()
        epoch_loss += loss.data[0]
        if (i + 1) % 10000 == 0:
            print("checkpoint{} avg loss: {:.4f}".format((i+1)/10000, epoch_loss/(i+1)))
            print("time since start: {}".format(timeSince(start)))
        if (i + 1) % 100000 == 0 and dev_data is not None:
            randomEval(dev_data, model, input_lang, output_lang)
            model.train()
    epoch_loss /= len(data)
    print("epoch total training time: {}".format(timeSince(start)))
    return epoch_loss
'''
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
'''

def eval(data, input_vocab, output_vocab, model, is_test=False):
    model.eval()
    num_correct = 0
    num_structural_correct = 0
    eval_loss = 0
    dev_output = "results/dev_output"
    subprocess.call("rm {}".format(dev_output), shell=True)
    for name, sig in data:
        input_variable = variableFromName(name, input_vocab)
        input_len = input_variable.size(1)
        sig_tree = Tree.from_str(sig)
        if not is_test:
            index_tree = copy.deepcopy(sig_tree).to_index(output_vocab.token_to_idx, unk_token)
            loss = model(input_variable, [input_len], is_train=True, reference=index_tree)
            eval_loss += loss.data[0]
        gen_result = model(input_variable, [input_len])
        if gen_result.to_str(output_vocab.idx_to_token) == sig_tree:
            num_correct += 1
        if gen_result.structural_eq(sig_tree):
            num_structural_correct += 1
        with open(dev_output, "a+") as f:
            f.write("name: {}\n".format(name))
            f.write("sig: {}\n".format(sig))
            f.write("prediction: {}\n".format(gen_result.to_sig()))
            f.write("\n")
    if not is_test:
        return eval_loss/len(data), float(num_correct)/len(data), float(num_structural_correct)/len(data)
    return float(num_correct)/len(data), float(num_structural_correct)/len(data)

def eval_test(data, input_lang, output_lang, model):
    return eval(data, input_lang, output_lang, model, is_test=True)

def tokensToString(tokens, lang):
    if tokens[-1] == end_token:
        tokens = tokens[:-1]
    s = ""
    for token in tokens:
        s += lang.idx_to_token[token] + " "
    return s.rstrip()

# data should be in original format (i.e. string)
def randomEval(data, model, input_lang, output_lang):
    model.eval()
    name, sig = random.choice(data)
    input_variable = variableFromName(name, input_lang)
    gen_result = model(input_variable, [input_variable.size(1)])
    gen_result.to_str(output_lang.idx_to_token)
    predict_sig = gen_result.to_sig()
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
        _, _, dev_data = prepareDataWithFileName(arg.dev_data_qualified, use_full_path)
        _, _, test_data = prepareDataWithFileName(arg.test_data_qualified, use_full_path)
    else:
        input_lang, output_lang, train_data = prepareData(arg.train_data)
        _, _, dev_data = prepareData(arg.dev_data)
        _, _, test_data = prepareData(arg.test_data)

    train_data = map(lambda p: (variableFromName(p[0], input_lang), Tree.from_str(p[1]).to_index(output_lang.token_to_idx, unk_token)), train_data)

    weight = [10 * math.sqrt(1.0/output_lang.token_to_count[output_lang.idx_to_token[x]])
              for x in output_lang.idx_to_token if x > arrow_token]
    weight = [0] * (arrow_token + 1) + weight
    loss_weight = torch.FloatTensor(weight)
    # dump = arg.dump_result == 1
    model = Model(input_lang.n_word, output_lang.n_word, arg.embed_size,
                  arg.hidden_size, output_lang.kind_dict,
                  topo_loss_factor=arg.topo_loss_factor, rec_depth=arg.rec_depth,
                  weight=loss_weight)
    if use_cuda:
        model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    best_accuracy = 0
    best_model = model.state_dict()
    print("Start training...")
    for epoch in range(arg.num_epoch):
        try:
            epoch_loss = 0
            print("epoch {}/{}".format(epoch+1, arg.num_epoch))
            epoch_loss = train(train_data, model, optimizer, dev_data=dev_data, input_lang=input_lang, output_lang=output_lang)
            print("train loss: {:.4f}".format(epoch_loss))
            dev_loss, accuracy, structural_acc = eval(dev_data, input_lang, output_lang, model)
            print("dev loss: {:.4f} accuracy: {:.4f} structural accuracy: {:.4f}".format(dev_loss, accuracy, structural_acc))
            randomEval(dev_data, model, input_lang, output_lang)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model.state_dict()
                torch.save(best_model, arg.model_state_file)
        except KeyboardInterrupt:
            print("Keyboard Interruption.")
            break
    print("best accuracy: {:.4f}".format(best_accuracy))
    print("Start testing...")
    model.load_state_dict(torch.load(arg.model_state_file))
    accuracy, structural_acc = eval_test(test_data, input_lang, output_lang, model)
    print("test accuracy: {:.4f} structural accuracy: {:.4f}".format(accuracy, structural_acc))
    '''
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
    '''
if __name__ == "__main__":
    arg = parser.parse_args()
    main(arg)
