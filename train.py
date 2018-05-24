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
from shutil import copyfile

from identifier_segmentor import segment
from prepare_data import readSigTokens, prepareData, start_token, unk_token, arrow_token, prepareDataWithFileName
from model import Encoder, Decoder, AttnDecoder, Model
from utils import *
from type_signatures import Tree

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
parser.add_argument("--checkpoint_dir", default="./checkpoints/")
parser.add_argument("--resume", default=None)

def indexFromName(name, lang):
    tokens = []
    for ident in name:
        tokens += segment(ident)
    indices = [lang.lookup(x.lower()) for x in tokens]
    return indices

def variableFromName(name, lang):
    indices = indexFromName(name, lang)
    var = torch.tensor(indices, dtype=torch.long, requires_grad=True).unsqueeze(0).to(device)
    return var

def train(data, model, optimizer,
          epoch=0, checkpoint_base=0, best_accuracy=0,
          dev_data=None, input_lang=None, output_lang=None):
    epoch_loss = 0.0
    start = time.time()
    model.train()
    for i, datum in enumerate(data):
        optimizer.zero_grad()
        input_variable, sig_tree = datum
        input_len = input_variable.size(1)
        loss = model(input_variable, [input_len], is_train=True, reference=sig_tree)
        loss.backward()
        if arg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), arg.grad_clip)
        optimizer.step()
        epoch_loss += loss.item()
        if (i + 1) % 10000 == 0:
            checkpoint_num = (i + 1) // 10000 + checkpoint_base
            print("checkpoint{} avg loss: {:.4f}".format(checkpoint_num, epoch_loss/(i+1)))
            print("time since start: {}".format(timeSince(start)))
            cur_checkpoint = arg.checkpoint_dir + "checkpoint{}-{}.pth".format(epoch, checkpoint_num)
            torch.save({"epoch": epoch,
                        "checkpoint": checkpoint_num,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_acc": best_accuracy
                        }, cur_checkpoint)
            copyfile(cur_checkpoint, arg.checkpoint_dir + "checkpoint.pth")
        if (i + 1) % 100000 == 0 and dev_data is not None:
            randomEval(dev_data, model, input_lang, output_lang)
            model.train()
    epoch_loss /= len(data)
    print("epoch total training time: {}".format(timeSince(start)))
    return epoch_loss

def eval(data, input_vocab, output_vocab, model, is_test=False):
    model.eval()
    num_correct = 0
    num_structural_correct = 0
    eval_loss = 0
    for name, sig in data:
        input_variable = variableFromName(name, input_vocab)
        input_len = input_variable.size(1)
        sig_tree = Tree.from_str(process_sig(sig))
        if not is_test:
            index_tree = copy.deepcopy(sig_tree).to_index(output_vocab.token_to_idx, unk_token)
            loss = model(input_variable, [input_len], is_train=True, reference=index_tree)
            eval_loss += loss.item()
        gen_result = model(input_variable, [input_len])
        if gen_result.to_str(output_vocab.idx_to_token) == sig_tree:
            num_correct += 1
        if gen_result.structural_eq(sig_tree):
            num_structural_correct += 1
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

    train_data = [(variableFromName(p[0], input_lang), Tree.from_str(p[1]).to_index(output_lang.token_to_idx, unk_token)) for p in train_data]

    '''
    weight = [10 * math.sqrt(1.0/output_lang.token_to_count[output_lang.idx_to_token[x]])
              for x in output_lang.idx_to_token if x > arrow_token]
    weight = [0] * (arrow_token + 1) + weight
    loss_weight = torch.FloatTensor(weight)
    '''
    # dump = arg.dump_result == 1
    model = Model(input_lang.n_word, output_lang.n_word, arg.embed_size,
                  arg.hidden_size, output_lang.kind_dict,
                  topo_loss_factor=arg.topo_loss_factor, rec_depth=arg.rec_depth,
                  weight=None)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    best_accuracy = 0
    best_model = model.state_dict()
    epoch_start = 0
    checkpoint_num = 0
    print("Start training...")
    if arg.resume is not None:
        print("loading from {}".format(arg.resume))
        checkpoint = torch.load(arg.checkpoint_dir + arg.resume)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_accuracy = checkpoint["best_acc"]
        epoch_start = checkpoint["epoch"]
        checkpoint_num = checkpoint["checkpoint"]

    for epoch in range(epoch_start, arg.num_epoch):
        try:
            epoch_loss = 0
            print("epoch {}/{}".format(epoch+1, arg.num_epoch))
            if checkpoint_num != 0:
                epoch_loss = train(train_data[checkpoint_num*10000:], model, optimizer,
                                   epoch=epoch, checkpoint_base=checkpoint_num, best_accuracy=best_accuracy,
                                   dev_data=dev_data, input_lang=input_lang, output_lang=output_lang)
            else:
                epoch_loss = train(train_data, model, optimizer,
                                   epoch=epoch, dev_data=dev_data, best_accuracy=best_accuracy,
                                   input_lang=input_lang, output_lang=output_lang)
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

if __name__ == "__main__":
    arg = parser.parse_args()
    main(arg)
