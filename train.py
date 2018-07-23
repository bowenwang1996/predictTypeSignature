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
from model import Model
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
parser.add_argument("--use_context", default=0, type=int)
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
parser.add_argument("--lr", default=2e-4, type=float)
parser.add_argument("--grad_clip", default=4.0, type=float)
parser.add_argument("--rec_depth", default=6, type=int)
parser.add_argument("--dropout", default=0.0, type=float)
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
    var = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)
    return var


class ContextInfo():
    '''
    a class for holding context information (names and sigs)
    '''

    def __init__(self, input_vocab, target_vocab, names, sigs):
        indices = [indexFromName(name, input_vocab) for name in names]
        name_vars = [torch.LongTensor(x).to(device) for x in indices]

        sigs_indices = []
        sig_trees = []
        oov_token_to_idx = {}
        oov_idx_to_token = {}
        oov_kind_dict = {}
        for sig in sigs:
            sig_indices = []
            sig = process_sig(sig)
            sig_tree = Tree.from_str(sig)\
                           .to_index_augment(target_vocab.token_to_idx,
                                             unk_token,
                                             oov_token_to_idx,
                                             oov_idx_to_token,
                                             oov_kind_dict)
            # this is cumbersome
            sig_trees.append(sig_tree)
            tree = Tree.from_str(sig)
            tree.decorate()
            node_map = tree.traversal(ignore_node="->")
            for _, v in node_map.items():
                if v in target_vocab.token_to_idx:
                    sig_indices.append(target_vocab.token_to_idx[v])
                else:
                    assert(v in oov_token_to_idx)
                    sig_indices.append(oov_token_to_idx[v])
            sigs_indices.append(torch.tensor(sig_indices, dtype=torch.long).to(device))

        self.names = name_vars
        self.sigs = sig_trees
        self.indices = sigs_indices
        self.num = len(names)
        self.oov_token_to_idx = oov_token_to_idx
        self.oov_idx_to_token = oov_idx_to_token
        self.oov_kind_dict = oov_kind_dict

def train(data, model, optimizer,
          epoch=0, checkpoint_base=0, best_accuracy=0,
          dev_data=None, input_lang=None, output_lang=None):
    epoch_loss = 0.0
    start = time.time()
    model.train()
    for i, datum in enumerate(data):
        optimizer.zero_grad()
        input_variable, sig_tree, context_info = datum
        input_len = input_variable.size(1)
        loss = model(input_variable, [input_len], context_info, is_train=True, reference=sig_tree)
        loss.backward()
        if arg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), arg.grad_clip)
        optimizer.step()
        epoch_loss += loss.item()
        if (i + 1) % 10000 == 0:
            checkpoint_num = (i + 1) // 10000 + checkpoint_base
            print("checkpoint{} avg loss: {:.4f}".format(checkpoint_num, epoch_loss/(i+1)))
            print("time since start: {}".format(timeSince(start)))
            checkpoint_file = arg.checkpoint_dir + "checkpoint.pth"
            torch.save({"epoch": epoch,
                        "checkpoint": checkpoint_num,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_acc": best_accuracy
                        }, checkpoint_file)
        if (i + 1) % 100000 == 0 and dev_data is not None:
            randomEval(dev_data, model, input_lang, output_lang)
            model.train()
    epoch_loss /= len(data)
    print("epoch total training time: {}".format(timeSince(start)))
    return epoch_loss

def eval(data, input_vocab, output_vocab, model, is_test=False, out_file=None, dict_out=None):
    model.eval()
    num_correct = 0
    num_structural_correct = 0
    eval_loss = 0
    depth_dict = {}
    for name, sig, context_names, context_sigs in data:
        input_variable = variableFromName(name, input_vocab)
        input_len = input_variable.size(1)
        sig_tree = Tree.from_str(process_sig(sig))
        if context_names == []:
            context_info = None
        else:
            context_info = ContextInfo(input_vocab, output_vocab, context_names, context_sigs)
        if not is_test:
            index_tree = copy.deepcopy(sig_tree)\
                             .to_index(output_vocab.token_to_idx, unk_token)
            loss = model(input_variable, [input_len], context_info,
                         is_train=True, reference=index_tree)
            eval_loss += loss.item()
        gen_result = model(input_variable, [input_len], context_info)
        if context_info:
            gen_result.to_str(output_vocab.idx_to_token, oov_dict=context_info.oov_idx_to_token)
        else:
            gen_result.to_str(output_vocab.idx_to_token)
        #predict_sig = gen_result.to_sig()
        #predict_sig_tree = Tree.from_str(predict_sig)
        if gen_result == sig_tree:
            num_correct += 1
            depth = sig_tree.depth()
            if depth in depth_dict:
                depth_dict[depth] += 1
            else:
                depth_dict[depth] = 1
        if gen_result.structural_eq(sig_tree):
            num_structural_correct += 1
        if out_file is not None:
            with open(out_file, "a+") as f:
                if context_info is not None:
                    for context_name, context_sig in zip(context_names, context_sigs):
                        f.write("name: {} sig: {}\n".format(context_name, context_sig))
                f.write("name: {}\n".format(name))
                f.write("sig: {}\n".format(process_sig(sig)))
                f.write("prediction: {}\n".format(gen_result.to_sig()))
                f.write("\n")
    if dict_out is not None:
        with open(dict_out, "w+") as f:
            f.write("{}\n".format(depth_dict))
    if not is_test:
        return eval_loss/len(data), float(num_correct)/len(data), float(num_structural_correct)/len(data)
    return float(num_correct)/len(data), float(num_structural_correct)/len(data)

def eval_test(data, input_lang, output_lang, model, out_file=None, dict_out=None):
    return eval(data, input_lang, output_lang, model,
                is_test=True, out_file=out_file, dict_out=dict_out)

# data should be in original format (i.e. string)
def randomEval(data, model, input_lang, output_lang):
    model.eval()
    name, sig, context_names, context_sigs = random.choice(data)
    input_variable = variableFromName(name, input_lang)
    if context_names == []:
        context_info = None
    else:
        context_info = ContextInfo(input_lang, output_lang, context_names, context_sigs)
    gen_result = model(input_variable, [input_variable.size(1)], context_info)
    if context_info:
        gen_result.to_str(output_lang.idx_to_token, oov_dict=context_info.oov_idx_to_token)
    else:
        gen_result.to_str(output_lang.idx_to_token)
    predict_sig = gen_result.to_sig()
    for context_name, context_sig in zip(context_names, context_sigs):
        print("name: {} sig: {}".format(context_name, context_sig))
    print(name)
    print(process_sig(sig))
    print(predict_sig)

def process_data(datum, input_vocab, target_vocab):
    '''
    a helper function for mapping raw data into processed data
    '''
    name, sig, context_names, context_sigs = datum
    if context_names == []:
        return (variableFromName(name, input_vocab),
                Tree.from_str(sig).to_index(target_vocab.token_to_idx, unk_token),
                None
                )
    return (variableFromName(name, input_vocab),
            Tree.from_str(sig).to_index(target_vocab.token_to_idx, unk_token),
            ContextInfo(input_vocab, target_vocab, context_names, context_sigs)
            )

def main(arg):
    use_context = arg.use_context == 1
    if arg.use_qualified_name == 1:
        use_full_path = arg.use_full_path == 1
        input_lang, output_lang, train_data = prepareDataWithFileName(arg.train_data_qualified, use_full_path, use_context=use_context)
        _, _, dev_data = prepareDataWithFileName(arg.dev_data_qualified, use_full_path, use_context=use_context)
        _, _, test_data = prepareDataWithFileName(arg.test_data_qualified, use_full_path, use_context=use_context)
    else:
        input_lang, output_lang, train_data = prepareData(arg.train_data, use_context=use_context)
        _, _, dev_data = prepareData(arg.dev_data, use_context=use_context)
        _, _, test_data = prepareData(arg.test_data, use_context=use_context)

    train_data = [process_data(d, input_lang, output_lang) for d in train_data]

    '''
    weight = [10 * math.sqrt(1.0/output_lang.token_to_count[output_lang.idx_to_token[x]])
              for x in output_lang.idx_to_token if x > arrow_token]
    weight = [0] * (arrow_token + 1) + weight
    loss_weight = torch.FloatTensor(weight)
    '''

    model = Model(input_lang.n_word, output_lang.n_word, arg.embed_size,
                  arg.hidden_size, output_lang.kind_dict, dropout_p=arg.dropout,
                  topo_loss_factor=arg.topo_loss_factor, rec_depth=arg.rec_depth,
                  weight=None)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=arg.lr)

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
            epoch_loss = train(train_data[checkpoint_num*10000:], model, optimizer,
                               epoch=epoch, checkpoint_base=checkpoint_num, best_accuracy=best_accuracy,
                               dev_data=dev_data, input_lang=input_lang, output_lang=output_lang)
            checkpoint_num = 0
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
