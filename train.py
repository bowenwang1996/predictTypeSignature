import torch
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn

import time
import random
import argparse

from utils import *
from prepare_data import start_token, end_token, unk_token, prepareData, readSigTokens, prepareDataWithFileName
from model import Model
from batch import Batch, TrainInfo
from beam import Beam
from type_signatures import Tree
from tensorboardX import SummaryWriter

use_cuda = torch.cuda.is_available()
#use_cuda = False

parser = argparse.ArgumentParser(description="train model")
exp_name = "unstructured"
writer = SummaryWriter("/share/data/vision-greg/whc/bowen/experiments/{}/log_dir".format(exp_name))

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
parser.add_argument("--num_epoch", default=30, type=int)
parser.add_argument("--lr", default=3e-4, type=float)
parser.add_argument("--dump_result", default=0, type=int)
parser.add_argument("--dev_result", default="results/dev_result.csv")
parser.add_argument("--test_results", default="results/test_result.csv")
parser.add_argument("--resume")
parser.add_argument("--checkpoint_dir")

def train(data, model, optimizer, epoch=0, checkpoint_base=0, best_accuracy=0):
    epoch_loss = 0.0
    start = time.time()
    model.train()
    model.batch.set_batch_size(model.train_batch_size)
    for i, trainInfo in enumerate(data):
        optimizer.zero_grad()
        loss = model(trainInfo)
        loss.backward()
        if arg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), arg.grad_clip)
        optimizer.step()
        epoch_loss += loss.item()/model.train_batch_size
        if (i+1) % 1000 == 0:
            checkpoint_num = (i+1) // 1000 + checkpoint_base
            writer.add_scalar("avg loss", epoch_loss/(i+1), epoch * 12 + checkpoint_num)
            print("checkpoint{} avg loss: {:.4f}".format(checkpoint_num, epoch_loss/(i+1)))
            print("time since start: {}".format(timeSince(start)))
            checkpoint_file = arg.checkpoint_dir + "checkpoint.pth"
            torch.save({"epoch": epoch,
                        "checkpoint": checkpoint_num,
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "best_acc": best_accuracy
                    }, checkpoint_file)
    epoch_loss /= len(data)
    print("epoch total training time:{}".format(timeSince(start)))
    return epoch_loss

def eval(data, model, is_test=False):
    if is_test:
        model.batch.set_batch_size(1)
    else:
        model.batch.set_batch_size(model.eval_batch_size)
    batch_size = model.batch.batch_size
    data = model.batch.batchify(data)
    num_correct = 0
    num_structural_correct = 0
    eval_loss = 0
    data_len = len(data)
    model.eval()
    for batch in data:
        trainInfo = model.batch.variableFromBatch(batch)
        if not is_test:
            loss, decoded_tokens = model(trainInfo, is_dev=True)
            eval_loss += loss.item()/model.eval_batch_size
        else:
            decoded_tokens = model(trainInfo, is_test=True)
        batch = sorted(batch, key=lambda p: len(model.batch.indexFromName(p[0])), reverse=True)
        for i in range(batch_size):
            _, sig, _ = batch[i]
            predict_sig = tokensToString([x.item() for x in decoded_tokens[i]], model.batch.target_vocab, trainInfo.idx_oov_dict)
            if predict_sig == process_sig(sig):
                num_correct += 1
            try:
                predict_tree = Tree.from_str(predict_sig)
                actual_tree = Tree.from_str(process_sig(sig))
                if predict_tree.structural_eq(actual_tree):
                    num_structural_correct += 1
            except ValueError:
                pass
    print("number of correct predictions: {}".format(num_correct))
    accuracy = float(num_correct)/(data_len * batch_size)
    structural_acc = num_structural_correct/(data_len * batch_size)
    if is_test:
        return accuracy, structural_acc
    return loss/data_len, accuracy, structural_acc

def eval_test(data, model):
    return eval(data, model, is_test=True)

def tokensToString(tokens, lang, idx_oov_dict):
    if tokens[-1] == end_token:
        tokens = tokens[:-1]
    s = ""
    for token in tokens:
        if token in lang.idx_to_token:
            s += lang.idx_to_token[token] + " "
        elif token in idx_oov_dict:
            s += idx_oov_dict[token] + " "
        else:
            s += "<unk> "
    return s.rstrip()

# data should be in original format (i.e. string)
def randomEval(data, model):
    datum = random.choice(data)
    model.batch.set_batch_size(1)
    trainInfo = model.batch.variableFromBatch([datum])
    decoded_tokens = model(trainInfo, is_test=True)
    decoded_token = [x.item() for x in decoded_tokens[0]]
    predict_sig = tokensToString(decoded_token, model.batch.target_vocab, trainInfo.idx_oov_dict)
    print("Name:{}".format(datum[0]))
    print("Sig:{}".format(datum[1]))
    print("Context:{}".format(datum[2]))
    print("Prediction:{}".format(predict_sig))

def main(arg):
    use_context = arg.use_qualified_name == 1
    if use_context:
        input_lang, output_lang, train_data = prepareDataWithFileName(arg.train_data_qualified, use_context=True)
        _, _, dev_data = prepareDataWithFileName(arg.dev_data_qualified, use_context=True)
        _, _, test_data = prepareDataWithFileName(arg.test_data_qualified, use_context=True)
    else:
        input_lang, output_lang, train_data = prepareData(arg.train_data)
        _, _, dev_data = prepareData(arg.dev_data)
        _, _, test_data = prepareData(arg.test_data)


    #output_lang.trim_tokens(threshold=2)
    print("Input vocab size: {}".format(input_lang.n_word))
    print("Target vocab size: {}".format(output_lang.n_word))

    criterion = nn.NLLLoss(reduce=False)
    model = Model(input_lang, output_lang, arg.embed_size, arg.hidden_size,
                  arg.batch_size, arg.eval_batch_size, criterion
                  )
    if use_cuda:
        model = model.cuda()

    train_data = list(map(lambda p: model.batch.variableFromBatch(p), model.batch.batchify(train_data)))

    optimizer = optim.Adam(model.parameters(), lr=arg.lr)
    optimizer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True, factor=0.5)

    best_accuracy = 0
    best_loss = float('inf')
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
            print("epoch {}/{}".format(epoch+1, arg.num_epoch))
            epoch_loss = train(train_data[checkpoint_num*1000:], model, optimizer, 
                               epoch=epoch, checkpoint_base=checkpoint_num, 
                               best_accuracy=best_accuracy)
            writer.add_scalar("Train loss", epoch_loss, epoch+1)
            print("train loss: {:.4f}".format(epoch_loss))
            checkpoint_num = 0

            dev_loss, accuracy, structural_acc = eval(dev_data, model)
            writer.add_scalar("Dev loss", dev_loss, epoch+1)
            writer.add_scalar("Dev acc", accuracy, epoch+1)
            writer.add_scalar("Dev structural acc", structural_acc, epoch+1)
            print("dev loss: {:.4f} accuracy: {:.4f} structural accuracy: {:.4f}".format(dev_loss, accuracy, structural_acc))

            optimizer_scheduler.step(dev_loss)
            randomEval(dev_data, model)

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
    test_accuracy, test_structural_acc = eval_test(test_data, model)
    writer.add_scalar("Test acc", test_accuracy)
    writer.add_scalar("Test structural acc", test_structural_acc)
    print("test accuracy: {:.4f} test structural accuracy: {:.4f}".format(test_accuracy, test_structural_acc))

if __name__ == "__main__":
    arg = parser.parse_args()
    main(arg)
