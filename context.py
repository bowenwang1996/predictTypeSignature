import torch
from torch.autograd import Variable
import torch.optim as optim

import time
import random
import argparse

from utils import *
from prepare_data import start_token, end_token, unk_token, prepareData, readSigTokens, prepareDataWithFileName
from model import *
from batch import Batch, TrainInfo
from beam import Beam
from type_signatures import Tree
from tensorboardX import SummaryWriter

use_cuda = torch.cuda.is_available()
#use_cuda = False

parser = argparse.ArgumentParser(description="train model")
exp_name = "baseline_copy"
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

parser.add_argument("--encoder_state_file", default="encoder_state.pth")
parser.add_argument("--context_encoder_state_file", default="context_encoder_state.pth")
parser.add_argument("--decoder_state_file", default="decoder_state.pth")
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--eval_batch_size", default=8, type=int)
parser.add_argument("--hidden_size", default=256, type=int)
parser.add_argument("--embed_size", default=128, type=int)
parser.add_argument("--grad_clip", default=4.0, type=float)
parser.add_argument("--num_epoch", default=30, type=int)
parser.add_argument("--dump_result", default=0, type=int)
parser.add_argument("--dev_result", default="results/dev_result.csv")
parser.add_argument("--test_results", default="results/test_result.csv")
parser.add_argument("--resume")
parser.add_argument("--checkpoint_dir")

def step(trainInfo, batch, encoder, context_encoder, decoder, encoder_optimizer, context_encoder_optimizer, decoder_optimizer, criterion, is_train):
    if is_train:
        encoder_optimizer.zero_grad()
        context_encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoder.train()
        context_encoder.train()
        decoder.train()
    else:
        encoder.eval()
        context_encoder.eval()
        decoder.eval()

    batch_size = batch.batch_size
    encoder_hidden = encoder.initHidden(batch_size)
    context_encoder_hidden = context_encoder.initHidden(batch_size)
    if use_cuda:
        encoder_hidden = (encoder_hidden[0].cuda(), encoder_hidden[1].cuda())
        context_encoder_hidden = (context_encoder_hidden[0].cuda(), context_encoder_hidden[1].cuda())
    target_len = trainInfo.target_variable.size(1)

    encoder_outputs, encoder_hidden = encoder(trainInfo.input_variable,
                                              trainInfo.input_lengths,
                                              encoder_hidden)
    context_variable = batch.unk_batch(trainInfo.context_variable)  # for feeding to context encoder
    context_encoder_outputs, context_encoder_hidden = context_encoder(context_variable, trainInfo.context_lengths, trainInfo.context_sort_index, trainInfo.context_inv_index, context_encoder_hidden)
    '''
    decoder_hidden = (encoder_hidden[0].view(1, batch_size, -1),
                      encoder_hidden[1].view(1, batch_size, -1))
    '''
    decoder_hidden = (encoder_hidden[0].view(1, batch_size, -1) + context_encoder_hidden[0].view(1, batch_size, -1),
                      encoder_hidden[1].view(1, batch_size, -1) + context_encoder_hidden[1].view(1, batch_size, -1))
    '''
    decoder_hidden = (torch.cat((encoder_hidden[0].view(1, batch_size, -1),
                                 context_encoder_hidden[0].view(1, batch_size, -1)),
                                2),
                      torch.cat((encoder_hidden[1].view(1, batch_size, -1),
                                 context_encoder_hidden[0].view(1, batch_size, -1)),
                                2)
                      )
    '''
    loss = 0.0
    decoder_input = Variable(torch.LongTensor(batch_size, 1).fill_(start_token))
    length_tensor = torch.LongTensor(trainInfo.target_lengths)
    # at the begining, no sequence has ended so we use -1 as its end index
    eos_tensor = torch.LongTensor(batch_size).fill_(-1)
    if use_cuda:
        decoder_input = decoder_input.cuda()
        length_tensor = length_tensor.cuda()
        eos_tensor = eos_tensor.cuda()

    for i in range(target_len):
        decoder_output, decoder_hidden = decoder(decoder_input,
                                                 decoder_hidden,
                                                 encoder_outputs,
                                                 context_encoder_outputs,
                                                 trainInfo.context_variable)
        cur_loss = criterion(decoder_output, trainInfo.target_variable[:, i])
        loss_mask = (length_tensor > i) & (eos_tensor == -1)
        loss_mask = Variable(loss_mask).cuda() if use_cuda else Variable(loss_mask)
        loss += torch.masked_select(cur_loss, loss_mask).sum()
        _, topi = decoder_output.data.topk(1, dim=1)
        next_in = batch.unk_batch(topi)
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
        torch.nn.utils.clip_grad_norm(context_encoder.parameters(), clip)
        encoder_optimizer.step()
        context_encoder_optimizer.step()
        decoder_optimizer.step()

    return loss.item()/batch_size

def train_step(trainInfo, batch, encoder, context_encoder, decoder, encoder_optimizer, context_encoder_optimizer, decoder_optimizer, criterion):
    return step(trainInfo, batch, encoder, context_encoder, decoder,
                encoder_optimizer, context_encoder_optimizer, decoder_optimizer,
                criterion, True)

def eval_step(trainInfo, batch, encoder, context_encoder, decoder, criterion):
    return step(trainInfo, batch, encoder, context_encoder, decoder, None, None, None,
                criterion, False)

def generate_step(trainInfo, batch, encoder, context_encoder, decoder, max_length = 30):
    encoder.eval()
    context_encoder.eval()
    decoder.eval()

    batch_size = trainInfo.input_variable.size(0)
    encoder_hidden = encoder.initHidden(batch_size)
    context_encoder_hidden = context_encoder.initHidden(batch_size)
    if use_cuda:
        encoder_hidden = (encoder_hidden[0].cuda(), encoder_hidden[1].cuda())
        context_encoder_hidden = (context_encoder_hidden[0].cuda(),
                                  context_encoder_hidden[1].cuda())

    encoder_outputs, encoder_hidden = encoder(trainInfo.input_variable, trainInfo.input_lengths, encoder_hidden)
    context_variable = batch.unk_batch(trainInfo.context_variable)
    context_encoder_outputs, context_encoder_hidden = context_encoder(context_variable, trainInfo.context_lengths, trainInfo.context_sort_index, trainInfo.context_inv_index, context_encoder_hidden)
    '''
    decoder_hidden = (encoder_hidden[0].view(1, batch_size, -1),
                      encoder_hidden[1].view(1, batch_size, -1))
    '''
    decoder_hidden = (encoder_hidden[0].view(1, batch_size, -1) + context_encoder_hidden[0].view(1, batch_size, -1),
                      encoder_hidden[1].view(1, batch_size, -1) + context_encoder_hidden[1].view(1, batch_size, -1))
    '''
    decoder_hidden = (torch.cat((encoder_hidden[0].view(1, batch_size, -1),
                                 context_encoder_hidden[0].view(1, batch_size, -1)),
                                2),
                      torch.cat((encoder_hidden[1].view(1, batch_size, -1),
                                 context_encoder_hidden[0].view(1, batch_size, -1)),
                                2)
                      )
    '''
    '''
    decoder_in = Variable(torch.LongTensor(batch_size, 1).fill_(start_token))
    # -1 for not taken
    decoded_tokens = torch.LongTensor(batch_size, max_length).fill_(-1)
    eos_tensor = torch.LongTensor(batch_size).fill_(-1)
    if use_cuda:
        decoder_in = decoder_in.cuda()
        decoded_tokens = decoded_tokens.cuda()
        eos_tensor = eos_tensor.cuda()
    '''
    beam_size = 1
    beams = [ Beam(beam_size,
                   start_token,
                   start_token,
                   end_token,
                   cuda=use_cuda)
              for _ in range(batch_size)
            ]
    decoder_hiddens = [decoder_hidden for _ in range(beam_size)]
    for i in range(max_length):
        if all([b.done() for b in beams]):
            break
        decoder_in = torch.cat([b.get_current_state() for b in beams], 0)\
                          .view(batch_size, -1)\
                          .transpose(0, 1)\
                          .unsqueeze(2)
        decoder_in = Variable(batch.unk_batch(decoder_in))
        word_probs = []
        for j in range(beam_size):
            decoder_out, decoder_hidden = decoder(decoder_in[j],
                                                  decoder_hiddens[j],
                                                  encoder_outputs,
                                                  context_encoder_outputs,
                                                  trainInfo.context_variable)
            decoder_hiddens[j] = decoder_hidden
            word_probs.append(decoder_out)
        word_probs = torch.cat(word_probs, 0).data
        for j, b in enumerate(beams):
            b.advance(word_probs[j:beam_size*batch_size:batch_size, :])
    decoded_tokens = []
    for b in beams:
        _, ks = b.sort_finished(minimum=b.n_best)
        hyps = []
        for i, (times, k) in enumerate(ks[:b.n_best]):
            hyp = b.get_hyp(times, k)
            hyps.append(hyp)
        decoded_tokens.append(hyps[0])
    return decoded_tokens

def train(data, batch, encoder, context_encoder, decoder, encoder_optimizer, context_encoder_optimizer, decoder_optimizer, criterion, epoch=0, checkpoint_base=0, best_accuracy=0):
    epoch_loss = 0.0
    start = time.time()
    for i, trainInfo in enumerate(data):
        if i < checkpoint_base * 1000:
            continue
        epoch_loss += train_step(trainInfo, batch, encoder, context_encoder, decoder, encoder_optimizer, context_encoder_optimizer, decoder_optimizer, criterion)
        if (i+1) % 1000 == 0:
            checkpoint_num = (i+1) // 1000 + checkpoint_base
            writer.add_scalar("avg loss", epoch_loss/(i+1), epoch * len(data)//1000 + checkpoint_num)
            print("checkpoint{} avg loss: {:.4f}".format((i+1)//1000, epoch_loss/(i+1)))
            print("time since start: {}".format(timeSince(start)))
            checkpoint_file = arg.checkpoint_dir + "checkpoint.pth"
            torch.save({"epoch": epoch,
                        "checkpoint": checkpoint_num,
                        "encoder_state": encoder.state_dict(),
                        "context_encoder_state": context_encoder.state_dict(),
                        "decoder_state": decoder.state_dict(),
                        "encoder_optimizer_state": encoder_optimizer.state_dict(),
                        "context_encoder_optimizer_state": context_encoder_optimizer.state_dict(),
                        "decoder_optimizer_state": decoder_optimizer.state_dict(),
                        "best_acc": best_accuracy
                    }, checkpoint_file)
    epoch_loss /= len(data)
    print("epoch total training time:{}".format(timeSince(start)))
    return epoch_loss

def eval(data, batch_object, encoder, context_encoder, decoder, criterion, is_test=False):
    data = batch_object.batchify(data)
    num_correct = 0
    num_structural_correct = 0
    loss = 0
    data_len = len(data)
    batch_size = batch_object.batch_size
    for batch in data:
        trainInfo = batch_object.variableFromBatch(batch)
        decoded_tokens = generate_step(trainInfo, batch_object, encoder, context_encoder, decoder)
        batch = sorted(batch, key=lambda p: len(batch_object.indexFromName(p[0])), reverse=True)
        for i in range(batch_size):
            _, sig, _ = batch[i]
            predict_sig = tokensToString([x.item() for x in decoded_tokens[i]], batch_object.target_vocab, trainInfo.idx_oov_dict)
            if predict_sig == process_sig(sig):
                #print("correct prediction: {}".format(predict_sig))
                num_correct += 1
            try:
                predict_tree = Tree.from_str(predict_sig)
                actual_tree = Tree.from_str(process_sig(sig))
                if predict_tree.structural_eq(actual_tree):
                    num_structural_correct += 1
            except ValueError:
                pass
        if not is_test:
            loss += eval_step(trainInfo, batch_object, encoder, context_encoder, decoder, criterion)
    print("number of correct predictions: {}".format(num_correct))
    accuracy = float(num_correct)/(data_len * batch_size)
    structural_acc = num_structural_correct/(data_len * batch_size)
    if is_test:
        return accuracy, structural_acc
    return loss/data_len, accuracy, structural_acc

def eval_test(data, batch, encoder, context_encoder, decoder):
    return eval(data, batch, encoder, context_encoder, decoder, None, is_test=True)

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
def randomEval(data, batch, encoder, context_encoder, decoder):
    datum = random.choice(data)
    trainInfo = batch.variableFromBatch([datum])
    decoded_tokens = generate_step(trainInfo, batch, encoder, context_encoder, decoder)
    decoded_token = [x.item() for x in decoded_tokens[0]]
    predict_sig = tokensToString(decoded_token, batch.target_vocab, trainInfo.idx_oov_dict)
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

    batch_object = Batch(arg.batch_size, input_lang, output_lang, use_context=use_context)

    train_data = list(map(lambda p: batch_object.variableFromBatch(p), batch_object.batchify(train_data)))

    encoder = Encoder(input_lang.n_word, arg.embed_size, arg.hidden_size)
    context_encoder = ContextEncoder(output_lang.n_word, arg.embed_size, arg.hidden_size)
    decoder = ContextAttnDecoder(output_lang.n_word, arg.embed_size, arg.hidden_size)
    if use_cuda:
        encoder = encoder.cuda()
        context_encoder = context_encoder.cuda()
        decoder = decoder.cuda()
    learning_rate = 3e-4
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    context_optimizer = optim.Adam(context_encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

    encoder_optimizer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, patience=3, verbose=True, factor=0.5)
    context_optimizer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(context_optimizer, patience=3, verbose=True, factor=0.5)
    decoder_optimizer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, patience=3, verbose=True, factor=0.5)

    criterion = nn.NLLLoss(reduce=False)

    best_accuracy = 0
    best_loss = float('inf')
    best_model = (encoder.state_dict(), context_encoder.state_dict(), decoder.state_dict())
    epoch_start = 0
    checkpoint_num = 0
    print("Start training...")
    if arg.resume is not None:
        print("loading from {}".format(arg.resume))
        checkpoint = torch.load(arg.checkpoint_dir + arg.resume)
        encoder.load_state_dict(checkpoint["encoder_state"])
        context_encoder.load_state_dict(checkpoint["context_encoder_state"])
        decoder.load_state_dict(checkpoint["decoder_state"])
        encoder_optimizer.load_state_dict(checkpoint["encoder_optimizer_state"])
        context_optimizer.load_state_dict(checkpoint["context_encoder_optimizer_state"])
        decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer_state"])
        best_accuracy = checkpoint["best_acc"]
        epoch_start = checkpoint["epoch"]
        checkpoint_num = checkpoint["checkpoint"]

    for epoch in range(epoch_start, arg.num_epoch):
        try:
            print("epoch {}/{}".format(epoch+1, arg.num_epoch))
            epoch_loss = train(train_data, batch_object, encoder, context_encoder, decoder, encoder_optimizer, context_optimizer, decoder_optimizer, criterion, epoch=epoch, checkpoint_base=checkpoint_num, best_accuracy=best_accuracy)
            writer.add_scalar("Train loss", epoch_loss, epoch+1)
            print("train loss: {:.4f}".format(epoch_loss))

            dev_loss, accuracy, structural_acc = eval(dev_data, batch_object, encoder, context_encoder, decoder, criterion)
            writer.add_scalar("Dev loss", dev_loss, epoch+1)
            writer.add_scalar("Dev acc", accuracy, epoch+1)
            writer.add_scalar("Dev structural acc", structural_acc, epoch+1)
            print("dev loss: {:.4f} accuracy: {:.4f} structural accuracy: {:.4f}".format(dev_loss, accuracy, structural_acc))

            encoder_optimizer_scheduler.step(dev_loss)
            context_optimizer_scheduler.step(dev_loss)
            decoder_optimizer_scheduler.step(dev_loss)

            randomEval(dev_data, batch_object, encoder, context_encoder, decoder)

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = (encoder.state_dict(), context_encoder.state_dict(),  decoder.state_dict())
                torch.save(best_model[0], arg.encoder_state_file)
                torch.save(best_model[1], arg.context_encoder_state_file)
                torch.save(best_model[2], arg.decoder_state_file)

        except KeyboardInterrupt:
          print("Keyboard Interruption.")
          break
    print("best accuracy: {:.4f}".format(best_accuracy))
    print("Start testing...")
    encoder.load_state_dict(torch.load(arg.encoder_state_file))
    context_encoder.load_state_dict(torch.load(arg.context_encoder_state_file))
    decoder.load_state_dict(torch.load(arg.decoder_state_file))
    test_accuracy, test_structural_acc = eval_test(test_data, batch_object, encoder, context_encoder, decoder)
    writer.add_scalar("Test acc", test_accuracy)
    writer.add_scalar("Test structural acc", test_structural_acc)
    print("test accuracy: {:.4f} test structural accuracy: {:.4f}".format(test_accuracy, test_structural_acc))

if __name__ == "__main__":
    arg = parser.parse_args()
    main(arg)
