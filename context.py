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

use_cuda = torch.cuda.is_available()
#use_cuda = False

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
'''
class TrainInfo():
    
    this class is purely used to hold data. I don't want to rewrite a function that has 20 arguments
    
    def __init__(self, input_var, input_lengths, target_var, target_lengths, context_var, context_lengths):
        self.input_variable = input_var
        self.input_lengths = input_lengths
        self.target_variable = target_var
        self.target_lengths = target_lengths
        self.context_variable = context_var
        self.context_lengths = context_lengths

        
def variableFromBatchWithContext(batch, input_lang, output_lang):
    batch_size = len(batch)
    batch = map(lambda p: (indexFromName(p[0], input_lang),
                           indexFromSignature(p[1], output_lang),
                           indexFromSignatures(p[2], output_lang)
                           ),
                batch
                )
    batch = sorted(batch, key=lambda p:len(p[0]), reverse=True)
    input_lengths = map(lambda p: len(p[0]), batch)
    output_lengths = map(lambda p: len(p[1]), batch)
    context_lengths = map(lambda p: len(p[2]), batch)
    max_input_len = len(batch[0][0])
    max_output_len = max(output_lengths)
    max_context_len = max(context_lengths)
    batch = map(lambda p: (p[0] + (max_input_len - len(p[0])) * [0],
                           p[1] + (max_output_len - len(p[1])) * [0],
                           p[2] + (max_context_len - len(p[2])) * [0]
                           ),
                batch
                )
    input_batch = torch.LongTensor(batch_size, max_input_len)
    output_batch = torch.LongTensor(batch_size, max_output_len)
    context_batch = torch.LongTensor(batch_size, max_context_len)
    for i in range(batch_size):
        input_batch[i] = torch.LongTensor(batch[i][0])
        output_batch[i] = torch.LongTensor(batch[i][1])
        context_batch[i] = torch.LongTensor(batch[i][2])
    input_variable = Variable(input_batch)
    output_variable = Variable(output_batch)
    context_variable = Variable(context_batch)
    if use_cuda:
        input_variable = input_variable.cuda()
        output_variable = output_variable.cuda()
        context_variable = context_variable.cuda()
    return TrainInfo(input_variable, input_lengths, output_variable, output_lengths, context_variable, context_lengths)
'''

def step(trainInfo, batch, encoder, context_encoder, decoder, encoder_optimizer, context_encoder_optimizer, decoder_optimizer, criterion, is_train):
    if is_train:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
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
    context_variable = batch.unk_batch(trainInfo.context_variable) # for feeding to context encoder
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
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs, context_encoder_outputs, trainInfo.context_variable)
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

    #print(loss.data[0]/batch_size)
    return loss.data[0]/batch_size

def train_step(trainInfo, batch, encoder, context_encoder, decoder, encoder_optimizer, context_encoder_optimizer, decoder_optimizer, criterion):
    return step(trainInfo, batch, encoder, context_encoder, decoder,
                encoder_optimizer, context_encoder_optimizer, decoder_optimizer,
                criterion, True)

def eval_step(trainInfo, batch, encoder, context_encoder, decoder, criterion):
    return step(trainInfo, batch, encoder, context_encoder, decoder, None, None, None,
                criterion, False)

def generate_step(trainInfo, batch, encoder, context_encoder, decoder, max_length = 30):
    encoder.eval()
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
    decoder_in = Variable(torch.LongTensor(batch_size, 1).fill_(start_token))
    # -1 for not taken
    decoded_tokens = torch.LongTensor(batch_size, max_length).fill_(-1)
    eos_tensor = torch.LongTensor(batch_size).fill_(-1)
    if use_cuda:
        decoder_in = decoder_in.cuda()
        decoded_tokens = decoded_tokens.cuda()
        eos_tensor = eos_tensor.cuda()
    for i in range(max_length):
        decoder_out, decoder_hidden = decoder(decoder_in, decoder_hidden, encoder_outputs, context_encoder_outputs, trainInfo.context_variable)
        _, topi = decoder_out.data.topk(1, dim=1)
        pred = topi[:, 0]
        decoded_tokens[:, i] = pred
        decoded_tokens[:, i].masked_fill_(eos_tensor > -1, -1)
        end_mask = pred == end_token
        eos_tensor.masked_fill_(end_mask & (eos_tensor == -1), i+1)
        decoder_in = Variable(batch.unk_batch(topi))
    return decoded_tokens, eos_tensor

def train(data, batch, encoder, context_encoder, decoder, encoder_optimizer, context_encoder_optimizer, decoder_optimizer, criterion):
    epoch_loss = 0.0
    start = time.time()
    for i, trainInfo in enumerate(data):
        epoch_loss += train_step(trainInfo, batch, encoder, context_encoder, decoder, encoder_optimizer, context_encoder_optimizer, decoder_optimizer, criterion)
        if (i+1) % 1000 == 0:
            print("checkpoint{} avg loss: {:.4f}".format((i+1)/1000, epoch_loss/(i+1)))
            print("time since start: {}".format(timeSince(start)))
    epoch_loss /= len(data)
    print("epoch total training time:{}".format(timeSince(start)))
    return epoch_loss

def eval(data, batch_object, encoder, context_encoder, decoder, criterion, is_test=False):
    data = batch_object.batchify(data)
    num_correct = 0
    loss = 0
    data_len = len(data)
    batch_size = batch_object.batch_size
    for batch in data:
        trainInfo = batch_object.variableFromBatch(batch)
        decoded_tokens, eos_tensor = generate_step(trainInfo, batch_object, encoder, context_encoder, decoder)
        batch = sorted(batch, key=lambda p: len(batch_object.indexFromName(p[0])), reverse=True)
        for i in range(batch_size):
            _, sig, _ = batch[i]
            predict_sig = tokensToString(decoded_tokens[i, :eos_tensor[i]].tolist(), batch_object.target_vocab, trainInfo.idx_oov_dict)
            if predict_sig == process_sig(sig):
                num_correct += 1
        if not is_test:
            loss += eval_step(trainInfo, batch_object, encoder, context_encoder, decoder, criterion)
    accuracy = float(num_correct)/(data_len * batch_size)
    if is_test:
        return accuracy
    return loss/data_len, accuracy

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
    decoded_tokens, eos_tensor = generate_step(trainInfo, batch, encoder, context_encoder, decoder)
    decoded_tokens = decoded_tokens.squeeze(0)[:eos_tensor[0]].tolist()
    predict_sig = tokensToString(decoded_tokens, batch.target_vocab, trainInfo.idx_oov_dict)
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

    
    output_lang.trim_tokens(threshold=2)
    print("Input vocab size: {}".format(input_lang.n_word))
    print("Target vocab size: {}".format(output_lang.n_word))

    batch_object = Batch(arg.batch_size, input_lang, output_lang, use_context=use_context)

    train_data = map(lambda p: batch_object.variableFromBatch(p), batch_object.batchify(train_data))

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

    encoder_optimizer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, patience=1, verbose=True, factor=0.5)
    context_optimizer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(context_optimizer, patience=1, verbose=True, factor=0.5)
    decoder_optimizer_scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, patience=1, verbose=True, factor=0.5)

    criterion = nn.NLLLoss(reduce=False)
    
    best_accuracy = 0
    best_loss = float('inf')
    best_model = (encoder.state_dict(), context_encoder.state_dict(), decoder.state_dict())
    print("Start training...")
    for epoch in range(arg.num_epoch):
        try:
            epoch_loss = 0
            print("epoch {}/{}".format(epoch+1, arg.num_epoch))
            epoch_loss = train(train_data, batch_object, encoder, context_encoder, decoder, encoder_optimizer, context_optimizer, decoder_optimizer, criterion)
            print("train loss: {:.4f}".format(epoch_loss))
            dev_loss, accuracy = eval(dev_data, batch_object, encoder, context_encoder, decoder, criterion)
            print("dev loss: {:.4f} accuracy: {:.4f}".format(dev_loss, accuracy))

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
    test_accuracy = eval_test(test_data, batch_object, encoder, context_encoder, decoder)
    print("test accuracy: {:.4f}".format(test_accuracy))

if __name__ == "__main__":
    arg = parser.parse_args()
    main(arg)
