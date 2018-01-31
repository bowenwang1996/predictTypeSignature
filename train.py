import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from identifier_segmentor import segment
from prepare_data import readSigTokens, prepareData, start_token, end_token, unk_token
from model import Encoder, Decoder, AttnDecoder
import time
import math
import random

use_cuda = torch.cuda.is_available()

def asMinutes(s):
    m = math.floor(s/60)
    s -= m * 60
    return "%dm %ds" % (m, s)

def timeSince(since):
    now = time.time()
    s = now - since
    return "%s" % asMinutes(s)

def variableFromSignature(sig, lang):
    tokens = sig.split()
    def foo(x):
        if x[0].isupper():
            token = x.split('.')[-1]
        else:
            token = x
        return lang.lookup(token)
    indices = map(foo, tokens)
    #indices = map(lambda x: lang.token_to_idx[x.split('.')[-1]], tokens)
    indices.append(end_token)
    var = Variable(torch.LongTensor(indices))
    if use_cuda:
        var = var.cuda()
    return var

def variableFromName(name, lang):
    tokens = segment(name)
    indices = map(lambda x: lang.lookup(x), tokens)
    indices.append(end_token)
    var = Variable(torch.LongTensor(indices))
    if use_cuda:
        var = var.cuda()
    return var
    
def step(input_variable, output_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, is_train):
    if is_train:
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        encoder.train()
        decoder.train()
    else:
        encoder.eval()
        decoder.eval()

    encoder_hidden = encoder.initHidden()
    if use_cuda:
        encoder_hidden = (encoder_hidden[0].cuda(), encoder_hidden[1].cuda())
    input_len = input_variable.size(0)
    encoder_outputs = Variable(torch.zeros(input_len, encoder.hidden_size))
    if use_cuda:
        encoder_outputs = encoder_outputs.cuda()
    for i in range(input_len):
        encoder_output, encoder_hidden = encoder(input_variable[i], encoder_hidden)
        encoder_outputs[i] = encoder_output[0][0]

    decoder_hidden = (encoder_hidden[0].view(1, 1, -1), encoder_hidden[1].view(1, 1, -1))
    output_len = output_variable.size()[0]
    decoder_in = Variable(torch.LongTensor([start_token]))
    if use_cuda:
        decoder_in = decoder_in.cuda()
    loss = 0.0

    for i in range(output_len):
        decoder_out, decoder_hidden = decoder(decoder_in, decoder_hidden, encoder_outputs)
        #decoder_out, decoder_hidden = decoder(decoder_in, decoder_hidden)
        loss += criterion(decoder_out, output_variable[i])
        _, topi = decoder_out.data.topk(1)
        pred = topi[0][0]
        
        if pred == end_token:
            break
        decoder_in = Variable(torch.LongTensor([pred]))
        if use_cuda:
            decoder_in = decoder_in.cuda()

    if is_train:
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
    return loss.data[0]/output_len

def train_step(input_variable, output_variable, encoder, decoder,
               encoder_optimizer, decoder_optimizer, criterion):
    return step(input_variable, output_variable,
                encoder, decoder, encoder_optimizer, decoder_optimizer,
                criterion, True)

def eval_step(input_variable, output_variable, encoder, decoder, criterion):
    return step(input_variable, output_variable,
                encoder, decoder, None, None,
                criterion, False)

def generate_step(input_variable, encoder, decoder, max_length = 30):
    encoder.eval()
    decoder.eval()
    
    encoder_hidden = encoder.initHidden()
    if use_cuda:
        encoder_hidden = (encoder_hidden[0].cuda(), encoder_hidden[1].cuda())
    input_len = input_variable.size(0)
    encoder_outputs = Variable(torch.zeros(input_len, encoder.hidden_size))
    if use_cuda:
        encoder_outputs = encoder_outputs.cuda()
    for i in range(input_len):
        encoder_output, encoder_hidden = encoder(input_variable[i], encoder_hidden)
        encoder_outputs[i] = encoder_output[0][0]

    decoder_hidden = (encoder_hidden[0].view(1, 1, -1), encoder_hidden[1].view(1, 1, -1))
    decoder_in = Variable(torch.LongTensor([start_token]))
    if use_cuda:
        decoder_in = decoder_in.cuda()
    decoded_tokens = []
    for i in range(max_length):
        decoder_out, decoder_hidden = decoder(decoder_in, decoder_hidden, encoder_outputs)
        #decoder_out, decoder_hidden = decoder(decoder_in, decoder_hidden)
        _, topi = decoder_out.data.topk(1)
        pred = topi[0][0]
        decoded_tokens.append(pred)
        if pred == end_token:
            break
    return decoded_tokens


def train(data, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion):
    epoch_loss = 0.0
    start = time.time()
    for i, pair in enumerate(data):
        input_variable, output_variable = pair
        epoch_loss += train_step(input_variable, output_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        if (i+1) % 10000 == 0:
            print("checkpoint{} avg loss: {:.4f}".format((i+1)/10000, epoch_loss/(i+1)))
            print("time since start: {}".format(timeSince(start)))
    epoch_loss /= len(data)
    return epoch_loss

def eval(data, encoder, decoder, criterion, is_test=False):
    num_correct = 0
    loss = 0
    data_len = len(data)
    for input_variable, output_variable in data:
        decoded_tokens = generate_step(input_variable, encoder, decoder)
        decoded_tensor = torch.LongTensor(decoded_tokens)
        if use_cuda:
            decoded_tensor = decoded_tensor.cuda()
        if torch.equal(decoded_tensor, output_variable.data):
            num_correct += 1
        if not is_test:
            loss += eval_step(input_variable, output_variable, encoder, decoder, criterion)
    accuracy = float(num_correct)/data_len
    if is_test:
        return accuracy
    else:
        return loss.data[0]/data_len, accuracy

def eval_test(data, encoder, decoder):
    return eval(data, encoder, decoder, None, is_test=True)

def randomEval(data, encoder, decoder, input_lang, output_lang):
    name, sig = random.choice(data)
    input_variable = variableFromName(name, input_lang)
    decoded_tokens = generate_step(input_variable, encoder, decoder)
    if decoded_tokens[-1] == end_token:
        decoded_tokens = decoded_tokens[:-1]
    predict_sig = ""
    for token in decoded_tokens:
        predict_sig += output_lang.idx_to_token[token] + " "
    print(name)
    print(sig)
    print(predict_sig)

def main():
    type_token_file = "simple_types_vocab.txt"
    train_file = "train_simple_sigs_parsable_normalized.txt"
    dev_file = "dev_simple_sigs_parsable_normalized.txt"

    input_lang, train_data = prepareData(train_file)
    output_lang = readSigTokens(type_token_file)
    _, original_dev_data = prepareData(dev_file)
    train_data = map(lambda p: (variableFromName(p[0], input_lang), variableFromSignature(p[1], output_lang)), train_data)
    dev_data = map(lambda p: (variableFromName(p[0], input_lang), variableFromSignature(p[1], output_lang)), original_dev_data)

    encoder_state_file = "encoder_state.pth"
    decoder_state_file = "decoder_state.pth"

    hidden_size = 256
    encoder = Encoder(input_lang.n_word, hidden_size)
    decoder = AttnDecoder(output_lang.n_word, hidden_size)
    #decoder = Decoder(output_lang.n_word, hidden_size)
    if use_cuda:
        encoder = encoder.cuda()
        decoder = decoder.cuda()
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=5e-4)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=5e-4)
    criterion = nn.NLLLoss()
    num_epoch = 50
    
    best_accuracy = 0
    best_model = (encoder.state_dict(), decoder.state_dict())
    print("Start training...")
    for epoch in range(num_epoch):
        try:
            print("epoch {}/{}".format(epoch+1, num_epoch))
            epoch_loss = train(train_data, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
            print("train loss: {:.4f}".format(epoch_loss))
        
            dev_loss, accuracy = eval(dev_data, encoder, decoder, criterion)
            print("dev loss: {:.4f} accuracy: {:.4f}".format(dev_loss, accuracy))
            
            randomEval(original_dev_data, encoder, decoder, input_lang, output_lang)
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = (encoder.state_dict(), decoder.state_dict())
                torch.save(best_model[0], encoder_state_file)
                torch.save(best_model[1], decoder_state_file)
        except KeyboardInterrupt:
          print("Keyboard Interruption.")
          break
    print("best accuracy: {:.4f}".format(best_accuracy))
    print("Start testing...")
    test_file = "test_simple_sigs_parsable_normalized.txt"
    _, test_data = prepareData(test_file)
    test_data = map(lambda p: (variableFromName(p[0], input_lang), variableFromSignature(p[1], output_lang)), test_data)
    encoder.load_state_dict(torch.load(encoder_state_file))
    decoder.load_state_dict(torch.load(decoder_state_file))
    test_accuracy = eval_test(test_data, encoder, decoder)
    print("test accuracy: {:.4f}".format(test_accuracy))
    
if __name__ == "__main__":
    main()
    
