import torch
import argparse

from model import Model
from prepare_data import prepareDataWithFileName
from train import eval_test

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--train_data", metavar="TRAIN DATA",
                    default="data/new_data/train_simple_sigs_parsable_normalized.txt"
                    )

parser.add_argument("--dev_data", metavar="DEV DATA",
                    default="data/new_data/dev_simple_sigs_parsable_normalized.txt"
                    )

parser.add_argument("--test_data", metavar="TEST DATA",
                    default="data/new_data/test_simple_sigs_parsable_normalized.txt"
                    )
parser.add_argument("--topo_loss_factor", default=1, type=float)
parser.add_argument("--dropout", default=0, type=float)
parser.add_argument("--rec_depth", default=6, type=int)
parser.add_argument("--embed_size", default=128, type=int)
parser.add_argument("--hidden_size", default=256, type=int)
parser.add_argument("--model_state_file", default="model_state.pth")
parser.add_argument("--out_file", default="results/structured_test_output")

def main(arg):
    input_lang, output_lang, train_data = prepareDataWithFileName(arg.train_data, use_context = True, shuffle=False)
    _, _, test_data = prepareDataWithFileName(arg.test_data, use_context=True, shuffle=False)

    model = Model(input_lang.n_word, output_lang.n_word, arg.embed_size,
                  arg.hidden_size, output_lang.kind_dict, dropout_p=arg.dropout,
                  topo_loss_factor=arg.topo_loss_factor, rec_depth=arg.rec_depth,
                  weight=None)
    model = model.to(device)
    model.load_state_dict(torch.load(arg.model_state_file))
    print("load model")
    accuracy, structural_acc = eval_test(test_data, input_lang, output_lang, model,
                                         out_file=arg.out_file, dict_out=arg.out_file + "_dict")   
    print("test accuracy: {:.4f} structural acc: {:.4f}".format(accuracy, structural_acc))
    
if __name__ == "__main__":
    arg = parser.parse_args()
    main(arg)
