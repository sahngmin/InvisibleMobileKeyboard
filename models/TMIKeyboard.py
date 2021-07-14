import torch
import torch.nn as nn
from data import chars
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertTokenizer, BertModel, BertConfig, BertForMaskedLM
from transformers.modeling_bert import BertOnlyMLMHead
from utils import load_model, evaluate, save_model
from options import TrainOptions
import utils
from torch.nn import init
import time
from test import test_
import copy
from .BiRNN import BidirectionalRNN
from .BERT import BERT
from torch.nn import DataParallel


class TMIKeyboard(nn.Module):
    def __init__(self, args=None):
        super(TMIKeyboard, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.load_gru = args.load_gru
        self.intermediate = args.intermediate_loss
        self.args = args
        # nhid = 256
        # nlayer = 6
        self.bigru = BidirectionalRNN(char_embed_size=128, nhid=args.nhid, nlayer=args.nlayers, vocab_size=len(chars), rnn_type="GRU",
                                         semantic_decoding=False, statistic_decoding=True)
        if args.load_gru:
            self.bigru = load_model(self.bigru, args.geometric_decoder_path)

        self.bert_config = BertConfig(args=args)
        self.bert = BERT(args=args)
        self.threshold = args.cm_threshold
        if args.load_bert:
            self.bert = load_model(self.bert, args.semantic_decoder_path)

    def forward(self, x_batch, input_len):
        output_statistic = self.bigru(x_batch, input_len)
        # bert_input = torch.argmax(self.softmax(output_statistic), dim=1)
        val_k, top_k = torch.topk(nn.functional.softmax(output_statistic, dim=1), 3, dim=1)
        top1_predicted = top_k[:, 0, :]
        top1_value = val_k[:, 0, :]
        # conifidence masking
        filtered_input = top1_predicted * (top1_value > self.threshold)
        filtered_input[filtered_input == 0] = 1
        bert_input = filtered_input
        output, intermediate_output = self.bert(bert_input, input_len, masked_LM=True)

        return output_statistic, output, intermediate_output


def train_(predictor, dataloader_train, criterion, args, device, dataloader_val, save_path):
    max_acc, count, best_model = 0, 0, copy.deepcopy(predictor.state_dict())

    optimizer = optim.Adam(predictor.bert.parameters(), lr=args.lr)
    optimizer2 = optim.SGD(predictor.bigru.parameters(), lr= args.gru_lr_down * args.lr)

    scheduler1 = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.90)
    scheduler2 = optim.lr_scheduler.StepLR(optimizer2, step_size=1, gamma=0.90)

    if args.multi_gpu:
        predictor = DataParallel(predictor)

    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        epoch_loss = 0
        intermediate_loss = 0
        step = 0
        loss_interval = 150
        predictor.train()
        for x_batch, y_batch, input_len, _, _ in dataloader_train:
            step += 1
            optimizer.zero_grad()
            optimizer2.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            output_statistic, output, all_hidden_states = predictor(x_batch, input_len)
            loss = criterion(output, y_batch)
            # loss += criterion(output_statistic, y_batch)
            if predictor.intermediate:
                for k in range(predictor.bert_config.num_hidden_layers):
                    coeff = (k + 1) / (2 * predictor.bert_config.num_hidden_layers)
                    loss += coeff * criterion(all_hidden_states[k], y_batch)

            epoch_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 0.5)

            optimizer.step()
            optimizer2.step()

            if step % loss_interval == 0:
                print_loss = (epoch_loss - intermediate_loss) / loss_interval
                print('| epoch {:3d} | lr {:02.6f} | '
                      'loss {:4.4f} '.format(epoch + 1, scheduler1.get_last_lr()[0], print_loss))
                intermediate_loss = epoch_loss

        scheduler1.step()
        scheduler2.step()

        epoch_loss = epoch_loss / step

        val_loss, val_acc, _ = test_(args, predictor, None, dataloader_val)
        print('epoch_loss: {:6.4f}'.format(epoch_loss))
        print('val_acc: {:2.2f}%, val_loss: {:5.4f}'.format(val_acc, val_loss))

        if epoch == args.intermediate_stop:
            predictor.intermediate = False

        if val_acc < max_acc:
            count += 1
            if count > 2:
                if args.save:
                    print("=> saving checkpoints '{}'".format(save_path))
                    save_model(best_model, save_path)
                return best_model
        else:
            max_acc = val_acc
            if args.multi_gpu:
                best_model = copy.deepcopy(predictor.module.state_dict())
            else:
                best_model = copy.deepcopy(predictor.state_dict())
            count = 0
    # predictor = load_model(predictor, '/home/smyoo/TMI_keyboard/TMIKeyboard400_12_1.pth')
    # best_model = copy.deepcopy(predictor.state_dict())

    return best_model
