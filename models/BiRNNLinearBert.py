import torch
import torch.nn as nn
from data import chars
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import load_model, evaluate, save_model
from torch.nn import init
import time
from test import test_
import copy
from torch.nn import DataParallel
from transformers import BertModel, BertConfig
from .BiRNN import BidirectionalRNN


class RNNLinearBERT(nn.Module):
    def __init__(self, semantic_decoding=False, statistical_decoding=True, gru_stack_bert=True, args=None):
        super(RNNLinearBERT, self).__init__()
        self.bert_config = BertConfig(args=args)
        self.bert = BertModel(self.bert_config)
        self.semantic = semantic_decoding
        self.statistic = statistical_decoding
        self.gru_stack_bert = gru_stack_bert
        self.intermediate_output = args.intermediate_output
        nhid = 512
        nlayer = 6
        if gru_stack_bert:
            self.init_rnn = BidirectionalRNN(char_embed_size=128, nhid=nhid, nlayer=nlayer, vocab_size=len(chars), rnn_type="GRU")
            self.bigru = load_model(self.init_rnn, args.geometric_decoder_path)
        self.linear = nn.Linear(nhid*2, self.bert_config.hidden_size)

    def params(self, lr):
        if self.gru_stack_bert:
            params = [{'params': self.bert.parameters()},
                      {'params': self.linear.parameters()},
                      {'params': self.bigru.parameters(), 'lr': 0.001 * lr}]
        else:
            params = [{'params': self.bert.parameters()},
                      {'params': self.linear.parameters()}]
        return params

    def forward(self, x_batch, input_len):
        # x_batch = x_batch[:, :, 1:]
        if self.gru_stack_bert:
            feat = self.bigru(x_batch, input_len, bert_stack=self.gru_stack_bert)
            feat = self.linear(feat)
            output, prev_prediction_score = self.bert(inputs_embeds=feat)
            output = output[0]

            if self.bert_config.excessive_output:
                output = output[:, :, :self.bert_config.vocab_size]
                prev_prediction_score = prev_prediction_score[:, :, :self.bert_config.vocab_size]

        else:
            output, prev_prediction_score = self.bert(x_batch[:, :, 1:])
            output = output[0]
        # needs more coding for independent bert

        if self.intermediate_output:
            return output.permute(0, 2, 1), prev_prediction_score.permute(0, 2, 1)
        else:
            return output.permute(0, 2, 1)

    def train_(self, dataloader_train, criterion, args, device, dataloader_val, save_path):
        max_acc, count, best_model = 0, 0, None
        optimizer = optim.Adam(self.params(args.lr), lr=args.lr)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=0.90)
        for epoch in range(args.epoch):
            epoch_start_time = time.time()
            epoch_loss = 0
            intermediate_loss = 0
            step = 0
            loss_interval = 50
            self.train()
            for x_batch, y_batch, input_len in dataloader_train:
                step += 1
                optimizer.zero_grad()
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                if args.intermediate_output:
                    output, prev_output = self.forward(x_batch, input_len)
                    co_eff = 0.5 - int(2 * epoch / args.epoch) / 2
                    loss = criterion(output, y_batch) + co_eff * criterion(prev_output, y_batch)
                else:
                    output = self.forward(x_batch, input_len)
                    loss = criterion(output, y_batch)
                epoch_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                optimizer.step()

                if step % loss_interval == 0:
                    print_loss = (epoch_loss - intermediate_loss) / loss_interval
                    print('| epoch {:3d} | lr {:02.6f} | '
                          'loss {:6.4f} '.format(epoch + 1, scheduler.get_lr()[0], print_loss))
                    intermediate_loss = epoch_loss
            scheduler.step()

            epoch_loss = epoch_loss / step

            val_loss, val_acc = evaluate(dataloader_val, self, criterion, args, state='val', print_result=False)
            print('epoch_loss: {:6.4f}'.format(epoch_loss))
            print('val_loss: {:6.4f}'.format(val_loss))

            if val_acc < max_acc:
                count += 1
                if count > args.early_stop:
                    if args.save:
                        print("=> saving checkpoints '{}'".format(args.save_path))
                        save_model(best_model, save_path)
                    return best_model
            else:
                max_acc = val_acc
                best_model = copy.deepcopy(self.state_dict())
                print('Best Model Updated!!')
                count = 0

        return best_model