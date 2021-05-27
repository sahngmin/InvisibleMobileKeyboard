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
from torch.nn import DataParallel
import time
from test import test_
import copy


class BERT(nn.Module):
    def __init__(self, args=None):
        super(BERT, self).__init__()

        self.intermediate = args.intermediate_loss
        self.args = args
        self.bert_config = BertConfig(args=args)
        self.bert = BertModel(self.bert_config)
        self.cls = BertOnlyMLMHead(self.bert_config)

    def forward(self, x_batch, input_len, masked_LM=False):
        if masked_LM:
            output, all_hidden_states = self.bert(x_batch)
        else:
            output, all_hidden_states = self.bert(x_batch[:, :, 1:])
        output = output[0]
        prediction = self.cls(output)
        if self.args.excessive_output:
            prediction = prediction[:, :, :self.bert_config.vocab_size]

        permuted_intermediate_pred = []
        for j in range(len(all_hidden_states)):
            hidden_state = all_hidden_states[j]
            if self.args.excessive_output:
                intermediate_prediction = self.cls(hidden_state)[:, :, :self.bert_config.vocab_size].permute(0, 2, 1)
            else:
                intermediate_prediction = self.cls(hidden_state).permute(0, 2, 1)
            permuted_intermediate_pred = permuted_intermediate_pred + [intermediate_prediction]
        return prediction.permute(0, 2, 1), permuted_intermediate_pred


def train_(predictor, dataloader_train, criterion, args, device, dataloader_val, save_path):
    min_val_loss, max_acc, count, best_model = 100, 0, 0, copy.deepcopy(predictor.state_dict())
    optimizer = optim.Adam(predictor.parameters(), lr=args.lr)
    # optimizer = optim.RMSprop(self.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=0.95)

    if args.multi_gpu:
        predictor = DataParallel(predictor)

    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        epoch_loss = 0
        intermediate_loss = 0
        step = 0
        loss_interval = 150
        predictor.train()
        for x_batch, y_batch, input_len, _, full_labels in dataloader_train:
            step += 1
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            full_labels = full_labels.to(device)
            output, all_hidden_states = predictor(x_batch, input_len, masked_LM=args.masked_LM)
            loss = criterion(output, y_batch)
            loss_full = criterion(output, full_labels)
            loss += loss_full
            if predictor.intermediate:
                for k in range(predictor.bert_config.num_hidden_layers):
                    coeff = (k + 1) / (2 * predictor.bert_config.num_hidden_layers)
                    loss += coeff * criterion(all_hidden_states[k], y_batch)
            epoch_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 0.5)
            optimizer.step()

            if step % loss_interval == 0:
                print_loss = (epoch_loss - intermediate_loss) / loss_interval
                print('| epoch {:3d} | lr {:02.6f} | '
                      'loss {:4.4f} '.format(epoch + 1, scheduler.get_last_lr()[0], print_loss))
                intermediate_loss = epoch_loss
        scheduler.step()

        epoch_loss = epoch_loss / step

        val_loss, val_acc, w_error = test_(args, predictor, None, dataloader_val)
        print('epoch_loss: {:6.4f}'.format(epoch_loss))
        print('val_acc: {:2.2f}%, val_loss: {:5.4f}'.format(val_acc, val_loss))

        if epoch == args.intermediate_stop:
            predictor.intermediate = False

        if args.train_data == './data/1BW_english.txt':
            print("=> saving checkpoints '{}'".format(save_path.replace('.pth', str(epoch + 1) + '.pth')))
            save_model(best_model, save_path)

        if val_acc < max_acc:
            count += 1
            if count > 5:
                if args.save:
                    print("=> saving checkpoints '{}'".format(save_path.replace('.pth', str(epoch + 1) + '.pth')))
                    save_model(best_model, save_path.replace('.pth', 'epoch' + str(epoch + 1) + '.pth'))
                return best_model
        else:
            max_acc = val_acc
            min_val_loss = val_loss
            if args.multi_gpu:
                best_model = copy.deepcopy(predictor.module.state_dict())
            else:
                best_model = copy.deepcopy(predictor.state_dict())
            count = 0

    return best_model