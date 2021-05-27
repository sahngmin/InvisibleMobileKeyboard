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


class Ikeyboard(nn.Module):
    def __init__(self, args=None):
        super(Ikeyboard, self).__init__()
        nhid = args.nhid
        nlayer = args.nlayers
        char_embed_size = 16
        vocab_size = len(chars)
        self.char_embed = nn.Embedding(vocab_size, char_embed_size)
        input_size_statistic = 2    # x and y
        self.bigru_statistic = nn.GRU(input_size=input_size_statistic, hidden_size=nhid, num_layers=nlayer, batch_first=True, bidirectional=True)
        input_size_semantic = char_embed_size
        self.bigru_semantic = nn.GRU(input_size=input_size_semantic, hidden_size=nhid, num_layers=nlayer, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(nhid*2, vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x_batch, input_len):
        x_packed = pack_padded_sequence(x_batch[:, :, 1:3], input_len, batch_first=True, enforce_sorted=False)
        x = x_packed.float()
        out_statistic, _ = self.bigru_statistic(x)
        out_statistic_padded, _ = pad_packed_sequence(out_statistic, batch_first=True)
        output_statistic = self.linear(out_statistic_padded)
        statistic_preds = torch.argmax(self.softmax(output_statistic), dim=2)
        embedded_input = self.char_embed(statistic_preds)
        emb_packed = pack_padded_sequence(embedded_input, input_len, batch_first=True, enforce_sorted=False)
        emb_packed = emb_packed.float()
        out_semantic, _ = self.bigru_semantic(emb_packed)
        out_semantic_padded, _ = pad_packed_sequence(out_semantic, batch_first=True)
        output_semantic = self.linear(out_semantic_padded)
        semantic_preds = torch.argmax(self.softmax(output_semantic), dim=2)
        return output_statistic, output_semantic, semantic_preds

def train_(predictor, dataloader_train, criterion, args, device, dataloader_val, save_path):
    counter, min_val_loss = [], 10000
    lr = 0.001
    optimizer = optim.Adam(predictor.parameters(), lr)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=0.90)
    num_epoch, num_epoch_decay = 500, 500
    max_acc, count, best_model = 0, 0, copy.deepcopy(predictor.state_dict())
    for epoch in range(3):
        epoch_loss = 0
        intermediate_loss = 0
        step = 0
        loss_interval = 50
        predictor.train()
        for x_batch, y_batch, input_len, _, _ in dataloader_train:
            step += 1
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            output_semantic = predictor(x_batch, input_len)[1]
            loss = criterion(output_semantic.permute(0, 2, 1), y_batch)
            output_statistic = predictor(x_batch, input_len)[0]
            loss += criterion(output_statistic.permute(0, 2, 1), y_batch)
            epoch_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 0.5)
            optimizer.step()
            if step % loss_interval == 0:
                print_loss = (epoch_loss - intermediate_loss) / loss_interval
                print('| epoch {:3d} | '
                      'loss {:6.4f} '.format(epoch + 1, print_loss))
                intermediate_loss = epoch_loss
        # scheduler.step()

        epoch_loss = epoch_loss / step

        val_loss, val_acc, _ = test_(args, predictor, None, dataloader_val)

        print('epoch_loss: {:6.4f}'.format(epoch_loss))
        print('val_acc: {:2.2f}%, val_loss: {:5.4f}'.format(val_acc, val_loss))
        best_model = copy.deepcopy(predictor.state_dict())

    return best_model