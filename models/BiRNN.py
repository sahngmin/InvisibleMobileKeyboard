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


class BidirectionalRNN(nn.Module):
    """Bi-RNN based statistical/semantic decoder.
    Decoder receives the raw inputs from users,
    and transforms the inputs to sequences of characters.
    """
    def __init__(self, char_embed_size=16, nhid=128, nlayer=2, vocab_size=len(chars), rnn_type="GRU", layer_norm=False):
        super(BidirectionalRNN, self).__init__()
        self.normalize = layer_norm
        if layer_norm:
            self.dense = nn.Linear(4, 4, bias=False)
            self.layerNorm = nn.LayerNorm(4)
        self.char_embed = nn.Embedding(vocab_size, char_embed_size)
        input_size = char_embed_size + 4    # [x, y, w, h + embedding size]
        # self.bn1 = nn.BatchNorm1d(num_features=4+char_embed_size)
        if rnn_type in ["GRU", "LSTM"]:
            self.rnn = getattr(nn, rnn_type)(
                input_size=input_size,
                hidden_size=nhid,
                num_layers=nlayer,  # since 2, it means stacking 2 GRUs together
                bidirectional=False
            )
        for layer in self.rnn._all_weights:
            for param in layer:
                if 'weight' in param:
                    init.xavier_uniform_(self.rnn.__getattr__(param))
        self.linear = nn.Linear(nhid, vocab_size)
        nn.init.xavier_uniform_(self.linear.weight)

        # self.bn1d = nn.BatchNorm1d(4)

    def forward(self, x_batch, input_len, bert_stack=False):

        # x_batch_flat = x_batch[:, :, 1:].view(-1, 4)
        # x_bn = self.bn1(x_batch_flat)
        if self.normalize:
            rnn_input = self.layerNorm(self.dense(x_batch[:, :, 1:].float()))
        else:
            rnn_input = x_batch[:, :, 1:]
        x_packed = pack_padded_sequence(rnn_input, input_len, batch_first=True, enforce_sorted=False)
        x = x_packed.float()

        out, _ = self.rnn(x)
        output_padded, _ = pad_packed_sequence(out, batch_first=True)
        # output = self.bn1(output_padded.shape[1])
        output = self.linear(output_padded)
        if bert_stack:
            return output_padded
        else:
            output = output.permute(0, 2, 1)
            return output


def train_(predictor, dataloader_train, criterion, args, device, dataloader_val, save_path):
    max_acc, count, best_model = 0, 0, copy.deepcopy(predictor.state_dict())
    args.lr = 3.0
    optimizer = optim.SGD(predictor.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=0.90)
    if args.multi_gpu:
        predictor = DataParallel(predictor)
    for epoch in range(2):
        epoch_start_time = time.time()
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
            output = predictor(x_batch, input_len)
            loss = criterion(output, y_batch)
            epoch_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 0.5)
            optimizer.step()
            if step % loss_interval == 0:
                print_loss = (epoch_loss - intermediate_loss) / loss_interval
                print('| epoch {:3d} | lr {:02.6f} | '
                      'loss {:6.4f} '.format(epoch + 1, scheduler.get_last_lr()[0], print_loss))
                intermediate_loss = epoch_loss
        scheduler.step()

        epoch_loss = epoch_loss / step

        val_loss, val_acc, w_error = test_(args, predictor, None, dataloader_val)

        print('epoch_loss: {:6.4f}'.format(epoch_loss))
        print('val_acc: {:2.2f}%, val_loss: {:5.4f}'.format(val_acc, val_loss))

        best_model = copy.deepcopy(predictor.state_dict())

        if val_acc < max_acc:
            count += 1
            if count > args.early_stop:
                if args.save:
                    print("=> saving checkpoints '{}'".format(args.save_path))
                    save_model(best_model, save_path)
                return best_model
        else:
            max_acc = val_acc
            if args.multi_gpu:
                best_model = copy.deepcopy(predictor.module.state_dict())
            else:
                best_model = copy.deepcopy(predictor.state_dict())
            print('Best Model Updated!!')
            count = 0

        if args.save:
            print("=> saving checkpoints '{}'".format(args.save_path))
            save_model(best_model, save_path)

    return best_model



