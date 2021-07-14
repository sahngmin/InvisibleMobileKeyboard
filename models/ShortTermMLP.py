import torch
import torch.nn as nn
from data import chars
import torch.optim as optim
from utils import load_model, evaluate, save_model
import time
import copy


class ShortTermDecoder(nn.Module):
    """
    Short-term decoder which decodes user inputs
    whose lengths are shorter than 9.
    Short-term decoder is based on a MLP network.
        Input : (x, y) positions of three characters
        Output : character index of each position
    """
    def __init__(self, vocab_size=len(chars), interval=True, length=3, hidden=[256, 256, 256]):
        super(ShortTermDecoder, self).__init__()
        if interval:
            input_size = 2 * (2 * length - 1) + 2
        else:
            input_size = 2 * length + 2

        mlp_modules = [nn.Linear(input_size, hidden[0]), nn.ReLU(), nn.BatchNorm1d(hidden[0])]

        for i in range(len(hidden) - 1):
            mlp_modules += [nn.Linear(hidden[i], hidden[i + 1])]
            mlp_modules += [nn.ReLU()]
            mlp_modules += [nn.BatchNorm1d(hidden[i + 1])]

        mlp_modules += [nn.Linear(hidden[-1], vocab_size * 3)]

        self.MLP = nn.Sequential(*mlp_modules)

        for m in self.MLP:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def params(self, lr):
        pass

    def forward(self, input_tensor, input_len=None):
        input_tensor = input_tensor.float()
        output = self.MLP(input_tensor)
        output = output.reshape(input_tensor.shape[0], -1, 3)

        return output


def train_(predictor, dataloader_train, criterion, args, device, dataloader_val, save_path):
    max_acc, count, best_model = 0, 0, copy.deepcopy(predictor.state_dict())

    optimizer = optim.Adam(predictor.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_lr, gamma=0.90)
    #

    for epoch in range(args.epoch):
        epoch_start_time = time.time()
        epoch_loss = 0
        intermediate_loss = 0
        step = 0
        loss_interval = 150
        predictor.train()
        for x_batch, y_batch in dataloader_train:
            step += 1
            optimizer.zero_grad()
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            output = predictor(x_batch)
            loss = criterion(output, y_batch)
            epoch_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), 0.5)
            optimizer.step()

            if step % loss_interval == 0:
                print_loss = (epoch_loss - intermediate_loss) / loss_interval
                print('| epoch {:3d} | lr {:02.6f} | '
                      'loss {:4.4f} '.format(epoch + 1, scheduler.get_lr()[0], print_loss))
                intermediate_loss = epoch_loss

        scheduler.step()

        epoch_loss = epoch_loss / step

        val_loss, val_acc = test_(args, predictor, None, dataloader_val)
        print('epoch_loss: {:6.4f}'.format(epoch_loss))
        print('val_acc: {:2.2f}%, val_loss: {:5.4f}'.format(val_acc, val_loss))

        if val_acc < max_acc:
            count += 1
            if count > 4:
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
    #     best_model = copy.deepcopy(predictor.state_dict())

    return best_model


def test_(args, predictor, best_model, dataloader_test, load_path=None, print_count=False):
    torch.manual_seed(args.test_seed)  # for reproducibility
    torch.backends.cudnn.deterministic = True  # for reproducibility
    torch.backends.cudnn.benchmark = False

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if best_model is not None:
        predictor.load_state_dict(best_model)
    if load_path is not None:
        predictor = load_model(predictor, load_path)

    predictor.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    total = 0.0
    correct = 0
    total_loss = 0.
    accuracy = 0
    step = 0

    with torch.no_grad():
        for x_batch, y_batch in dataloader_test:
            step += 1
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            output = predictor(x_batch)

            # output = output.permute(0, 2, 1)

            _, top_k = torch.topk(output, 3, dim=1)
            top1_predicted = top_k[:, 0, :]

            loss = criterion(output, y_batch)

            total += 3 * 16

            correct += (top1_predicted == y_batch).sum()
            total_loss += loss

        accuracy = 100 * correct / total
        avg_loss = total_loss / step
        pred_idx = top1_predicted[0]
        label = y_batch[0]

    return avg_loss, accuracy
