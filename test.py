import logging
from options import TrainOptions
import random
import utils
import torch.nn as nn
from utils import load_model, idx2chars
import torch
from data import get_dataloader
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils import wer
# from models import BiRNN


def test_(args, predictor, best_model, dataloader_test, load_path=None, print_count=False):
    # random.seed(args.test_seed)
    # torch.manual_seed(args.test_seed)  # for reproducibility
    # torch.backends.cudnn.deterministic = True  # for reproducibility
    # torch.backends.cudnn.benchmark = False

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
    distance_sum = 0
    length_sum = 0

    with torch.no_grad():
        for x_batch, y_batch, input_len, masks, _ in dataloader_test:
            step += 1
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if args.TMIkeyboard:
                output_stat, output, all_hidden_states = predictor(x_batch, input_len)
            elif args.bert:
                output, prev_outputs = predictor(x_batch, input_len, masked_LM=args.masked_LM)
            else:
                output = predictor(x_batch, input_len)
            if args.ikeyboard:
                output = output[1].permute(0, 2, 1)
                _, top_k = torch.topk(output, 3, dim=1)
                top1_predicted = top_k[:, 0, :]
            else:
                _, top_k = torch.topk(output, 3, dim=1)
                top1_predicted = top_k[:, 0, :]

            loss = criterion(output, y_batch)

            if type(masks) is not list:
                masks = masks.to(device)
                total += masks.sum()

                correct_packed = pack_padded_sequence((top1_predicted == y_batch) * masks, input_len, batch_first=True,
                                                      enforce_sorted=False)
                correct += correct_packed.data.sum()
            else:
                total += sum(input_len)
                correct_packed = pack_padded_sequence(top1_predicted == y_batch, input_len, batch_first=True,
                                                      enforce_sorted=False)
                correct += correct_packed.data.sum()
            if print_count:
                correct_count = (top1_predicted == y_batch).data.sum(dim=1)
                # print(list(input_len))
                # print(correct_count.tolist() + '\n')
                for i in range(len(list(input_len))):
                    ratio = float(correct_count.tolist()[i] / list(input_len)[i])
                    if ratio < 0.5:
                        # print(ratio)
                        fig, ax = plt.subplots()
                        plt.axes().set_aspect('equal')
                        x_list = x_batch[i][:input_len[i]][:, 1].tolist()
                        y_list = x_batch[i][:input_len[i]][:, 2].tolist()
                        plt.scatter(x_list, [-y for y in y_list], s=[1, 1])
                        ax.legend()
                        label_list = list(idx2chars(y_batch[i])[:input_len[i]])
                        for j in range(len(label_list)):
                            plt.text(x_list[j], -y_list[j], label_list[j], fontsize=12, weight='bold')
                        fig.savefig('results/plot_bad/ratio_' + idx2chars(y_batch[i])[:input_len[i]] + str(ratio) + '.jpg')
                        plt.show()
                        plt.close(fig)

            for k in range(y_batch.shape[0]):
                pred_idx = top1_predicted[k][:input_len[k]]
                label = y_batch[k][:input_len[k]]
                predicted = idx2chars(pred_idx)
                original = idx2chars(label)

                distance, length = wer(original, predicted, True)
                distance_sum += distance
                length_sum += input_len[k]

            accuracy = 100 * correct / total
            total_loss += loss

        avg_loss = total_loss / step
        w_error = float(distance_sum / length_sum) * 100

    return avg_loss, accuracy, w_error


if __name__ == "__main__":
    args = TrainOptions().parse()
    args = utils.bashRun(args)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    DataLoader = get_dataloader(args.test_data, batch_size=args.batch_size, min_length=args.length,
                                various=args.various_length, full_sentence=args.full_sentence)


    predictor = BiRNN.BidirectionalRNN(char_embed_size=args.char_embed_size, nhid=256, nlayer=6,
                                           semantic_decoding=args.semantic_decoding, rnn_type='GRU').to(device)

    save_path = './checkpoints/BiRNN256_6_5.pth'

    loss, acc, w_error = test_(args, predictor, None, DataLoader, load_path=save_path)
    print(acc)
    print(w_error)
