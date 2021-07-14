import torch
from data import get_diff
import editdistance
import re
from data import chars
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import matplotlib.pyplot as plt
from matplotlib import pylab
import numpy as np


char_to_idx = {ch: i for i, ch in enumerate(chars)}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


torch.manual_seed(0)                            # for reproducibility
torch.backends.cudnn.deterministic = True       # for reproducibility
torch.backends.cudnn.benchmark = False


def bashRun(args):
    """
    When bash running, convert string arguments to an appropriate type.
    """
    if type(args.lr) is str:
        args.lr = float(args.lr)

    if type(args.gru_lr_down) is str:
        args.gru_lr_down = float(args.gru_lr_down)

    if type(args.step_lr) is str:
        args.step_lr = float(args.step_lr)

    if type(args.seed) is str:
        args.seed = int(args.seed)

    if type(args.test_seed) is str:
        args.test_seed = int(args.test_seed)

    if type(args.epoch) is str:
        args.epoch = int(args.epoch)

    if type(args.case) is str:
        args.case = int(args.case)

    if type(args.nhid) is str:
        args.nhid = int(args.nhid)

    if type(args.nhead) is str:
        args.nhead = int(args.nhead)

    if type(args.nlayers) is str:
        args.nlayers = int(args.nlayers)

    if type(args.batch_size) is str:
        args.batch_size = int(args.batch_size)

    if str(args.excessive_output) == 'True':
        args.excessive_output = True
    else:
        args.excessive_output = False

    if str(args.intermediate_loss) == 'True':
        args.intermediate_loss = True
    else:
        args.intermediate_loss = False

    if str(args.augment) == 'True':
        args.augment = True
    else:
        args.augment = False

    if str(args.multi_gpu) == 'True':
        args.multi_gpu = True
    else:
        args.multi_gpu = False

    return args


def evaluate(dataloader, predictor, criterion, args, state='val', pretrained=None, print_result=False):
    """
    compute and print loss and accuracy
    If long_term or semantic, print (input, prediction, label) pair.
    """
    if pretrained is not None:
        predictor.load_state_dict(pretrained.state_dict())
    predictor.eval()
    total = 0.0
    correct = 0
    step = 0
    total_loss = 0.
    accuracy = 0
    semantic = True

    with torch.no_grad():
        for x_batch, y_batch, input_len in dataloader:
            step += 1
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            if args.sa_ncd:
                output_stat, output, all_hidden_states = predictor(x_batch, input_len)
            else:
                output = predictor(x_batch, input_len)
            if args.ikeyboard:
                loss = criterion(output[1].permute(0, 2, 1), y_batch)
                loss += criterion(output[0].permute(0, 2, 1), y_batch)
                _, top_k = torch.topk(output[1].permute(0, 2, 1), 3, dim=1)
                top1_predicted = top_k[:, 0, :]
                top2_predicted = top_k[:, 1, :]
                top3_predicted = top_k[:, 2, :]
            else:
                loss = criterion(output, y_batch)
                _, top_k = torch.topk(output, 3, dim=1)
                top1_predicted = top_k[:, 0, :]
                top2_predicted = top_k[:, 1, :]
                top3_predicted = top_k[:, 2, :]

            if args.various_length:
                total += sum(input_len)
                correct_packed = pack_padded_sequence(top1_predicted == y_batch, input_len, batch_first=True,
                                                      enforce_sorted=False)
                correct += correct_packed.data.sum()
            else:
                pass

            accuracy = 100 * correct / total
            total_loss += loss

        avg_loss = total_loss / step

        if print_result:
            if semantic:
                if len(x_batch.size()) == 3:
                    print("Input typo:" + idx2chars(x_batch[0, :, 0].long()))
                else:
                    print("Input typo:" + idx2chars(x_batch[0, :].long()))
                pred_idx = top1_predicted[0]
                label = y_batch[0]
                print("original sentence:" + idx2chars(label))
                print("top1_predicted sentence:" + idx2chars(pred_idx))
                print("top2_predicted sentence:" + idx2chars(top2_predicted[0]))
                print("top3_predicted sentence:" + idx2chars(top3_predicted[0]) + '\n')

    if print_result:
        print('Accuracy on the ' + state + ' data: {:2.2f} % \n'.format(accuracy))

    return avg_loss, accuracy


def three_hot_encoder(output, cls=False):
    # output : [batch_size, vocab_size, length]
    if cls:
        output = output[:, :, 1:]
    else:
        output = output
    zero_tensor = torch.zeros(output.shape)

    _, topk = torch.topk(output, 3, dim=1)
    top1_predicted = topk[:, 0, :]
    top2_predicted = topk[:, 1, :]
    top3_predicted = topk[:, 2, :]

    one_hot1 = (torch.arange(output.shape[1]) == top1_predicted[..., None]).long()
    one_hot2 = (torch.arange(output.shape[1]) == top2_predicted[..., None]).long()
    one_hot3 = (torch.arange(output.shape[1]) == top3_predicted[..., None]).long()

    return one_hot1 + one_hot2 + one_hot3


def save_model(state_dict_model, path):
    torch.save(state_dict_model, path, _use_new_zipfile_serialization=False)


def load_model(init_model, path, evaluate_mode=False):
    if torch.cuda.is_available():
        init_model.load_state_dict(torch.load(path))
    else:
        init_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

    if evaluate_mode:
        init_model.eval()
    return init_model


def idx2chars(indices):
    '''
    chars_from_indices = ''
    for idx in indices:
        chars_from_indices = chars_from_indices + chars[idx]
    '''
    chars_from_indices = ''.join([chars[ind] for ind in indices])
    return chars_from_indices


def remove_non_silence_noises(input_text):
    """
      Removes non_silence noises from a transcript
    """
    non_silence_noises = ["noise", "um", "ah", "er", "umm", "uh", "mm", "mn", "mhm", "mnh", "<START>", "<END>"]
    re_non_silence_noises = re.compile(r"\b({})\b".format("|".join(non_silence_noises)))
    return re.sub(re_non_silence_noises, '', input_text)


def wer(ref, hyp, remove_nsns=False):
    """
      Calculate word error rate between two string or time_aligned_text objects
      >>> wer("this is a cat", "this is a dog")
      25.0
    """
    # remove tagged noises
    # ref = re.sub(re_tagged_noises, ' ', ref)
    # hyp = re.sub(re_tagged_noises, ' ', hyp)
    ref = re.sub('^<START>|<EOS>$', '', ref)
    hyp = re.sub('^<START>|<EOS>$', '', hyp)

    # optionally, remove non silence noises
    if remove_nsns:
        ref = remove_non_silence_noises(ref)
        hyp = remove_non_silence_noises(hyp)

    # clean punctuation, etc.
    # ref = clean_up(ref)
    # hyp = clean_up(hyp)

    # calculate WER
    return editdistance.eval(ref.split(' '), hyp.split(' ')), len(ref.split(' '))


def cer(ref, hyp, remove_nsns=False):
    """
      Calculate character error rate between two strings or time_aligned_text objects
      >>> cer("this cat", "this bad")
      25.0
    """
    ref = re.sub('^<START>|<EOS>$', '', ref)
    hyp = re.sub('^<START>|<EOS>$', '', hyp)

    if remove_nsns:
        ref = remove_non_silence_noises(ref)
        hyp = remove_non_silence_noises(hyp)

    # ref = clean_up(ref)
    # hyp = clean_up(hyp)

    # calculate per line CER
    return editdistance.eval(ref, hyp), len(ref)


def test_plot(x_batch, label, save_path):
    x_dic = {}
    y_dic = {}
    width = x_batch[0, 0, 3]
    height = x_batch[0, 0, 4]
    for i, char in enumerate(idx2chars(label)):
        x_value = float(x_batch[0][i][1] * width * 100)
        y_value = -float(x_batch[0][i][2] * height * 100)

        x_dic.setdefault(char, []).append(x_value)
        y_dic.setdefault(char, []).append(y_value)

    # Plot all points per name
    fig, ax = plt.subplots()

    ##only the mean points
    # for c in x_dic.keys():
    #     plt.scatter(np.mean(x_dic[c]), np.mean(y_dic[c]), s=10)
    #     pylab.ylim([-100, 550])
    #
    # for char in x_dic.keys():
    #     ax.annotate(char, (np.mean(x_dic[char]), np.mean(y_dic[char])), xytext=(np.mean(x_dic[char]) + 10,
    #                                                                             np.mean(y_dic[char]) + 10))

    # ax.text(500, 1, 'user analysis', verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=15)

    # for all the points
    for c in x_dic.keys():
        plt.scatter(x_dic[c], y_dic[c], s=2)
        # pylab.ylim([-100, 550])

    for d in x_dic.keys():
        plt.scatter(np.mean(x_dic[d]), np.mean(y_dic[d]), s=10, c='black')

    for char in x_dic.keys():
        # ax.annotate(char, (np.mean(x_dic[char]), np.mean(y_dic[char])), xytext=(np.mean(x_dic[char]) + 10, \
        #                                                                         np.mean(y_dic[char]) + 10))
        plt.text(np.mean(x_dic[char]) + 5, np.mean(y_dic[char]) + 5, char, weight='bold')

    # ax.text(900, -50, 'user analysis', verticalalignment='bottom', horizontalalignment='right', color='green', fontsize=15)

    # plt.legend(loc='best')
    fig.savefig(save_path)
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    distance, length = wer('I am a boy but you are a girl.', 'I as d d s a a asdfg fga sd you are s girl.', True)
    print(distance)
    print(length)

