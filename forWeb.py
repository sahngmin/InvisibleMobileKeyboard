import model
import torch
from flask import Flask, render_template, jsonify, request


app = Flask(__name__)


x_list_str = []
y_list_str = []

chars = ['PADDING', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', \
         'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '.']
char_to_idx = {ch: i for i, ch in enumerate(chars)}


def predict_sentence(x_list, y_list, part, predictor, w, h):
    sentence = ''
    length = len(x_list)
    d, r = length // part, length % part
    for i in range(d):
        part_x = x_list[part * i: part * (i + 1)]
        part_y = y_list[part * i: part * (i + 1)]
        if part == 3:
            sentence = sentence + get_three_char(part_x, part_y, predictor, w, h)
        elif part == 9:
            sentence = sentence + get_nine_char(part_x, part_y, predictor, w, h)

    last_part_x, last_part_y = x_list[-part:], y_list[-part:]

    if part == 3:
        sentence = sentence + get_three_char(last_part_x, last_part_y, predictor, w, h)[part - r:]
    elif part == 9:
        sentence = sentence + get_nine_char(last_part_x, last_part_y, predictor, w, h)[part - r:]

    return sentence

def get_three_char(three_list_x, three_list_y, predictor, w, h, relative=True):
    """
    returns three characters predicted by Short-term decoder
    """
    x_input = [x / w for x in three_list_x]
    y_input = [y / h for y in three_list_y]

    one_list = [w/100, h/100]
    for x in x_input:
        one_list = one_list + [x]

    for y in y_input:
        one_list = one_list + [y]

    if relative:
        one_list = one_list + get_diff(x_input)
        one_list = one_list + get_diff(y_input)

    one_input = torch.FloatTensor(one_list)
    one_input = one_input.unsqueeze(0)
    input_len = torch.LongTensor([9])
    output = predictor(one_input, input_len)
    _, predicted = torch.max(output, 1)
    phrase = idx2chars(predicted[0])
    return phrase


def get_nine_char(nine_list_x, nine_list_y, predictor, w, h):
    """
    returns nine characters predicted by Semantic decoder
    """
    x_input = [x / w for x in nine_list_x]
    y_input = [y / h for y in nine_list_y]


    one_list = []
    for x, y in zip(x_input, y_input):
        one_list.append([0, x, y, w / 100, h / 100])

    one_input = torch.FloatTensor(one_list)
    one_input = one_input.unsqueeze(0)
    input_len = torch.LongTensor([9])
    output = predictor(one_input, input_len)
    _, predicted = torch.max(output, 1)
    phrase = idx2chars(predicted[0])
    return phrase

def get_diff(sequential_data):
    """
    get the difference between prior and later data in sequential data list
    """
    return [x - sequential_data[i - 1] for i, x in enumerate(sequential_data)][1:]

def load_model(init_model, path, evaluate_mode=False):
    init_model.load_state_dict(torch.load(path))
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



touch_x = request.args.get('num_x')  # get the variable name 'num_x' from jquery and store it in touch_x
touch_y = request.args.get('num_y')
w = float(request.args.get('device_width'))
h = float(request.args.get('device_height'))
short_predict = True if request.args.get('short_pred') == 'true' else False
long_predict = True if request.args.get('long_pred') == 'true' else False
x_list_str.append(touch_x)
x_list = [float(x) for x in x_list_str]
y_list_str.append(touch_y)
y_list = [float(y) for y in y_list_str]
if long_predict:
    full_predict_model = model.StatisticalDecoder(feat_size=4, rnn_units=128, num_layers=4, rnn_type="GRU",
                                 semantic_decoding=True,
                                 data_collected=True, use_char_embed=False,
                                 char_embedding_size=8)

    full_predict_model = load_model(full_predict_model, 'checkpoints/ShortPredict+GRU_second', evaluate_mode=True)

    if len(x_list) > 9:
        predicted = predict_sentence(x_list, y_list, part=9, predictor=full_predict_model, w=w, h=h)
print(x_list, type(x_list[0]))
print(y_list, len(y_list))
print(w, h, type(w))
print("short is {} and the type is {}".format(short_predict, type(short_predict)))
print("long is {} and the type is {}".format(long_predict, type(long_predict)))
print("the predicted sentence is {}".format(predicted))


