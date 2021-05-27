import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import IterableDataset
import pandas as pd
import numpy as np
import random
from random import randint
from torch.nn.utils.rnn import pad_sequence
import math

np.random.seed(1)
random.seed(1)
chars = ['[PAD]', '[MASK]', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', \
         'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', ' ', '.']
char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}


class DataFixedLength(Dataset):
    """Data loader for the short-term decoder.
    DataShorTerm parses the raw data and extracts the following pairs:
    (input_1, input_2, input_3, gt_char_1, gt_char_2, gt_char_3).
    """

    def __init__(self, csv_path, interval=False, length=3):
        df = pd.read_csv(csv_path)
        self.relative = True
        self.length = length
        self.dataset_input, self.dataset_label = self.preprocessing(df, interval=interval)

    def preprocessing(self, csv_data, interval=False):
        total_input = []
        total_label = []
        sentence_column = csv_data['sentence']
        x_column, y_column = csv_data['x_list'], csv_data['y_list']
        width_column, height_column = csv_data['width'], csv_data['height']
        # time_column = csv_data['time']
        for sentence, x_list, y_list, width, height in zip(sentence_column, x_column, y_column,
                                                           width_column, height_column):

            sentence_list = list(sentence)
            normalized_x_list = [float(x) / width for x in x_list.split(',')]
            normalized_y_list = [float(y) / height for y in y_list.split(',')]
            width_list = len(sentence_list) * [width / 1000]
            # print("width list is", width_list)
            height_list = len(sentence_list) * [height / 1000]
            # print("height list is", height_list)

            if len(normalized_x_list) != len(normalized_y_list):
                print("Error!! x and y different lengths!")

            if len(sentence_list) != len(normalized_y_list):
                print("Error! sentence and y have different lengths!")
            full_length = len(sentence_list)

            one_input = []
            for i, char in enumerate(sentence_list):
                if i < full_length - self.length - 1:
                    one_input = [width_list[0]] + [height_list[0]]
                    if interval:
                        three_x = normalized_x_list[i:i + self.length]
                        three_y = normalized_y_list[i:i + self.length]
                        one_input = one_input + three_x + three_y + get_diff(three_x) + get_diff(three_y)
                    else:
                        one_input = one_input + normalized_x_list[i:i + self.length] + normalized_y_list[i:i + self.length]

                    one_label = [char_to_idx[character] for character in sentence_list[i:i + 3]]
                total_input.append(one_input)
                total_label.append(one_label)

        return np.array(total_input), np.array(total_label)

    def __len__(self):
        return self.dataset_input.shape[0]

    def __getitem__(self, idx):
        return self.dataset_input[idx], np.array(self.dataset_label[idx])


class DataVariableLength(Dataset):
    """Data loader for the long-term decoder.
    DataLongTerm parses the raw data and extracts
    (seq. of user input, g.t. seq. of characters) pairs of each full sentence.
    """
    def __init__(self, csv_path, min_length=13, full_sentence=False, augment=False):
        df = pd.read_csv(csv_path)
        self.full_sentence = full_sentence
        self.min_length = min_length
        data_file = df[df['length'] >= self.min_length]
        self.df = data_file
        self.data_length = self.df.count()[0]
        self.min_x = 0
        self.min_y = 0
        self.augment = augment

    def process_csv(self, data, idx):
        one_sample = data.iloc[idx]
        selected_length = random.randint(self.min_length, one_sample['length'])
        # width_one = one_sample['width']
        # height_one = one_sample['height']
        width_one = one_sample['width'] * 1.1
        height_one = one_sample['height'] * 1.4

        var_x = width_one - max([float(x) for x in one_sample['x_list'].split(',')])
        var_y = height_one - max([float(y) for y in one_sample['y_list'].split(',')])
        shift_x = random.randint(-3, 3)
        shift_y = random.randint(-3, 3)
        # if var_x < self.min_x:
        #     self.min_x = var_x
        #     print('x: ' + str(var_x))
        # if var_y < self.min_y:
        #     self.min_y = var_y
        #     print('y: ' + str(var_y))

        if self.full_sentence:
            cropped_chars = one_sample['sentence']
            norm_x = np.expand_dims(np.array([(float(x) + random.uniform(-3, 3)) / width_one if random.random() < 0.3
                                              else (float(x)) / width_one for x in one_sample['x_list'].split(',')]), 1)
            norm_y = np.expand_dims(np.array([(float(y) + random.uniform(-3, 3)) / height_one if random.random() < 0.3
                                              else (float(y)) / height_one for y in one_sample['y_list'].split(',')]), 1)
        else:
            cropped_chars = one_sample['sentence'][0:selected_length]
            if self.augment:
                random.seed(0)
                norm_x = np.expand_dims(np.array([(float(x) + random.uniform(-3, 3)) / width_one
                                                  if random.random() < 0.3 else (float(x)) / width_one for x in one_sample['x_list'].split(',')][0:selected_length]), 1)
                norm_y = np.expand_dims(np.array([(float(y) + random.uniform(-3, 3)) / height_one
                                                  if random.random() < 0.3 else (float(y)) / height_one for y in one_sample['y_list'].split(',')][0:selected_length]), 1)
            else:
                norm_x = np.expand_dims(np.array(
                    [float(x) / width_one for x in one_sample['x_list'].split(',')][
                    0:selected_length]), 1)
                norm_y = np.expand_dims(np.array(
                    [float(y) / height_one for y in one_sample['y_list'].split(',')][
                    0:selected_length]), 1)
        char_np = np.array([char_to_idx[char] for char in cropped_chars])
        input_char_np = np.expand_dims(char_np, 1)
        # char_idx_list = np.copy(char_np)
        # label = np.copy(char_np)
        # assuming we have user input
        width = np.expand_dims(np.array([(one_sample['width'])] * len(char_np)) / 1000, 1)
        # print("width is ", width)
        height = np.expand_dims(np.array([(one_sample['height'])] * len(char_np)) / 1000, 1)
        # print("height is ", height)
        char_idx_list = torch.tensor(np.concatenate((input_char_np, norm_x, norm_y, width, height), axis=1))
        label = torch.tensor(np.copy(char_np))

        return char_idx_list, label

    def __getitem__(self, idx):
        data_arr, label = self.process_csv(self.df, idx)
        return data_arr, label

    def __len__(self):
        return self.data_length


class MaskedLM(Dataset):
    """Data loader for the long-term decoder.
    DataLongTerm parses the raw data and extracts
    (seq. of user input, g.t. seq. of characters) pairs of each full sentence.

    """
    def __init__(self, csv_path, min_length=13, full_sentence=False, inference=False):
        df = pd.read_csv(csv_path)
        self.full_sentence = full_sentence
        self.min_length = min_length
        data_file = df[df['length'] >= self.min_length]
        self.df = data_file
        self.data_length = self.df.count()[0]
        self.inference = inference

    def process_csv(self, data, idx):
        one_sample = data.iloc[idx]
        if self.full_sentence:
            cropped_chars = one_sample['sentence']
        else:
            selected_length = random.randint(self.min_length, one_sample['length'])
            cropped_chars = one_sample['sentence'][0:selected_length]
        masked_sentence = []
        label = []
        mask = []
        full_label = []
        for char in cropped_chars:
            if char not in chars:
                masked_sentence += [0]
                label += [0]
                mask += [0]
            else:
                prob = random.random()
                if prob < 0.85:
                    masked_sentence += [char_to_idx[char]]
                    if self.inference:
                        label += [char_to_idx[char]]
                        mask += [1]
                    else:
                        label += [0]
                        mask += [0]
                elif prob < 0.97:
                    masked_sentence += [char_to_idx['[MASK]']]
                    label += [char_to_idx[char]]
                    mask += [1]
                elif prob < 0.985:
                    masked_sentence += [randint(0, len(chars) - 1)]
                    label += [char_to_idx[char]]
                    mask += [1]
                else:
                    masked_sentence += [char_to_idx[char]]
                    label += [char_to_idx[char]]
                    mask += [1]

                # if prob < 0.85:
                #     masked_sentence += [char_to_idx[char]]
                #     if self.inference:
                #         label += [char_to_idx[char]]
                #         mask += [1]
                #     else:
                #         label += [0]
                #         mask += [0]
                # elif prob < 0.985:
                #     masked_sentence += [randint(0, len(chars) - 1)]
                #     label += [char_to_idx[char]]
                #     mask += [1]
                # else:
                #     masked_sentence += [char_to_idx[char]]
                #     label += [char_to_idx[char]]
                #     mask += [1]

                full_label += [char_to_idx[char]]

        masked_char = torch.tensor(masked_sentence)
        label = torch.tensor(label)
        mask = torch.tensor(mask)
        full_label = torch.tensor(full_label)

        return masked_char, label, mask, full_label

    def __getitem__(self, idx):
        data_arr, label, mask, full_label = self.process_csv(self.df, idx)
        return data_arr, label, mask, full_label

    def __len__(self):
        return self.data_length


class Masked1BW(IterableDataset):
    """Data loader for the long-term decoder.
    DataLongTerm parses the raw data and extracts
    (seq. of user input, g.t. seq. of characters) pairs of each full sentence.

    """
    def __init__(self, csv_path, min_length=13, full_sentence=False, inference=False):
        self.filename = csv_path
        self.full_sentence = full_sentence
        self.min_length = min_length
        self.inference = inference


    def line_mapper(self, one_sample):
        if self.full_sentence:
            cropped_chars = one_sample
        else:
            if len(one_sample) < self.min_length:
                selected_length = len(one_sample)
            else:
                selected_length = random.randint(self.min_length, len(one_sample))
                if selected_length > 200:
                    selected_length = 200
            cropped_chars = one_sample[0:selected_length]
        masked_sentence = []
        label = []
        for char in cropped_chars:
            if char not in chars:
                char = '[UNK]'
                masked_sentence += [char_to_idx[char]]
                label += [0]
            else:
                prob = random.random()
                if prob < 0.85:
                    masked_sentence += [char_to_idx[char]]
                    if self.inference:
                        label += [char_to_idx[char]]
                    else:
                        label += [0]
                elif prob < 0.97:
                    masked_sentence += [char_to_idx['[MASK]']]
                    label += [char_to_idx[char]]
                elif prob < 0.985:
                    masked_sentence += [randint(0, len(chars) - 1)]
                    label += [char_to_idx[char]]
                else:
                    masked_sentence += [char_to_idx[char]]
                    label += [char_to_idx[char]]


        masked_char = torch.tensor(masked_sentence)
        label = torch.tensor(label)

        return masked_char, label


    def __iter__(self):
        file_itr = open(self.filename)
        mapped_itr = map(self.line_mapper, file_itr)
        return mapped_itr


def pad_variable(batch):
    sorted_batch = sorted(batch, key=lambda x: x[0].shape[0], reverse=True)             # sort the batch in descending order
    sequences = [x[0] for x in sorted_batch]                                            # length of sequence
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)     # pad the sequence
    lengths = [len(x) for x in sequences]
    labels = [x[1] for x in sorted_batch]
    labels_padded = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True)

    if len(sorted_batch[0]) > 2:
        masks = [x[2] for x in sorted_batch]
        masks_padded = torch.nn.utils.rnn.pad_sequence(masks, batch_first=True)

        full_labels = [x[3] for x in sorted_batch]
        full_labels_padded = torch.nn.utils.rnn.pad_sequence(full_labels, batch_first=True)

    else:
        masks_padded = lengths
        full_labels_padded = labels_padded

    return sequences_padded, labels_padded.long(), lengths, masks_padded, full_labels_padded.long()


def get_diff(sequential_data):
    """
    get the difference between prior and later data in sequential data list
    """
    return [x - sequential_data[i - 1] for i, x in enumerate(sequential_data)][1:]


def make_typo_dictionary(short_predicted_csv_path, nearest=False):
    """
    make typos for each character
    if nearest is
        True: make artificial typos for each character depending on nearest positions on the physical keyboard
        False: get the real typos from trained short predictor
    """
    typo_dict = {}
    if nearest:
        typo_dict['q'] = list('asw')
        typo_dict['w'] = list('qasde')
        typo_dict['e'] = list('wsdfr')
        typo_dict['r'] = list('edfgt')
        typo_dict['t'] = list('rfghy')
        typo_dict['y'] = list('tghju')
        typo_dict['u'] = list('yhjki')
        typo_dict['i'] = list('ujklo')
        typo_dict['o'] = list('iklp')
        typo_dict['p'] = list('ol')
        typo_dict['a'] = list('qwsxz')
        typo_dict['s'] = list('qwedxaz')
        typo_dict['d'] = list('wersfxc')
        typo_dict['f'] = list('ertdgcv')
        typo_dict['g'] = list('rtyfhvb')
        typo_dict['h'] = list('tyugjbn')
        typo_dict['j'] = list('yuihknm')
        typo_dict['k'] = list('uoijlm')
        typo_dict['l'] = list('iopk')
        typo_dict['z'] = list('asx')
        typo_dict['x'] = list('zsdc ')
        typo_dict['c'] = list('xdfv ')
        typo_dict['v'] = list('cfgb ')
        typo_dict['b'] = list('vghn ')
        typo_dict['n'] = list('bhjm ')
        typo_dict['m'] = list('njk ')
        typo_dict['.'] = list(' l')
        typo_dict[' '] = list('zxcvbnm.')

    else:
        df = pd.read_csv(short_predicted_csv_path)
        ori_sentences = df['sentence']
        pred_sentences = df['pred_sentence']
        for ori_chars, pred_chars in zip(ori_sentences, pred_sentences):
            char_np = np.array([char_to_idx[char] for char in ori_chars])
            pred_char_np = np.array([char_to_idx[char] for char in pred_chars])

            for ori_char, pred_char in zip(char_np, pred_char_np):
                if ori_char != pred_char:
                    if typo_dict.get(ori_char) is None:
                        typo_dict[ori_char] = [pred_char]
                    else:
                        if not pred_char in typo_dict[ori_char]:
                            typo_dict[ori_char].append(pred_char)

    return typo_dict


def get_dataloader(data_path, batch_size, min_length, various=True, full_sentence=False, test=False, augment=False, masked_LM=False, inference=False):
    if various:
        if masked_LM:
            dataset = MaskedLM(data_path, full_sentence=full_sentence, min_length=min_length, inference=inference)
            # dataset = Masked1BW(data_path, full_sentence=full_sentence, min_length=min_length, inference=inference)
            data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=pad_variable)
            return data_loader
        dataset = DataVariableLength(data_path, full_sentence=full_sentence, min_length=min_length, augment=augment)
        shuffle = True
        if test:
            shuffle = False

        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=pad_variable)
    else:
        dataset = DataFixedLength(data_path, interval=True, length=3)
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return data_loader


if __name__ == "__main__":
    fixed = DataFixedLength('./data/TMI_Data/data_val.csv', True)
    # variable = MaskedLM('sample_1bw.txt', full_sentence=False, min_length=13)
    # print(len(variable))
    # print(variable[100])
    dataloader = DataLoader(fixed, batch_size=16, shuffle=True)
    x_batch, y_batch = next(iter(dataloader))

    print("x is {}".format(x_batch))
    print("x shape is {}".format(x_batch.shape))
    # print("length is {}".format(length))
    print("y is {}".format(y_batch))
    print("y shape is {}".format(y_batch.shape))
