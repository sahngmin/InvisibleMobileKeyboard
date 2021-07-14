import torch.nn as nn
from options import TrainOptions
from utils import evaluate, save_model
import torch
import data
import utils
from test import test_
from torch.nn import DataParallel
from models import BiRNN, IKeyboard, SANCD, ShortTermMLP, BERT, BiRNNLinearBert


if __name__ == "__main__":
    args = TrainOptions().parse()
    args = utils.bashRun(args)
    torch.manual_seed(args.seed)                            # for reproducibility
    torch.backends.cudnn.deterministic = True       # for reproducibility
    torch.backends.cudnn.benchmark = False          # for reproducibility

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.bigru:
        if args.bert:
            predictor = BiRNNLinearBert.RNNLinearBERT(gru_stack_bert=args.gru_stack_bert, args=args).to(device)
        else:
            predictor = BiRNN.BidirectionalRNN(char_embed_size=args.char_embed_size, nhid=args.nhid, nlayer=args.nlayers,
                                               rnn_type='GRU').to(device)
            model_name = 'BiRNN'

    elif args.bert:
        if args.masked_LM:
            args.train_data = './data/Masked_CLM/1BW_LM.csv'
            args.val_data = './data/data_train.csv'
            model_name = 'BERT'
        else:
            args.custom_input = True
            model_name = 'BERT'
        predictor = BERT.BERT(args=args).to(device)

    elif args.ikeyboard:
        predictor = IKeyboard.Ikeyboard(args=args)
        model_name = 'IKeyboard'

    elif args.sa_ncd:
        predictor = SANCD.SANCD(args=args).to(device)
        model_name = 'SANCD'

    else:
        predictor = ShortTermMLP.ShortTermDecoder().to(device)
        model_name = 'ShortTermMLP'

    save_path = '{}_h{}_n{}_{}.pth'.format(model_name, str(args.nhid), args.nlayers, 'MLM' if args.masked_LM else '')

    if args.multi_gpu:
        predictor = DataParallel(predictor, output_device=1)
    else:
        predictor = predictor.to(device)


    dataloader_train = data.get_dataloader(args.train_data, batch_size=args.batch_size, min_length=args.length,
                                           various=args.various_length, full_sentence=True,
                                           augment=args.augment, masked_LM=args.masked_LM)
    dataloader_val = data.get_dataloader(args.val_data, batch_size=args.batch_size, min_length=args.length,
                                         various=args.various_length, full_sentence=args.full_sentence,
                                         augment=args.augment, masked_LM=args.masked_LM, inference=True)
    dataloader_mckenzie = data.get_dataloader(args.mckenzie_data, batch_size=args.batch_size, min_length=args.length,
                                              various=args.various_length, full_sentence=args.full_sentence,
                                              augment=False, masked_LM=args.masked_LM, inference=True)
    dataloader_freq1000 = data.get_dataloader(args.freq1000_data, batch_size=args.batch_size, min_length=args.length,
                                              various=args.various_length, full_sentence=args.full_sentence,
                                              augment=False, masked_LM=args.masked_LM, inference=True)
    dataloader_test = data.get_dataloader(args.test_data, batch_size=args.batch_size, min_length=args.length,
                                          various=args.various_length, full_sentence=args.full_sentence, augment=False,
                                          masked_LM=args.masked_LM, inference=True)

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    best_model = getattr(eval(model_name), 'train_')(predictor, dataloader_train, criterion, args, device, dataloader_val, save_path=save_path)

    val_loss, val_acc, val_wer = test_(args, predictor, best_model, dataloader_val, print_count=False)
    freq1000_loss, freq1000_acc, freq1000_wer = test_(args, predictor, best_model, dataloader_freq1000)
    mckenzie_loss, mckenzie_acc, mckenzie_wer = test_(args, predictor, best_model, dataloader_mckenzie, print_count=False)
    test_loss, test_acc, test_wer = test_(args, predictor, best_model, dataloader_test, print_count=False)

    print('val ({:5.4f} | {:2.2f}%) |  mckenzie ({:5.4f} | {:2.2f}%) | freq1000 ({:5.4f} | {:2.2f}%) |'
          ' Test ({:5.4f} | {:2.2f}%)'.format(val_loss, val_acc, mckenzie_loss, mckenzie_acc, freq1000_loss,
                                                   freq1000_acc, test_loss, test_acc))

    print("WER : val {:2.2f}%  mckenzie {:2.2f}%  freq1000 {:2.2f}%   test {:2.2f}%".format(val_wer, mckenzie_wer, freq1000_wer, test_wer))

    print('-' * 80)

