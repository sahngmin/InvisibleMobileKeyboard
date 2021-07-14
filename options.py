import argparse


class BaseOptions:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--name', type=str, default='TMI-Keyboard',
                            help='name of the experiment. It decides where to store samples and models')

        self.parser = parser
        self.arg_parsed = False

    def parse(self):
        # get the basic options
        if not self.arg_parsed:
            opt = self.parser.parse_args()
            self.opt = opt
            self.arg_parsed = True
        self.print_options(self.opt)

        return self.opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        '''
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        mkdir(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')
        '''


class TrainOptions(BaseOptions):
    def __init__(self):
        super(TrainOptions, self).__init__()
        # Training options
        self.parser.add_argument('--seed', default=0, help="fix randomness")
        self.parser.add_argument('--test_seed', default=1, help="fix randomness")
        self.parser.add_argument('--epoch', default=1000, help="number of training epochs")
        self.parser.add_argument('--batch_size', default=16)
        self.parser.add_argument('--batch_norm', default=False)
        self.parser.add_argument('--optimizer', default='ADAM', help='SGD or ADAM or RMS')
        self.parser.add_argument('--lr', default=0.0001,
                                 help="learning rate : 0.0001 for BERT, 3.0 for BiGRU")
        self.parser.add_argument('--gru_lr_down', default=0.5)
        self.parser.add_argument('--augment', default=True)
        self.parser.add_argument('--multi_gpu', default=False)

        self.parser.add_argument('--step_lr', default=10, type=int)
        self.parser.add_argument('--case', default=0, help='for ablation case study')
        self.parser.add_argument('--early_stop', default=10)

        # Dataset
        self.parser.add_argument('--train_data', default='./data/data_normalized/data_train.csv')
        self.parser.add_argument('--val_data', default='./data/data_normalized/data_val.csv')
        self.parser.add_argument('--test_data', default='./data/data_normalized/data_test.csv')
        self.parser.add_argument('--mckenzie_data', default='./data/data_normalized/data_mckenzie.csv')
        self.parser.add_argument('--freq1000_data', default='./data/data_normalized/data_freq1000.csv')

        # Options of Input
        self.parser.add_argument('--statistic_decoding', default=True, help="prediction only based on x, y location")
        self.parser.add_argument('--semantic_decoding', default=False,
                                 help="prediction using Language Model, input has character index")

        self.parser.add_argument('--various_length', default=True,
                                 help="whether input phrase has various or fixed length")
        self.parser.add_argument('--full_sentence', default=False,
                                 help="")
        self.parser.add_argument('--length', default=9, help="minimum length of input phrase")
        self.parser.add_argument('--typo_generate', default=False,
                                 help="")
        self.parser.add_argument('--nearest_typo', default=False,
                                 help="if True, make artificial typos for each character depending on nearest positions on the physical keyboard")

        # Embedding size
        self.parser.add_argument('--char_embed_size', default=128, help="char embedding dimension")
        self.parser.add_argument('--feat_size', default=4,
                                 help="final feature size of seperate path before softmax decoding")

        # Model Selection
        self.parser.add_argument('--short_term_predict', default=False, help="short prediction for three characters")

        self.parser.add_argument('--bigru', default=False, help="use Bi-directional GRU as a decoding model")
        self.parser.add_argument('--bert', default=False, help="use Bert as a decoding model")
        self.parser.add_argument('--masked_LM', default=False)
        self.parser.add_argument('--ikeyboard', default=False, help="use ikeyboard")
        self.parser.add_argument('--gru_stack_bert', default=False, help="use Bert as a decoding model")
        self.parser.add_argument('--custom_input', default=False, help="use 4 dimension statistic input from our data")
        self.parser.add_argument('--TMIkeyboard', default=True, help="use TMIkeyboard")


        # options for ShortTermDecoder
        self.parser.add_argument('--hidden_nodes', default=[128, 128, 128],
                                 help="size of first hidden layer for Short-term decoder")

        # Options of Transformer(BERT) & BidirectionalRNN
        self.parser.add_argument('--nhid', default=512,
                                 help="number of nodes in hidden layer in Decoder")
        self.parser.add_argument('--nlayers', default=12,
                                 help="number of encoder block in Transformer(BERT) or number of layers in Bi-GRU")
        self.parser.add_argument('--nhead', default=8,
                                 help="the number of heads in the multi-head attention models")
        self.parser.add_argument('--intermediate_size', default=1024,
                                 help="Intermediate size of BERT")
        self.parser.add_argument('--excessive_output', default=True, help="Excessive output for BERT (only uses first 31 dimension of 256 output")
        self.parser.add_argument('--intermediate_loss', default=True, help="")
        self.parser.add_argument('--intermediate_stop', default=30, type=int, help="")
        self.parser.add_argument('--cm_threshold', default=0.45, type=float, help="")


        # Save&load options
        self.parser.add_argument('--save', default=True)
        self.parser.add_argument('--save_path', default='', help="define name when saving the model")

        self.parser.add_argument('--load_gru', default=True)
        self.parser.add_argument('--geometric_decoder_path', default='BiRNN_h512_n12_.pth', help="location of pre-trained Geometric Decoder")

        self.parser.add_argument('--load_bert', default=True)
        self.parser.add_argument('--semantic_decoder_path', default='BERT_h512_n12_MLM_epoch2.pth',
                                 help="location of pre-trained Semantic Decoder (Transformer Encoder trained as Masked Character Language Model)")








