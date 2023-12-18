import os
import json
import data
import time
import models
import argparse
from util import util

class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', required=True, help='path to victim dataset')
        parser.add_argument('--name', type=str, default='none', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--result_dir', default='./results', type=str, help='directory used to save evaluation results')
        # model parameters
        parser.add_argument('--model', type=str, default='backbone', help='chooses which model to use')
        parser.add_argument('--backbone', type=str, default='vgg16', help='chooses which the victim model to use')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--norm', type=str, default='batch', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        # dataset parameters
        parser.add_argument('--classes', type=int, default=0, help='number of classes')
        parser.add_argument('--seed', type=int, default=992, help='number of seed')
        parser.add_argument('--serial_batches', action='store_true', default = True, help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=2, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=0, help='input batch size')
        parser.add_argument('--load_size', type=int, default=32, help='scale images to this size')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--clear', action='store_false', help='if specified, clear all the historical weight files till current training epoch.')
        self.initialized = True
        return parser
    
    def read_configs(self, parser, config_names):
        """_summary_

        Args:
            parser (argparse): argparse
            config_names (List): All the configs you need to read. For instance, configs of networks (e.g., generator, discriminator) and optimizers
        """
        phase_dicts = {'netG':'generators', 'netD':'discriminators', 'optim':'optimizers'}
        for obj in config_names:
            obj_lists = {k:v for k,v in vars(parser.parse_args()).items() if obj in k}
            phase= phase_dicts[obj]
            all_dicts = {}
            if len(obj_lists) > 0:
                for objid, name in obj_lists.items():
                    try:
                        with open(f'./configs/{phase}/{name}.json', 'r', encoding='UTF-8') as f:
                            paras = json.load(f)
                        all_dicts[objid] = paras
                    except:
                        raise NotImplementedError('[%s] model name [%s] is not recognized. Maybe there is no configs or implementation' % (phase_dicts[obj], name))
                parser.add_argument(f'--{obj}dicts', type=dict, default=all_dicts, help='specified parameters for [%s]'%objid)
        return parser
    
    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        # modify dataset-related parser options
        self.dataset_name = opt.dataroot.split('/')[-1]
        dataset_option_setter = data.get_option_setter(self.dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)
        
        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        parser = self.read_configs(parser, ['netG', 'netD', 'optim'])
        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
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
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, '{}_opt.txt'.format(opt.phase))
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test'
        if opt.name == 'none':
            prefix = time.strftime('%Y%m%d_%H%M%S', time.localtime())
            opt.name = f'{prefix}'
        self.print_options(opt)
        self.opt = opt
        return self.opt
