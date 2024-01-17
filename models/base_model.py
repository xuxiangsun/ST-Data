import os
import sys
sys.path.append("..")
import torch
import shutil
from tqdm import tqdm
from . import model_utils
import torch.nn.functional as F
from collections import OrderedDict
from abc import ABC, abstractmethod

class BaseModel(ABC):
    """This class is an abstract base class (ABC) for models.
    To create a subclass, you need to implement the following five functions:
        -- <__init__>:                      initialize the class; first call BaseModel.__init__(self, opt).
        -- <set_input>:                     unpack data from dataset and apply preprocessing.
        -- <forward>:                       produce intermediate results.
        -- <optimize_parameters>:           calculate losses, gradients, and update network weights.
        -- <modify_commandline_options>:    (optionally) add model-specific options and set default options.
    """

    def __init__(self, opt, meanstd):
        """Initialize the BaseModel class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions

        When creating your custom class, you need to implement your own initialization.
        In this fucntion, you should first call <BaseModel.__init__(self, opt)>
        Then, you need to define four lists:
            -- self.loss_names (str list):          specify the training losses that you want to plot and save.
            -- self.model_names (str list):         specify the images that you want to display and save.
            -- self.visual_names (str list):        define networks used in our training.
            -- self.optimizers (optimizer list):    define and initialize optimizers. You can define one optimizer for each network. If two networks are updated at the same time, you can use itertools.chain to group them. See cycle_gan_model.py for an example.
        """
        self.opt = opt
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  # get device name: CPU or GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)  # save all the checkpoints to save_dir
        self.load_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.netarc_dir = os.path.join(self.save_dir, 'netarc.txt')
        self.log_name = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'loss_log.txt')
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.optimizers = []
        self.meanstd = meanstd
        current_path = os.path.abspath(__file__)
        father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
        ffather_path = os.path.abspath(os.path.dirname(father_path) + os.path.sep + ".")
        savefile = ['models', 'util', 'data', 'options', 'tools', 'configs', ]
        for file in savefile:
            self.copy_allfiles(os.path.join(ffather_path, file), os.path.join(os.path.abspath(self.save_dir), 'src', file))
        self.metric = 0  # used for learning rate policy 'plateau'
        self.y_class = []
        self.cla_num = []
        self.data_name = self.opt.dataroot.split('/')[-1]
        self.result_path = os.path.join(self.opt.result_dir, self.opt.name)

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new model-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    def set_input(self, imgs, labels):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): includes the data itself and its metadata information.
        """
        pass

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        pass
 
    def restore_logits(self, inputs):
        if self.opt.nologits:
            pred_victim = F.log_softmax(inputs, dim=1).detach()
            if self.opt.logit_corr == 'min':
                pred_victim -= pred_victim.min(dim=1).values.view(-1, 1).detach()
            elif self.opt.logit_corr == 'mean':
                pred_victim -= pred_victim.mean(dim=1).view(-1, 1).detach()
            t_logits = pred_victim.detach()
        else:
            t_logits = inputs.detach() 
        return t_logits
    
    def copy_allfiles(self, src, dest):
        src_files = os.listdir(src)
        for file_name in src_files:
            if file_name in ['__pycache__', 'caches']:
                continue
            full_file_name = os.path.join(src, file_name)
            if os.path.isfile(full_file_name):
                if not os.path.exists(dest):
                    os.makedirs(dest)
                shutil.copy(full_file_name, dest)
            elif os.path.exists(full_file_name):
                self.copy_allfiles(full_file_name, dest+'/'+full_file_name.strip().split('/')[-1])

    @abstractmethod
    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        pass

    def setup(self, opt):
        """Load and print networks; create schedulers

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        if self.isTrain:
            self.schedulers = [model_utils.get_scheduler(optimizer, opt) for optimizer in self.optimizers]
        if not self.isTrain or opt.continue_train:
            load_suffix = '%d' % opt.load_iter if opt.load_iter > 0 else opt.epoch
            self.load_networks(load_suffix)
        self.print_networks()

    def eval(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.eval()

    def train(self):
        """Make models eval mode during test time"""
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                net.train()

    def update_learning_rate(self, ignore_lists=[]):
        """Update learning rates for all the networks; called at the end of every epoch"""
        for id, scheduler in enumerate(self.schedulers):
            if id in ignore_lists:
                continue
            if self.opt.lr_policy == 'plateau':
                scheduler.step(self.metric)
            else:
                scheduler.step()
        lrs = str(['%.7f'%optim.param_groups[0]['lr'] for optim in self.optimizers])
        message = 'learning rates = %s' % lrs
        print(message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    def get_current_losses(self):
        """Return traning losses / errors."""
        errors_ret = OrderedDict()
        for name in self.loss_names:
            if isinstance(name, str):
                errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
        return errors_ret

    def get_current_acc(self):
        """Return traning losses / accuracy"""
        acc_ret = OrderedDict()
        for name in self.acc_names:
            if name == 'totalacc':
                if isinstance(name, str):
                    acc_ret[name] = float(getattr(self, 'acc_' + name))  # float(...) works for both scalar tensor and float number
            else:
                if isinstance(name, str):
                    acc_ret[name] = getattr(self, 'acc_' + name)
        return acc_ret

    def save_networks(self, epoch, clear=True, suffix=None):
        """Save all the networks to the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                save_filename = '%s_%s_net_%s.pth' % (epoch, suffix, name) if suffix is not None else '%s_net_%s.pth' % (epoch, name)
                save_path = os.path.join(self.save_dir, save_filename)
                if os.path.exists(save_path):
                    os.remove(save_path)
                net = getattr(self, 'net' + name)
                if torch.cuda.is_available():
                    torch.save(net.state_dict(), save_path, _use_new_zipfile_serialization=False)
                if clear and os.path.exists(self.save_dir):
                    files = os.listdir(self.save_dir)
                    wfiles = [f for f in files if 'pth' in f and 'latest' not in f and 'best' not in f]
                    if suffix is None:
                        netwfiles = [int(f.strip().split('_')[0]) for f in wfiles if name in f]
                    else:
                        netwfiles = [int(f.strip().split('_')[0]) for f in wfiles if f'{suffix}_net_{name}' in f]
                    netwfiles.sort()
                    if len(netwfiles)>=2:
                        for delind in range(len(netwfiles)):
                            if delind == len(netwfiles)-1:
                                break
                            delfile = [f for f in wfiles if name in f and str(netwfiles[delind]) in f][0]
                            os.remove(os.path.join(self.save_dir, delfile))

    def __patch_instance_norm_state_dict(self, state_dict, module, keys, i=0):
        """Fix InstanceNorm checkpoints incompatibility (prior to 0.4)"""
        key = keys[i]
        if i + 1 == len(keys):  # at the end, pointing to a parameter/buffer
            if module.__class__.__name__.startswith('InstanceNorm') and \
                    (key == 'running_mean' or key == 'running_var'):
                if getattr(module, key) is None:
                    state_dict.pop('.'.join(keys))
            if module.__class__.__name__.startswith('InstanceNorm') and \
               (key == 'num_batches_tracked'):
                state_dict.pop('.'.join(keys))
        else:
            self.__patch_instance_norm_state_dict(state_dict, getattr(module, key), keys, i + 1)

    def load_networks(self, epoch):
        """Load all the networks from the disk.

        Parameters:
            epoch (int) -- current epoch; used in the file name '%s_net_%s.pth' % (epoch, name)
        """
        for name in self.model_names:
            if isinstance(name, str):
                load_filename = '%s_net_%s.pth' % (epoch, name)
                load_path = os.path.join(self.load_dir, load_filename)
                net = getattr(self, 'net' + name)
                if isinstance(net, torch.nn.DataParallel):
                    net = net.module
                print('loading the model from %s' % load_path)
                # if you are using PyTorch newer than 0.4 (e.g., built from
                # GitHub source), you can remove str() on self.device
                state_dict = torch.load(load_path, map_location=self.device)
                if hasattr(state_dict, '_metadata'):
                    del state_dict._metadata
                # patch InstanceNorm checkpoints prior to 0.4
                for key in list(state_dict.keys()):  # need to copy keys here because we mutate in loop
                    self.__patch_instance_norm_state_dict(state_dict, net, key.split('.'))

                net.load_state_dict(state_dict)

    def print_networks(self):
        """Print the total number of parameters in the network and (if verbose) network architecture

        Parameters:
            verbose (bool) -- if verbose: print the network architecture
        """
        messages = '---------- Networks initialized -------------\n'
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                messages += str(net) + '\n'
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
                messages += '[Network %s] Total number of parameters : %.4f M\n' % (name, num_params / 1e6)
        print('-----------------------------------------------')
        messages += '-----------------------------------------------\n'
        with open(self.netarc_dir, "a") as f:
            f.write(messages)

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
            
