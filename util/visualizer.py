import os
import time
import numpy as np
import torchvision.utils as tvutils
from tensorboardX import SummaryWriter

class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt, suffix=''):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.name = opt.name
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)
        if suffix == '':            
            self.writer = SummaryWriter(os.path.join(self.opt.result_dir, self.opt.name))
        else:
            self.writer = SummaryWriter(os.path.join(self.opt.result_dir, self.opt.name, suffix))


    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, total_iters, losses, datatime, caltime):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            iters (int) -- total training iterations, used for display losses
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            datatime (float or none) -- data loading time per data point (normalized by batch_size)
            caltime (float) -- computational time per data point (normalized by batch_size)
        """
        if datatime == 'none':
            message = '(epoch: %d, iters: %d, caltime: %.3f) ' % (epoch, iters, caltime)
        else:
            message = '(epoch: %d, iters: %d, datatime: %.3f, caltime: %.3f) ' % (epoch, iters, datatime, caltime)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
            self.writer.add_scalar(k, v, total_iters)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message


    def print_current_acc(self, epoch, acc, time):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            acc (float) -- current accuracy
            time (float) -- calculation time
        """
        message = '(epoch: %d) ' % (epoch)

        message += 'Totalacc: %.3f%% ' % (acc)

        message += '(calculation time:%.4f) ' % (time)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message