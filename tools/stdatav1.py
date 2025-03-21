import os
import sys
import time
import shutil
sys.path.append('.')
from data import fixstepiter
from data import create_dataset
from models import create_model
import torch.nn.functional as F
from util.visualizer import Visualizer
from util.util import seed_torch, gen_subsetinds
from options.stdatav1_options import STDATAV1Options



if __name__ == "__main__":
    opt = STDATAV1Options().parse()   # get training options
    seed_torch(opt.seed)
    ref_dataset, _ = create_dataset(opt, opt.flag, opt.subroot, True) # build proxy dataset to fix the index of subset
    print('The number of proxy images = %d' % len(ref_dataset))
    inds = gen_subsetinds(len(ref_dataset), opt.sublen)
    ref_dataset.subset.indices = inds
    ref_dataset.saveinds(inds)
    dataset, meanstd = create_dataset(opt, 'test', opt.dataroot, False)
    print('The number of images = %d' % len(dataset))
    model = create_model(opt, meanstd)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    current_path = os.path.abspath(__file__)
    shutil.copy(current_path, os.path.join(os.path.abspath(model.save_dir), 'src', current_path.strip().split('/')[-1]))
    total_iters = 0
    refiters = fixstepiter(ref_dataset, opt.batch_size)
    model.refiter = refiters
    visualizer = Visualizer(opt, model.suffix)
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        for iterind in range(opt.iters):
            cal_start_time = time.time()
            total_iters += opt.costiter
            epoch_iter += opt.costiter
            if total_iters > opt.q:
                break
            model.optimize_parameters()
            cal_end_time = time.time()
            caltime = cal_end_time - cal_start_time
            losses = model.get_current_losses()
            visualizer.print_current_losses(epoch, epoch_iter, total_iters, losses, 'none', caltime)
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest', opt.clear)
            model.save_networks(epoch, opt.clear)
        print(f'Current Run Name:{opt.name}')
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate([0, 1])   # update learning rates at the end of every epoch.
    model.save_networks('latest', opt.clear)