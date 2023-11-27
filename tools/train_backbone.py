import sys
import time
import numpy as np
sys.path.append('.')
from data import create_dataset
from models import create_model
from util.visualizer import Visualizer
from options.train_options import TrainOptions

if __name__ == "__main__":
    opt = TrainOptions().parse()   # get training options
    dataset, meanstd = create_dataset(opt, 'train', opt.dataroot, False)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)
    print('The number of training images = %d' % dataset_size)
    test_dataset, _ = create_dataset(opt, 'test', opt.dataroot, False)  # create a dataset given opt.dataset_mode and other options
    print('The number of testing images = %d' % len(test_dataset))
    model = create_model(opt, meanstd)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters = 0                # the total number of training iterations/
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        data_start_time = time.time()
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            data_end_time = iter_start_time
            datatime = data_end_time - data_start_time
            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            cal_start_time = time.time()
            model.optimize_parameters()
            cal_end_time = time.time()
            caltime = cal_end_time - cal_start_time
            losses = model.get_current_losses()
            visualizer.print_current_losses(epoch, epoch_iter, total_iters, losses, datatime, caltime)
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix, opt.clear)
            iter_data_time = time.time()
            data_start_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest', opt.clear)
            model.save_networks(epoch, opt.clear)
        if epoch%opt.eval_freq == 0 and opt.eval:
            acc_totalacc = 0
            model.eval()
            start_time = time.time()
            for i, data in enumerate(test_dataset):
                model.set_input(data)
                model.test()
                acc = model.get_current_acc()
                assert acc['totalacc'] == np.sum(acc['perclass'])
                acc_totalacc += acc['totalacc']
            end_time = time.time()
            visualizer.print_current_acc(epoch, ((acc_totalacc / len(test_dataset)) * 100), end_time-start_time)
            model.train()
        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()   # update learning rates at the end of every epoch.