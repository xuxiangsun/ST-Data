"""This package includes all the modules related to data loading and preprocessing

 To add a custom dataset class called 'dummy', you need to add a file called 'dummy_dataset.py' and define a subclass 'DummyDataset' inherited from BaseDataset.
 You need to implement four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point from data loader.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.

Now you can use the dataset class by specifying flag '--dataset_mode dummy'.
See our template dataset class 'template_dataset.py' for more details.
"""
import torch
import pickle
import importlib
import itertools
import numpy as np
from torch.utils.data import random_split
from data.base_dataset import BaseDataset
    
    
class fixstepiter(object):
    """
    Customized iter, which load data with a fixed step.
    """
    def __init__(self, basicloader, batch):
        """
        Args:
            basiciter (Iterator): A basic dataloader for torch.utils.data.DataLoader,
            i.e., itertools.cycle(dataloader). Note that iter(dataloader) is not supported.
        """
        self.basicloader = basicloader
        self.basiciter = itertools.cycle(basicloader)
        self.batch = batch
        
    def __iter__(self):
        return self
    
    def __next__(self):
        data, labs = next(self.basiciter)
        if data.shape[0]<self.batch:
            del self.basiciter
            self.basiciter = itertools.cycle(self.basicloader)
            newdata, newlabs = next(self.basiciter)
            remain = self.batch - data.shape[0]
            rand_inds = np.random.choice(np.arange(self.batch), remain, replace=False)
            data = torch.cat([data, newdata[rand_inds]], 0)
            labs = torch.cat([labs, newlabs[rand_inds]], 0)
        return data, labs

def find_dataset_using_name(dataset_name):
    """Import the module "data/[dataset_name]_dataset.py".

    In the file, the class called DatasetNameDataset() will
    be instantiated. It has to be a subclass of BaseDataset,
    and it is case-insensitive.
    """
    dataset_filename = "data." + dataset_name + "_dataset"
    datasetlib = importlib.import_module(dataset_filename)

    dataset = None
    target_dataset_name = dataset_name.replace('_', '') + 'dataset'
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
           and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise NotImplementedError("In %s.py, there should be a subclass of BaseDataset with class name that matches %s in lowercase." % (dataset_filename, target_dataset_name))

    return dataset


def get_option_setter(dataset_name):
    """Return the static method <modify_commandline_options> of the dataset class."""
    dataset_class = find_dataset_using_name(dataset_name)
    return dataset_class.modify_commandline_options


def create_dataset(opt, flag, root, split):
    """Create a dataset given the option.

    This function wraps the class CustomDatasetDataLoader.
        This is the main interface between this package and 'train.py'/'test.py'

    Arg "ifsub" is used to define which dataset (main/proxy) is created.
    Example:
        >>> from data import create_dataset
        >>> dataset = create_dataset(opt)
    """
    data_loader = CustomDatasetDataLoader(opt, flag, root, split)
    dataset = data_loader.load_data()
    return dataset, data_loader.get_meanstd()


class CustomDatasetDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt, flag, root, split):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        data_class = root.split('/')[-1]
        dataset_class = find_dataset_using_name(data_class)
        self.dataset = dataset_class(opt, flag, root)
        self.meanstd = self.dataset.meanstd
        self.dataname = data_class
        if split:
            self.subset = random_split(self.dataset, [opt.sublen, len(self.dataset)-opt.sublen])\
                [1 if opt.sublen==0 else 0]
            assert len(set(self.subset.indices)) == len(self.subset)
            self.dataloader = torch.utils.data.DataLoader(
                self.subset,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=int(opt.num_threads))
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batch_size,
                shuffle=True,
                num_workers=int(opt.num_threads))  
        print("dataset [%s] was created" % type(self.dataset).__name__)

    def load_data(self):
        return self
    
    def get_meanstd(self):
        return self.meanstd
    
    def saveinds(self, indices):
        with open(f'{self.opt.checkpoints_dir}/{self.opt.name}/{self.dataname}_subsetinds.pkl', 'wb') as f:
            pickle.dump(indices, f)
            
    def __len__(self):
        """Return the number of data in the dataset"""
        return len(self.dataset) if not hasattr(self, 'subset') else len(self.subset)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            yield data
