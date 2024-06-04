import torch
import os.path
import torchvision
from PIL import Image
import numpy as np
from torchvision import transforms
from data.base_dataset import BaseDataset

class_cands={
    '10':['road', 'cloud', 'forest', 'mountain', 'plain', \
        'sea','castle', 'keyboard', 'skyscraper','bridge'],\
    '20':['plate', 'rose', 'castle', 'keyboard', 'house',\
        'forest', 'road','television', 'bottle', 'wardrobe',\
        'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',\
        'bed', 'chair', 'couch', 'table', 'wardrobe'],
    '40':['orchid', 'poppy', 'rose', 'sunflower', 'tulip',
            'bottle', 'bowl', 'can', 'cup', 'plate',
            'apple', 'mushroom', 'orange', 'pear', 'sweet_pepper',
            'clock', 'keyboard', 'lamp', 'telephone', 'television',
            'bed', 'chair', 'couch', 'table', 'wardrobe',
            'maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree',
            'bridge', 'castle', 'house', 'road', 'skyscraper',
            'cloud', 'forest', 'mountain', 'plain', 'sea']}


def filter_indices(trainset, classes_indices):
    index_list = [list(np.argwhere(np.array(trainset.targets)==i).squeeze(-1)) for i in classes_indices]
    return sum(index_list, [])

class CIFAR100Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=32, crop_size=32, output_nc=3)
        if parser.parse_args().classes == 0:
            parser.set_defaults(classes=100)
        try:
            if not parser.parse_args().cussub:
                parser.set_defaults(submodel='resnet18')
        except:
            pass
        return parser
    
    def __init__(self, opt, flag, root):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.flag = flag
        self.opt = opt
        self.datasets_dir = root
        self.output_nc = opt.output_nc
        self.y_class = {}
        self.data_name = self.datasets_dir.split('/')[-1]
        with open(os.path.join('./datasets_dict', '{}_dict.txt'.format(self.data_name)), "r") as f:
            lines = f.readlines()
            for i in lines:
                classes = i.strip().split(":")[0]
                idx = i.strip().split(":")[1]
                self.y_class[classes] = int(idx)
        if self.flag == 'train':
            phs = True
        else:
            phs = False
        if phs and 'backbone' in opt.model:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor()
                ])
        else:
            self.transform = transforms.Compose([
                                        transforms.Resize((32, 32), interpolation=Image.BICUBIC),
                                        transforms.ToTensor()
                                        ])

        self.meanstd = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
        officialdataset = torchvision.datasets.CIFAR100(root='./datasets/CIFAR100', train=phs, transform=self.transform, download=False)
        self.officialdataset = officialdataset


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        imgs, label = self.officialdataset.__getitem__(index)[0], self.officialdataset.__getitem__(index)[1]


        return imgs, label

    def __len__(self):
        return len(self.officialdataset)
