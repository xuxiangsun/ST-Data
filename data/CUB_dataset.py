import os.path
import pandas as pd
from PIL import Image
from torchvision import transforms
from data.base_dataset import BaseDataset


class CUBDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=32, crop_size=32, output_nc=3)
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
        self.root_dir = os.path.join(self.datasets_dir, 'CUB_200_2011')
        self.img_dir = os.path.join(self.root_dir, 'images')
        self.y_class = {}
        self.data_name = self.datasets_dir.split('/')[-1]

        with open(os.path.join('./datasets_dict', '{}_dict.txt'.format(self.data_name)), "r") as f:
            lines = f.readlines()
            for i in lines:
                classes = i.strip().split(":")[0]
                idx = i.strip().split(":")[1]
                self.y_class[classes] = int(idx)
        self.load_metadata()

        if self.flag == 'train' and 'gan' not in opt.model:
            self.transform = transforms.Compose([
                                        transforms.Resize((32, 32)),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor()
                                        ])
        else:
            self.transform = transforms.Compose([
                                        transforms.Resize((32, 32), interpolation=Image.BICUBIC),
                                        transforms.ToTensor()
                                        ])
        self.meanstd = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], axis=-3)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """

        img_path = self.data[self.data.index == self.data.index[index]].filepath.values[0]
        sample = Image.open(os.path.join(self.img_dir, img_path))
        sample = sample.convert('RGB')
        if self.transform is not None:
            imgs = self.transform(sample)
        label = int(self.data[self.data.index == self.data.index[index]].target.values[0])
        return imgs, label

    def __len__(self): 
        return len(self.data)


    def load_metadata(self):
        images = pd.read_csv(os.path.join(self.root_dir, 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root_dir, 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root_dir, 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.flag == 'train':
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]