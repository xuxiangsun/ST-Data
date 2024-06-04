import os.path
import numpy as np
from PIL import Image
from torchvision import transforms
from data.base_dataset import BaseDataset

class PLACES365Dataset(BaseDataset):
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
        self.output_nc = opt.output_nc
        self.y_class = {}
        self.data_name = self.datasets_dir.split('/')[-1]
        with open(os.path.join('./datasets_dict', '{}_dict.txt'.format(self.data_name)), "r") as f:
            lines = f.readlines()
            for i in lines:
                classes = i.strip().split(":")[0]
                idx = i.strip().split(":")[1]
                self.y_class[classes] = int(idx)

        self.path = os.path.join(self.datasets_dir, 'places365_standard/val')
        self.data = []
        self.label = []
        for label in os.listdir(self.path):
            id = self.y_class[label]
            image_dir = os.path.join(self.path, label)
            for ima in os.listdir(image_dir):
                im = os.path.join(image_dir, ima)
                self.data.append(im)
                self.label.append(id)
        self.label = np.asarray(self.label)
        if self.flag == 'train':
            phs = True
        else:
            phs = False
        if phs and 'gan' not in opt.model:
            self.transform = transforms.Compose([
                                        transforms.Resize((32, 32)),
                                        transforms.RandomCrop(32, padding=4),
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

        img, label = self.data[index], self.label[index]
        sample = Image.open(img)
        sample = sample.convert('RGB')
        imgs = self.transform(sample)

        return imgs, label

    def __len__(self):
        return len(self.data)