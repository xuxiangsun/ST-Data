import os.path
from torchvision import transforms
from data.base_dataset import BaseDataset
import torchvision
from PIL import Image



class CIFAR10Dataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(load_size=32, crop_size=32, output_nc=3)
        if parser.parse_args().classes == 0:
            parser.set_defaults(classes=10)
        try:
            if not parser.parse_args().cussub:
                parser.set_defaults(submodel='vgg11')
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
        class_set = None
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
        if phs and 'gan' not in opt.model:
            self.transform = transforms.Compose([
                                        transforms.Resize((self.opt.load_size, self.opt.load_size)),
                                        transforms.RandomCrop(self.opt.load_size, padding=4),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.RandomRotation(15),
                                        transforms.ToTensor()
                                        ])
        else:
            self.transform = transforms.Compose([
                                        transforms.Resize((self.opt.load_size, self.opt.load_size), interpolation=Image.BICUBIC),
                                        transforms.ToTensor()
                                        ])      
        self.meanstd = dict(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010], axis=-3)
        self.officialdataset = torchvision.datasets.CIFAR10(root='./datasets/CIFAR10', train=phs, transform=self.transform, download=False)



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
        return len(self.officialdataset.data)
