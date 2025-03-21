from .base_options import BaseOptions


class EVALSTOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='eval', help='specify the training/testing type you input')
        # base parameters
        # parser.add_argument('--netG', type=str, default='DCGAN', help='generator')
        # parser.add_argument('--zdim', type=int, default=100, help='dimension of the input noise')
        parser.add_argument('--attacks', default=['BIM', 'PGD', 'FGSM'], nargs='+', type=str, help='pgd, fgsm, cw')
        parser.add_argument('--modeltype', type=str, default='backbone', help='chooses which model to use. Inalterable')
        parser.add_argument('--cussub', action='store_true', default=False, help='if specified, input the substutite model you specified')
        parser.add_argument('--submodel', type=str, default='vgg13', help='choice of surrogate model')
        parser.add_argument('--tarmodel', type=str, default='resnet18', help='choice of target model')
        parser.add_argument('--targeted', action='store_false', default=True, help='if specified, calculate the targeted attack success rate')
        # testing parameters
        parser.add_argument('--para',type=str,default='none')
        self.isTrain = False
        return parser
