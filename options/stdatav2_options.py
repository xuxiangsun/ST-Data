from .base_options import BaseOptions


class STDATAV2Options(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--phase', type=str, default='stdatav2', help='specify the method')
        # base parameters
        parser.add_argument('--gap', type=int, default=40, help='update gap of stdatav2')
        parser.add_argument('--q_epoch', type=float, default=10000., help='query budget in Million')
        parser.add_argument('--subroot', type=str, default='./datasets/places365', help='path to proxy dataset')
        parser.add_argument('--modeltype', type=str, default='backbone', help='chooses which model to use. Inalterable')
        parser.add_argument('--cussub', action='store_true', default=False, help='if specified, input the substutite model you specified')
        parser.add_argument('--submodel', type=str, default='vgg13', help='choice of surrogate model')
        parser.add_argument('--tarmodel', type=str, default='resnet18', help='choice of target model')
        parser.add_argument('--lr_s', type=float, default=1e-2, help='initial learning rate for the surrogate')
        parser.add_argument('--lr_gan', type=float, default=2e-4, help='initial learning rate for the generator&discriminator')
        parser.add_argument('--netG', type=str, default='SelfCondDCGAN', help='choice of generator')
        parser.add_argument('--netD', type=str, default='DCGAN', help='choice of discriminator')
        parser.add_argument('--zdim', type=int, default=100, help='dimension of the input noise')
        parser.add_argument('--optim', type=str, default='adamv1', help='choice of the optimizer for the generator/discriminator')
        parser.add_argument('--optim2', type=str, default='sgd', help='choice of the optimizer for the surrogate')
        parser.add_argument('--sublen', type=int, default=4000, help='size of the proxy data')
        parser.add_argument('--labonly', action='store_true', help='if specified, only the decision label of the victim can be accessed')
        parser.add_argument('--nologits', action='store_true', help='if specified, only the probability of the victim can be accessed')
        parser.add_argument('--logit_corr', type=str, default='mean', choices=['none', 'mean'], help='which restoration is applied')
        parser.add_argument('--flag', type=str, default='val', help='choice of train/val/test of a dataset')
        parser.add_argument('--loss', type=str, default='vanilla', help='choice of optimized objective functions')
        parser.add_argument('--probloss', type=str, default='L1Loss', help='choice of optimized objective functions')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        # training parameters
        parser.add_argument('--niter', type=int, default=20, help='# of iter at starting learning rate')
        parser.add_argument('--niter_decay', type=int, default=0, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        # self.issubtrain = True
        self.isTrain = True
        # self.issubtest = False
        return parser
