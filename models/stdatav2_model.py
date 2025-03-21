import os
import torch
import torch.nn as nn
from .base_model import BaseModel
from . import model_utils as mutils
from . import tinysizemodels as tvmodels

class STDATAV2Model(BaseModel):
    """ 
    This class implements the PSGAN model.

    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser = STDATAV2Model.reset_epoch_iter(parser, is_train)
        return parser
    
    @staticmethod
    def reset_epoch_iter(parser, train):
        if train:
            costiter = parser.parse_args().batch_size
            parser.add_argument('--q', type=int, help='total iters per epoch')
            parser.add_argument('--iters', type=int, help='total iters per epoch')
            parser.add_argument('--costiter', type=float, help='query cost per iteration')
            total_q = parser.parse_args().q_epoch * \
                (parser.parse_args().niter + parser.parse_args().niter_decay)
            total_q -= parser.parse_args().sublen
            q_epoch = int(total_q // (parser.parse_args().niter+parser.parse_args().niter_decay)) 
            iters = int(q_epoch //costiter)
            parser.set_defaults(iters=iters, costiter=costiter, q=total_q, norm='batch')
        return parser
    
    def __init__(self, opt, meanstd):
        """Initialize the class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt, meanstd)
        self.opt = opt
        self.fix = None
        self.norm = mutils.Normalize(meanstd).cuda()
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_mce', 'G_GAN', 'D_fake', 'D_real', 'sub_c', 'sub_p']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['data']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['sub', 'G', 'D']
        else:  # during test time, only load C
            self.model_names = ['sub']
        self.target = mutils.init_net(tvmodels.__dict__[opt.tarmodel](opt.output_nc, opt.classes))
        self.netsub = mutils.init_net(tvmodels.__dict__[opt.submodel](opt.output_nc, opt.classes))
        tcheck = os.path.join(opt.checkpoints_dir, 'backbone', \
            '{}_{}/latest_net_{}.pth'.format(opt.dataroot.split('/')[-1], opt.tarmodel, opt.modeltype))
        self.target.load_state_dict(torch.load(tcheck))
        self.target.eval()
        self.suffix = 'STDatav2'
        if self.isTrain:
            self.fixiter = None
            self.refiter = None
            self.netG = mutils.define_G(args=opt)
            self.netD = mutils.define_D(args=opt)
            self.ce = nn.CrossEntropyLoss().cuda()
            self.D_GAN = mutils.D_GANLoss(opt.loss).cuda()
            self.G_GAN = mutils.G_GANLoss(opt.loss).cuda()
            self.criterion = mutils.GeneralCrite(labflag=self.opt.labonly, probcrite=self.opt.probloss).cuda()
            self.optimizer_G = mutils.define_optim(self.netG, self.opt.optim, self.opt.lr_gan, self.opt.optimdicts['optim'])
            self.optimizer_D = mutils.define_optim(self.netD, self.opt.optim, self.opt.lr_gan, self.opt.optimdicts['optim'])
            self.optimizer_sub = mutils.define_optim(self.netsub, self.opt.optim2, self.opt.lr_s, self.opt.optimdicts['optim2'])
            self.g_norm = mutils.Normalize(mutils.get_act_norm(getattr(opt, 'netGdicts')['netG']['final_act'], len(self.meanstd['mean']))).cuda()
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_sub)

        
    def gen(self, Gflag):
        self.optimizer_G.zero_grad()
        self.set_requires_grad(self.netG, Gflag)
        noise = torch.randn((self.opt.batch_size, self.opt.zdim))
        self.data, self.set_labels, self.sub_labels = self.netG(noise.cuda(), clsnum=self.opt.classes)
                
    def set_input(self, real=None):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.s
        """
        if real is not None:
            # del self.real
            self.real = real.cuda()
        else:
            self.real = None
            
    def query(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            with torch.no_grad():
                t_logits = self.target(self.norm(self.query_data))
                self.t_logits = self.restore_logits(t_logits)
                _, self.label = torch.max(t_logits, 1)
        
    def backward_sub(self):
        s_logits = self.netsub(self.norm(self.query_data))
        self.loss_sub_c, self.loss_sub_p, self.loss_sub_syn = self.criterion(s_logits, self.label, self.t_logits)
        s_outputs = self.netsub(self.norm(self.observer))
        t_outputs = self.target(self.norm(self.observer))
        _, t_preds = torch.max(t_outputs, 1)
        _, _, self.loss_sub_obs = self.criterion(s_outputs, t_preds.detach(), self.restore_logits(t_outputs.detach()))
        self.loss_sub = self.loss_sub_syn + self.loss_sub_obs
        self.loss_sub.backward()
        self.optimizer_sub.step()          # update sub's weights

    def backward_D(self):
        self.loss_D_real, self.loss_D_fake, _ = self.D_GAN(self.netD, \
            [self.g_norm.inverse(self.real.detach()), self.data.detach()], [True, False])  # type: ignore
        self.loss_D = self.loss_D_fake + self.loss_D_real
        self.loss_D.backward()
        self.optimizer_D.step()

    def backward_G(self):
        output = self.netsub(self.norm(self.g_norm(self.data)))
        self.loss_G_mce = self.ce(output, self.set_labels) + self.ce(output, self.sub_labels)
        fake_logits = self.netD(self.data)
        self.loss_G_GAN = self.G_GAN(fake_logits, True)
        self.loss_G = self.loss_G_GAN + 0.5 * self.loss_G_mce
        self.loss_G.backward()
        self.optimizer_G.step()

    def train_GAN(self):
        self.netsub.eval()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.set_requires_grad(self.netsub, False)  # sub requires no gradients when optimizing G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.backward_G()                   # calculate graidents for G
        self.netsub.train()
        
    def train_S(self):
        self.query_data = self.g_norm(self.data.detach())
        self.query()
        self.netsub.train()
        self.set_requires_grad(self.netsub, True)
        self.optimizer_sub.zero_grad()     # set sub's gradients to zero
        self.backward_sub()                # calculate gradients for sub    

    def optimize_parameters(self):
        imgs, _ = next(self.refiter)
        self.set_input(imgs)
        self.gen(True)
        self.train_S()
        self.train_GAN()
    
    def update_fix(self):
        imgs, _ = next(self.fixiter)
        self.observer = imgs.detach().cuda()