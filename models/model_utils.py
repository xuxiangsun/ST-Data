import torch
import functools
import numpy as np
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from . import generators as gen_dicts
from . import discriminators as dis_dicts


class Normalize(nn.Module):
    def __init__(self, meanstd):
        super(Normalize, self).__init__()
        mean = meanstd['mean']
        std = meanstd['std']
        self.mean = torch.Tensor(mean).reshape(1, len(mean), 1, 1).cuda()
        self.std = torch.Tensor(std).reshape(1, len(mean), 1, 1).cuda()
    def forward(self, input):
        size = input.size()
        outputs = torch.div(torch.sub(input, self.mean.expand(size)), self.std.expand(size))
        return outputs
    def inverse(self, input):
        size = input.size()
        outputs = torch.add(torch.mul(input, self.std.expand(size)), self.mean.expand(size))
        return outputs
    
class Identity(nn.Module):
    def forward(self, x):
        return x

def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_act_layer(act_type='tanh'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if act_type == 'tanh':
        norm_layer = functools.partial(nn.Tanh, {})
    elif act_type == 'sigmoid':
        norm_layer = functools.partial(nn.Sigmoid, {})
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return norm_layer

def get_act_norm(act_type='tanh', channels=3):
    if act_type == 'tanh':  
        meanstd=dict(mean=[-1]*channels, std=[2]*channels, axis=-3)
    elif act_type == 'sigmoid':
        meanstd=dict(mean=[0]*channels, std=[1]*channels, axis=-3)
    else:
        raise NotImplementedError('activation layer [%s] is not found' % act_type)
    return meanstd
        

def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions.
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'multistep':
        scheduler = lr_scheduler.MultiStepLR(optimizer, sorted([int(step * opt.niter) for step in opt.lr_steps_ratio]), opt.lr_decay_scale)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            if hasattr(m, 'weight') and m.weight is not None:
                init.normal_(m.weight.data, 1.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    
    init_weights(net, init_type, init_gain=init_gain)
    net.cuda()
    return net

def restore_paras(source_dicts):
    dicts = {}
    for k, v in source_dicts.items():
        if 'norm' in k:
            dicts = {**dicts, **{k:get_norm_layer(v)}}
        elif 'act' in k:
            dicts = {**dicts, **{k:get_act_layer(v)}}
        else:
            dicts = {**dicts, **{k:v}}
    return dicts

def define_G(args, gpu_ids=[]):
    """Create a generator
    Parameters:
        args -- parameters
        args.init_type (str)    -- the name of our initialization method.
        args.init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a generator

    The generator has been initialized by <init_net>.
    """
    nets = []
    baseparas = dict(z_dim=args.zdim, output_nc=args.output_nc, img_size=args.load_size, n_classes=args.classes, norm_layer=get_norm_layer(args.norm))
    for g_id in args.netGdicts.keys():
        extraparas = restore_paras(args.netGdicts[g_id])
        fullparas = {**baseparas, **extraparas}
        model = getattr(args, g_id) if '_' not in getattr(args, g_id) else getattr(args, g_id).strip().split('_')[0]
        net = gen_dicts.__dict__[model](**fullparas)
        nets.append(init_net(net, args.init_type, args.init_gain, gpu_ids))
    if len(args.netGdicts.keys()) == 1:
        return nets[0]
    else:
        return nets

def define_D(args, gpu_ids=[]):
    """Create a discriminator
    Parameters:
        args -- parameters
        args.init_type (str)    -- the name of our initialization method.
        args.init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Returns a discriminator
    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    nets = []
    baseparas = dict(input_nc=args.output_nc, norm_layer=get_norm_layer(args.norm))
    for d_id in args.netDdicts.keys():
        extraparas = restore_paras(args.netDdicts[d_id])
        fullparas = {**baseparas, **extraparas}
        model = getattr(args, d_id) if '_' not in getattr(args, d_id) else getattr(args, d_id).strip().split('_')[0]
        net = dis_dicts.__dict__[model](**fullparas)
        nets.append(init_net(net, args.init_type, args.init_gain, gpu_ids))
    if len(args.netDdicts.keys()) == 1:
        return nets[0]
    else:
        return nets

def define_optim(net, opti, lr, optizer_args):
    if 'adam' in opti:
        optimizer = optim.Adam(net.parameters(), lr=lr, betas=(optizer_args['beta1'], optizer_args['beta2']),\
                 weight_decay=optizer_args['weight_decay'])
    elif 'sgd' in opti:
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=optizer_args['momentum'],\
                              weight_decay=optizer_args['weight_decay'])  
    elif 'rmsprop' in opti:
        optimizer = optim.RMSprop(net.parameters(), lr=lr, alpha=optizer_args['alpha'], \
            weight_decay=optizer_args['weight_decay'], momentum=optizer_args['momentum'])
    return optimizer

##############################################################################
# losses
##############################################################################

class GeneralCrite(nn.Module):
    def __init__(self, labflag, reduction='mean', probcrite='MSELoss'):
        """Attention:
           The value of probcrite must be in the dict of torch.nn
           For instance,  L1Loss, MSELoss, KLDivLoss, etc.
           We now only support L1Loss and MSELoss. For KLDivLoss, we need
           a preprocess for the logits.
        """
        super(GeneralCrite, self).__init__()
        self.labflag = labflag
        self.lab = nn.CrossEntropyLoss(reduction=reduction)
        self.prob = nn.__dict__[probcrite](reduction=reduction)
        
    def __call__(self, logits, targets, target_logits):
        self.lab_loss = self.lab(logits, targets.detach())
        self.prob_loss = self.prob(logits, target_logits.detach()) if not self.labflag\
            else self.prob(logits.detach(), target_logits.detach())
        probcof = 1 if not self.labflag else 0
        self.loss = (1 - probcof) * self.lab_loss + probcof * self.prob_loss
        return self.lab_loss.detach(), self.prob_loss.detach(), self.loss

class BoundaryLoss(nn.Module):
    def __init__(self, class_nums):
        super(BoundaryLoss, self).__init__()
        self.classes = class_nums
    def __call__(self, prediction, orilabs, tarlabs=None):
        one_hot = F.one_hot(orilabs, num_classes=self.classes)
        origpreds = torch.sum(one_hot*prediction, 1)
        if tarlabs is None:
            nonlab_logits = (1 - one_hot) * prediction
            series_out = torch.sort(nonlab_logits, descending=True)[0]
            tarpreds = series_out[:, 0]
        else:
            tar_one_hot = F.one_hot(tarlabs, num_classes=self.classes)
            tarpreds = torch.sum(tar_one_hot*prediction, 1)
        # loss = torch.abs(torch.mean(origpreds - tarpreds))
        bdvalue = torch.mean(origpreds - tarpreds)
        return nn.L1Loss()(bdvalue, torch.zeros_like(bdvalue).cuda())
    
    
class ClassSTDDivLoss(nn.Module):
    def __init__(self, class_nums, clsbatch=10):
        super(ClassSTDDivLoss, self).__init__()
        self.classes = class_nums
        self.clsbatch = 10
        
    def clsstd(self, inputs):
        probs = F.softmax(inputs, dim=2)#.sum(1)
        frac = probs.max(2)[0]-probs.min(2)[0]
        frac[frac==0]=1
        norm_probs = (probs-probs.min(2)[0].unsqueeze(-1))/frac.unsqueeze(-1)
        loss = torch.std(norm_probs.sum(1), dim=1)
        return loss
                
    def groupstd(self, inputs):
        unit_indexs = [np.delete(np.arange(0, self.classes), k) for k in range(self.classes)]
        indexs = torch.tensor(np.array(unit_indexs)).to(torch.long).view(self.classes, 1, -1).repeat(1, inputs.shape[1], 1).cuda()
        logits = inputs.gather(-1, indexs)
        loss = self.clsstd(logits)
        return loss
    
    def remainstd(self, inputs, lab_cands):
        unit_indexs = [np.delete(np.arange(0, self.classes), k) for k in lab_cands]
        indexs = torch.tensor(np.array(unit_indexs)).to(torch.long).view(self.clsbatch, 1, self.classes-1).repeat(1, inputs.shape[1], 1).cuda()
        remains = inputs.detach().gather(-1, indexs)
        labind = torch.sort((remains-remains.min(2)[0].unsqueeze(-1)).sum(1), descending=True, dim=1)[1][:, :inputs.shape[1]]
        vallabs = labind.view(self.clsbatch, 1, -1).repeat(1, inputs.shape[1], 1).cuda()
        logits = remains.gather(-1, vallabs)
        loss = self.clsstd(logits)
        return loss
    
    def __call__(self, logits, labels):
        # logits: output logits of a model, not the probability after softmax
        # divide samples regarding the dominate labels
        label_cands = set(labels.cpu().numpy())
        clas_samples = torch.cat([logits[labels == unit, :].unsqueeze(dim=0) for unit in label_cands], dim=0)
        group_num = int(clas_samples.shape[1]//(clas_samples.shape[0]-1))
        remain_series = int(clas_samples.shape[1]%(clas_samples.shape[0]-1))
        try:
            loss_lists = torch.zeros((clas_samples.shape[0], group_num+np.sign(remain_series))).cuda()
            for group_ind in range(group_num):
                group_samples = clas_samples[:, group_ind*(clas_samples.shape[0]-1):(group_ind+1)*(clas_samples.shape[0]-1), :]
                loss_lists[:, group_ind] = self.groupstd(group_samples)
            if remain_series != 0:
                loss_lists[:, group_ind+1] = self.remainstd(clas_samples[:, (group_ind+1)*(clas_samples.shape[0]-1):, :])
            loss = nn.L1Loss()(loss_lists.mean(1), torch.zeros(loss_lists.shape[0]).cuda())
        except:
            outputs = self.remainstd(clas_samples, label_cands)
            loss = nn.L1Loss()(outputs, torch.zeros_like(outputs).cuda())
        return loss
    
    
class G_GANLoss(nn.Module):
    
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(G_GANLoss, self).__init__()
        self.gan_mode = gan_mode
        if gan_mode in ['lsgan', 'vanilla', 'wgangp', 'wgan']:
            self.loss = GANLoss(gan_mode, target_real_label, target_fake_label)
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
            
    def __call__(self, inputs, targets):
        if self.gan_mode in ['lsgan', 'vanilla', 'wgangp', 'wgan']:
            return self.loss(inputs, targets)
        
class D_GANLoss(nn.Module):
    
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(D_GANLoss, self).__init__()
        self.gan_mode = gan_mode
        if gan_mode in ['lsgan', 'vanilla']:
            self.loss = GANLoss(gan_mode, target_real_label, target_fake_label)
        elif 'wgan' in gan_mode:
            self.loss = None
            self.weight_cliping_limit = 0.01
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)
    def __call__(self, net, inputs, targets):
        """
        Args:
            inputs (tuple): [real_imgs, fake_imgs]
            net (nn.Module): discriminators
            targets (tensor, bool): [True, False]
        Returns:
            Tensor: return loss
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            return self.loss(net(inputs[0]), targets[0]), self.loss(net(inputs[1]), targets[1]), 0#cal_gradient_penalty(net, inputs[0], inputs[1].detach())
        elif 'wgan' in self.gan_mode:
            if self.gan_mode == 'wgangp':
                return -net(inputs[0]).mean(), net(inputs[1]).mean(), cal_gradient_penalty(net, inputs[0], inputs[1])
            elif self.gan_mode == 'wgan':
                # weight clipping
                for p in net.parameters():
                    p.data.clamp_(-self.weight_cliping_limit, self.weight_cliping_limit)
                return -net(inputs[0]).mean(), net(inputs[1]).mean(), 0#cal_gradient_penalty(net, inputs[0], inputs[1].detach())

    
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif 'wgan' in gan_mode:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real).cuda()
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode in ['wgangp', 'wgan']:
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( | |gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1).cuda()
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty
    else:
        return 0.0