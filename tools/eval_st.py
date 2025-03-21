import os
import sys
import warnings
import torch.nn as nn
sys.path.append('.')
from data import create_dataset
from models import create_model
from util.util import seed_torch
from options.evalst_options import EVALSTOptions
from advertorch.attacks import LinfBasicIterativeAttack
from advertorch.attacks import GradientSignAttack, PGDAttack
warnings.filterwarnings("ignore")

        
def adversary(attack, net, channel, target_flag):
    if attack.upper() == 'PGD':
        if channel == 1:
            adversary = PGDAttack(
                    net,
                    loss_fn=nn.CrossEntropyLoss(),
                    eps=0.3,
                    nb_iter=10, eps_iter=3/40., clip_min=0., clip_max=1.,
                    targeted=target_flag)
        else:
            adversary = PGDAttack(
                    net,
                    loss_fn=nn.CrossEntropyLoss(),
                    eps=8/255.,
                    nb_iter=10, eps_iter=2/255., clip_min=0., clip_max=1.,
                    targeted=target_flag)
    elif attack.upper() == 'BIM':
        if channel == 1:
            adversary = LinfBasicIterativeAttack(
                net,
                loss_fn=nn.CrossEntropyLoss(),
                eps=0.3,
                nb_iter=10, eps_iter=3/40., clip_min=0., clip_max=1.,
                targeted=target_flag)
        else:
            adversary = LinfBasicIterativeAttack(
                net,
                loss_fn=nn.CrossEntropyLoss(),
                eps=8/255., nb_iter=10, eps_iter=2/255., clip_min=0., clip_max=1.,
                targeted=target_flag)       
    # FGSM
    elif attack.upper() == 'FGSM':
        if channel == 1:
            adversary = GradientSignAttack(
                net,
                loss_fn=nn.CrossEntropyLoss(),
                eps=0.3,clip_min=0., clip_max=1.,
                targeted=target_flag)            
        else:
            adversary = GradientSignAttack(
                net,
                loss_fn=nn.CrossEntropyLoss(),
                eps=8/255.,clip_min=0., clip_max=1.,
                targeted=target_flag)
    return adversary

if __name__ == "__main__":
    opt = EVALSTOptions().parse()
    seed_torch(opt.seed)
    dataset, meanstd = create_dataset(opt, 'test', opt.dataroot, False)
    print('The number of images = %d' % len(dataset))
    model = create_model(opt, meanstd)
    model.setup(opt)
    model.eval()
    netsub = nn.Sequential(
            model.norm,
            model.netsub
    )
    netsub.eval()
    message = ''
    if opt.targeted:
        asrflag = 'tasr'
    else:
        asrflag = 'uasr'
    save_root = os.path.join(model.result_path, f'{asrflag}_results.txt')
    for j, attack in enumerate(opt.attacks):
        adver = adversary(attack, netsub, model.norm.mean.shape[1], opt.targeted)
        taracc, subacc, asr = model.eval_asr(adver, dataset)
        if j == 0:
            message += f'Victim ACC:{taracc}, Surrogate ACC:{subacc}\n'
            message += f'{attack}:{asr}\n'
        else:
            message += f'{attack}:{asr}\n'
    with open(f'{save_root}', 'a+') as f:
        f.write(message)