import torch
import random
import functools
import numpy as np
import torch.nn as nn

"""
https://github.com/EnchanterXiao/video-style-transfer/blob/master/model/SANet.py
"""
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

class AdaAttN(nn.Module):
    
    def __init__(self, in_planes, allout=True):
        super(AdaAttN, self).__init__()
        self.f = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.g = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.h = nn.Conv2d(in_planes, in_planes, (1, 1))
        self.sm = nn.Softmax(dim=-1)
        self.act = nn.ReLU()
        self.allout = allout
    def forward(self, content, style):
        F = self.f(mean_variance_norm(content))
        G = self.g(mean_variance_norm(style))
        H = self.h(style)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        S = self.sm(S)
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        H2 = H * H
        mean = torch.bmm(H, S.permute(0, 2, 1))
        M2 = torch.bmm(H2, S.permute(0, 2, 1))
        std = torch.sqrt(self.act(M2 - mean*mean)+1e-8)
        b, c, h, w = content.size()
        O = mean_variance_norm(content)
        b, c, h, w = content.size()
        out = O * std.view(b,c,h,w) + mean.view(b,c,h,w)
        if self.allout:
            return out, std, mean
        else:
            return out

class CACWNG(nn.Module):
    def __init__(self, z_dim, nzf, output_nc, img_size, n_classes, norm_layer=nn.BatchNorm2d,\
        final_act=nn.Sigmoid(), clsbatch=10, **extraparas):
        super(CACWNG, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if type(final_act) == functools.partial:
            actfunc = final_act.func
        else:
            actfunc = final_act
        self.cla = n_classes
        self.clsbatch = clsbatch
        assert self.clsbatch <= self.cla
        self.act = nn.ReLU(inplace=True)
        self.headlab = nn.Sequential(
            nn.Embedding(n_classes, 4*nzf),nn.Dropout(0.5),
            nn.Linear(4*nzf, 8*nzf),nn.Dropout(0.5)
        )
        self.necklab = nn.Sequential(
            nn.ConvTranspose2d(8*nzf, 8*nzf,
                                                kernel_size=4, stride=1,
                                                padding=0, bias=use_bias),
                            norm_layer(8*nzf),
                            self.act#,
        )
        self.headz = nn.Sequential(
            nn.Linear(z_dim, 4*nzf),
            nn.Linear(4*nzf, 8*nzf)
        )
        self.neckz = nn.Sequential(
            nn.ConvTranspose2d(8*nzf, 8*nzf,
                                                kernel_size=4, stride=1,
                                                padding=0, bias=use_bias),
                            norm_layer(8*nzf),
                            self.act#,
        )
        self.featsize = 4
        self.deconv = nn.ModuleList()
        numup = int(np.log2(img_size)-1)
        mult = int(nzf*8)
        self.salist = AdaAttN(mult)
        for i in range(numup):
            if i == numup-1:
                self.deconv.append(nn.Sequential(
                    nn.ConvTranspose2d(int(mult), output_nc, kernel_size=1,
                                       stride=1, padding=0),
                    actfunc()
                ))
            elif i == 0:
                    self.deconv.append(nn.Sequential(
                        nn.ConvTranspose2d(mult * 2, int(mult / 2),
                                                kernel_size=4, stride=2,
                                                padding=1, bias=use_bias),
                            norm_layer(int(mult / 2)),
                            self.act
                            ))
            else:
                self.deconv.append(nn.Sequential(
                    nn.ConvTranspose2d(mult, int(mult / 2),
                                            kernel_size=4, stride=2,
                                            padding=1, bias=use_bias),
                        norm_layer(int(mult / 2)),
                        self.act
                        ))
            mult = int(mult/2)

    def forward(self, noises):
        batch = noises.shape[0]
        labels = torch.tensor(random.sample(range(self.cla),\
            self.clsbatch)).to(torch.long).cuda()
        label_feats = self.necklab(self.headlab(labels).\
            view(self.clsbatch, -1, 1, 1)).unsqueeze(1).repeat(1, batch, 1, 1, 1)
        noise_feats = self.neckz(self.headz(noises).view(batch, -1, 1, 1)).\
            unsqueeze(0).repeat(self.clsbatch, 1, 1, 1, 1)
        render_feat, _, _ = self.salist(label_feats.\
            view(self.clsbatch*batch, -1, self.featsize, self.featsize),\
            noise_feats.view(self.clsbatch*batch, -1, self.featsize, self.featsize))
        input = torch.cat([noise_feats.view(self.clsbatch*batch, -1,\
            self.featsize, self.featsize), render_feat], 1)
        for layer in self.deconv: 
            out = layer(input)
            input = out
        output = out
        return output, labels.unsqueeze(-1).repeat(1, batch).flatten()


class SelfCondDCGAN(nn.Module):
    def __init__(self, z_dim=100, output_nc=3, ngf=64, final_act=nn.Tanh(), **extraparas):
        super(SelfCondDCGAN, self).__init__()
        if type(final_act) == functools.partial:
            actfunc = final_act.func
        else:
            actfunc = final_act
        self.zdim = z_dim
        self.head = nn.Sequential(
            nn.ConvTranspose2d(z_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4)
        )
        self.main = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, output_nc, kernel_size=1, stride=1, padding=0, bias=False)
        )
        self.final_act = actfunc()
        
    def forward(self, input, clsnum):
        if len(input.size()) == 2:
            noise = input.unsqueeze(-1).unsqueeze(-1).cuda()
            midfeat = self.head(noise)
            assert self.zdim % clsnum ==0 and self.zdim >= clsnum
            group_noise = input.view(-1, clsnum, int(self.zdim//clsnum)).mean(-1)\
                        if int(self.zdim/clsnum) > 1 else\
                            noise.view(-1, clsnum)
            label_ranks = torch.sort(group_noise, -1, descending=True)[1]
            dlabels = label_ranks[:, 0].cuda()
            slabels = label_ranks[:, 1].cuda()
        elif len(input.size()) == 4:
            midfeat = input
        pre_act = self.main(midfeat)
        img = self.final_act(pre_act)
        return img, dlabels, slabels
    