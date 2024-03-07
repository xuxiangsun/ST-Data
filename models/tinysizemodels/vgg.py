'''VGG11/13/16/19 in Pytorch.'''
"""
reference: https://github.com/zhoumingyi/DaST
"""
import torch.nn as nn

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512,
              512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512,
              'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512,
              512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def devidecfg(cfg, rank):
    head = []
    neck = []
    assert rank > 0
    for unit in cfg:
        if rank != 0:
            head += [unit]
            if unit == 'M':
                rank -= 1
            else:
                neckchannels = unit
        else:
            neck += [unit]
    return head, neck, neckchannels

class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, channels):
        super(VGG, self).__init__()
        self.number = 0
        rank = 3
        headcfg, neckcfg, neckinc = devidecfg(cfg[vgg_name], rank)
        self.head = self._make_layers(headcfg, False, channels)
        self.neck = self._make_layers(neckcfg, True, neckinc)
        self.pooling = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )

    def forward(self, x, feat=False, sign=0):
        if sign == 0:
            self.number += 1
        feature = self.head(x)
        out = self.neck(feature)
        out = self.pooling(out)
        emd = out.view(out.size(0), -1)
        out = self.classifier(emd)
        if feat:
            return out, emd
        else:
            return out

    def get_number(self):
        return self.number

    def _make_layers(self, cfg, ifneck=False, in_channels=3):
        layers = []
        count = 0
        for x in cfg:
            if x == 'M':
                count += 1
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        if ifneck:
            layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def vgg11(channels, classes_num):
    return VGG('VGG11', classes_num, channels)

def vgg13(channels, classes_num):
    return VGG('VGG13', classes_num, channels)

def vgg16(channels, classes_num):
    return VGG('VGG16', classes_num, channels)

def vgg19(channels, classes_num):
    return VGG('VGG19', classes_num, channels)