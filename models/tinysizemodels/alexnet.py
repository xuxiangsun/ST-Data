import torch.nn as nn


"""
Code Reference:https://www.cnblogs.com/zhengbiqing/p/10425503.html
"""
class AlexNet(nn.Module):
    def __init__(self, channels, classes_num):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, 96, 7, 2, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0)            
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 0)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.pooling = nn.AdaptiveMaxPool2d((3, 3))
        self.fc = nn.Sequential(
            nn.Linear(256*3*3, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, classes_num)
        )

    def forward(self, x, feat=False):
        feature = self.conv1(x)
        x = self.conv2(feature)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.pooling(x)
        emd = x.view(x.size()[0], -1)
        x = self.fc(emd)
        if feat:
            return x, emd
        else:
            return x

def alexnet(channels, classes_num):
    return AlexNet(channels, classes_num)