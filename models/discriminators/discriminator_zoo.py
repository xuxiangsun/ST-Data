import torch.nn as nn
#----------------------------------------------------------------------------
# https://github.com/val-iisc/Hard-Label-Model-Stealing/blob/main/code/train_generator/dcgan_model.py
class DCGAN(nn.Module):
    def __init__(self, input_nc=3, ndf=64, **extraparas):
        super(DCGAN, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(input_nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 2, 2, 0, bias=False)
        )

    def forward(self, input):
        output = self.main(input)
        return output.view(-1, 1).squeeze(1)