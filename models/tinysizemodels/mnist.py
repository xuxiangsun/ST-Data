'''small(4-layers)/medium(5-layers)/large(6-layers) for MNIST in Pytorch.'''
from re import X
import torch.nn as nn
import torch.nn.functional as F

class small(nn.Module):
    def __init__(self, channels, classes):
        super(small, self).__init__()
        self.conv1 = nn.Conv2d(channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.pool = nn.AdaptiveMaxPool2d(4)#Add a AdaMaxPooling to fit the input size of self.fc1
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, classes)

    def forward(self, x, feat=False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        feature = self.pool(x)
        # print(x.shape)
        x = feature.view(-1, 4*4*50)
        embs = F.relu(self.fc1(x))
        x = self.fc2(embs)
        if feat:
            return x, embs
        else:
            return x

class medium(nn.Module):
    def __init__(self, channels, classes):
        self.number = 0
        super(medium, self).__init__()
        self.conv1 = nn.Conv2d(channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)
        self.fc1 = nn.Linear(2*2*50, 500)
        self.fc2 = nn.Linear(500, classes)

    def forward(self, x, sign=0, feat=False):
        if sign == 0:
            self.number += 1
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 2*2*50)
        feature = F.relu(self.fc1(x))
        x = self.fc2(feature)
        if feat:
            return x, feature
        else:
            return x

    def get_number(self):
        return self.number


class large(nn.Module):
    def __init__(self, channels, classes):
        super(large, self).__init__()
        self.conv1 = nn.Conv2d(channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv3 = nn.Conv2d(50, 50, 3, 1, 1)
        self.conv4 = nn.Conv2d(50, 50, 3, 1, 1)
        self.fc1 = nn.Linear(50, 500)
        self.fc2 = nn.Linear(500, classes)

    def forward(self, x, feat=False):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv3(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 50)
        feature = F.relu(self.fc1(x))
        x = self.fc2(feature)
        if feat:
            return x, feature
        else:
            return x

class lenet5(nn.Module):
    
    def __init__(self, channels, classes):
        super(lenet5, self).__init__()

        self.conv1 = nn.Conv2d(channels, 6, kernel_size=(5, 5))
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # self.conv3 = nn.Conv2d(16, 120, kernel_size=(5, 5))
        # self.relu3 = nn.ReLU()
        self.fc3 = nn.Linear(16*5*5, 120)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(84, classes)

    def forward(self, img, feat=False):
        output = self.conv1(img)
        output = self.relu1(output)
        output = self.maxpool1(output)
        output = self.conv2(output)
        output = self.relu2(output)
        output = self.maxpool2(output)
        output = output.view(-1, 16*5*5)
        output = self.fc3(output)
        output = self.relu3(output)
        output = self.fc4(output)
        feature = self.relu4(output)
        x = self.fc5(feature)
        if feat:
            return x, feature
        else:
            return x