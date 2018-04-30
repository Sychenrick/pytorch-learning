import torch
from torch import nn



class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # (3,32,32)
        layer1 = nn.Sequential()
        layer1.add_module('conv1',nn.Conv2d(1,16,3))
        layer1.add_module('bn1',nn.BatchNorm2d(16))
        layer1.add_module('relu1',nn.ReLU(True))
        self.layer1 = layer1

        layer2 = nn.Sequential()
        layer2.add_module('conv2',nn.Conv2d(16,32,3))
        layer2.add_module('bn2',nn.BatchNorm2d(32))
        layer2.add_module('relu2',nn.ReLU(True))
        layer2.add_module('pool2',nn.MaxPool2d(kernel_size=2,stride=2))#32,12,12
        self.layer2 = layer2

        layer3 = nn.Sequential()
        layer3.add_module("conv3",nn.Conv2d(32,64,3))#64,10,10
        layer3.add_module('bn3',nn.BatchNorm2d(64))
        layer3.add_module('relu3',nn.ReLU(True))
        self.layer3 = layer3

        layer4 = nn.Sequential()
        layer4.add_module('fc1',nn.Conv2d(64,128,kernel_size=3))
        layer4.add_module('bn4',nn.BatchNorm2d(128))
        layer4.add_module('relu4',nn.ReLU(True))
        layer4.add_module('pl4',nn.MaxPool2d(kernel_size=2,stride=2))
        self.layer4 = layer4

        layer5 = nn.Sequential()
        layer5.add_module('fc1', nn.Linear(128*4*4,1024))
        layer5.add_module('fc_relu1', nn.ReLU(True))
        layer5.add_module('fc2', nn.Linear(1024, 128))
        layer5.add_module('fc_relu2', nn.ReLU(True))
        layer5.add_module('fc3', nn.Linear(128, 10))
        self.layer5 = layer5

    def forward(self, input):
        conv1 = self.layer1(input)
        conv2 = self.layer2(conv1)
        conv3 = self.layer3(conv2)
        conv3 = self.layer4(conv3)
        fc_input = conv3.view(conv3.size(0),-1)
        fc_out = self.layer5(fc_input)

        return fc_out