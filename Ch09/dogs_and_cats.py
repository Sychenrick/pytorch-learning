import os

import h5py
from tqdm import tqdm
import numpy as np
import argparse

import torch
from torchvision import models, transforms
from torch import optim, nn
from torch.autograd import Variable
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader


class feature_net(nn.Module):
    def __init__(self,model):
        super(feature_net, self).__init__()

        if model == "vgg":
            vgg = models.vgg19(pretrained=True)
            self.feature = nn.Sequential(*list(vgg.children())[:-1])
            self.feature.add_module('global average',nn.AvgPool2d(9))

        elif model == "inceptionv3":
            inception = models.inception_v3(pretrained=True)
            self.feature = nn.Sequential(*list(inception.children())[:-1])
            self.feature.__modules.pop('13')
            self.feature.add_module('global average',nn.AvgPool2d(35))
        elif model == "resnet152":
            resnet = models.resnet152(pretrained=True)
            self.feature = nn.Sequential(*list(resnet.children())[:-1])


    def forward(self, x):
        x = self.feature(x)
        x = x.view(x.size(0),-1)

        return x

class classifier(nn.Module):
    def __init__(self,dim,classes):
        super(classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim,1000),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(1000,classes)
        )

    def forward(self, x):
        x = self.fc(x)

        return x

img_transform = transforms.Compose([
    transforms.Scale(320),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

root = '/media/sherlock/Files/kaggle_dog_vs_cat/data'
data_folder = {
    'train': ImageFolder(os.path.join(root, 'train'), transform=img_transform),
    'val': ImageFolder(os.path.join(root, 'val'), transform=img_transform)
}

# define dataloader to load images
batch_size = 'bs'
dataloader = {
    'train':
    DataLoader(
        data_folder['train'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4),
    'val':
    DataLoader(
        data_folder['val'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)
}

# get train data size and validation data size
data_size = {
    'train': len(dataloader['train'].dataset),
    'val': len(dataloader['val'].dataset)
}

# get numbers of classes
img_classes = len(dataloader['train'].dataset.classes)



def CreateFeature(model, phase, outputPath='.'):
    """
    Create h5py dataset for feature extraction.

    ARGS:
        outputPath    : h5py output path
        model         : used model
        labelList     : list of corresponding groundtruth texts
    """
    featurenet = feature_net(model)

    feature_map = torch.FloatTensor()
    label_map = torch.LongTensor()
    for data in tqdm(dataloader[phase]):
        img, label = data

        img = Variable(img, volatile=True)
        out = featurenet(img)
        feature_map = torch.cat((feature_map, out.cpu().data), 0)
        label_map = torch.cat((label_map, label), 0)
    feature_map = feature_map.numpy()
    label_map = label_map.numpy()
    file_name = '_feature_{}.hd5f'.format(model)
    h5_path = os.path.join(outputPath, phase) + file_name
    with h5py.File(h5_path, 'w') as h:
        h.create_dataset('data', data=feature_map)
        h.create_dataset('label', data=label_map)
