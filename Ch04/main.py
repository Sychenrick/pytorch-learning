from cnn_bn import CNN
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
from untils import train
from torch import nn
import torch

data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5],[0.5])]
)
train_dataset = datasets.MNIST(
    root='../data',train=True,transform=data_tf,download=False
)
test_dataset = datasets.MNIST(
    root="../data",train=False,transform=data_tf
)

batch_size = 64
learning_rate = 1e-2
num_epoches = 20

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size*2,shuffle=False)
model = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 1e-1)  # 使用随机梯度下降，学习率 0.1
train(model,train_loader,test_loader,20,optimizer,criterion)