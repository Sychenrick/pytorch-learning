
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import simplenet
from untils import train

"""
generate the minis dataset

"""

data_tf = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0.5],[0.5])]
)
train_dataset = datasets.MNIST(
    root='../data',train=True,transform=data_tf,download=True
)
test_dataset = datasets.MNIST(
    root="../data",train=False,transform=data_tf
)

batch_size = 64
learning_rate = 1e-2
num_epoches = 20

train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=batch_size*2,shuffle=False)

in_dim,n_hidden_1,n_hidden_2,out_dim = 28*28,300,100,10

model1 = simplenet.simpleNet(in_dim,n_hidden_1,n_hidden_2,out_dim)
model2 = simplenet.activationNet(in_dim,n_hidden_1,n_hidden_2,out_dim)
model3 = simplenet.batchNet(in_dim,n_hidden_1,n_hidden_2,out_dim)

for model in [model3]:
    print("the {} start traing...".format(model.get_name()))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), 1e-1)  # 使用随机梯度下降，学习率 0.1
    train(model,train_loader,test_loader,20,optimizer,criterion)
    print("the {} complete traing...".format(model.get_name()))
