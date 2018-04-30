import torch
from torch.autograd import Variable
from torch import nn
from torchvision import transforms,datasets
from torch.utils.data import DataLoader
from untils import train

class RNN(nn.Module):

    def __init__(self,in_dim,hidden_dim,n_layer,n_class):
        super(RNN, self).__init__()
        self.n_layer = n_layer
        self.hidden = hidden_dim
        self.lstm = nn.LSTM(in_dim,hidden_dim,n_layer,batch_first=True)
        self.classifier = nn.Linear(hidden_dim,n_class)

    def forward(self, x):
        out,(h,c) = self.lstm(x)
        print(out.size(),h.size(),c.size())
        out = out[:,-1,:]
        out = self.classifier(out)

        return out

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
model = RNN(28,10,4,10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), 1e-1)  # 使用随机梯度下降，学习率 0.1
train(model,train_loader,test_loader,20,optimizer,criterion)