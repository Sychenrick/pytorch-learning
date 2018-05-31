import os

import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST
from torchvision import transforms as tfs
from torchvision.utils import save_image

im_tfs = tfs.Compose([
    tfs.ToTensor(),
    tfs.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # 标准化 (-1,1)
])
train_set = MNIST('../data', transform=im_tfs)
train_data = DataLoader(train_set, batch_size=128, shuffle=True)

# define network structure
class autoEncoder(nn.Module):
    def __init__(self):
        super(autoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(28*28,128),
            nn.ReLU(True),
            nn.Linear(128,64),
            nn.ReLU(True),
            nn.Linear(64,12),
            nn.ReLU(True),
            nn.Linear(12,3)
        )

        self.decoder = nn.Sequential(
            nn.Linear(3,12),
            nn.ReLU(True),
            nn.Linear(12,64),
            nn.ReLU(True),
            nn.Linear(64,128),
            nn.ReLU(True),
            nn.Linear(128, 28*28),
            nn.Tanh(),
        )

        # 这里定义的编码器和解码器都是 4 层神经网络作为模型，中间使用 relu 激活函数，
        # 最后输出的 code 是三维，注意解码器最后我们使用 tanh 作为激活函数，
        # 因为输入图片标准化在 -1 ~ 1 之间，所以输出也要在 -1 ~ 1 这个范围内，

    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return encode, decode

class convAutoEncoder(nn.Module):
        def __init__(self):
            super(convAutoEncoder, self).__init__()

            self.encoder = nn.Sequential(
                nn.Conv2d(1, 16, 3, stride=3, padding=1),  # (b, 16, 10, 10)
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=2),  # (b, 16, 5, 5)
                nn.Conv2d(16, 8, 3, stride=2, padding=1),  # (b, 8, 3, 3)
                nn.ReLU(True),
                nn.MaxPool2d(2, stride=1)  # (b, 8, 2, 2)
            )

            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(8, 16, 3, stride=2),  # (b, 16, 5, 5)
                #stride*(width-1)-2*padding+kernel
                nn.ReLU(True),
                nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # (b, 8, 15, 15)
                nn.ReLU(True),
                nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # (b, 1, 28, 28)
                nn.Tanh()
            )

        def forward(self, x):
            encode = self.encoder(x)
            decode = self.decoder(encode)
            return encode, decode


# test network
# net = convAutoEncoder()
# x = Variable(torch.randn(1, 1, 28, 28))
# code, _ = net(x)
# print(code.shape)

def to_img(x):
    '''
    定义一个函数将最后的结果转换回图片
    '''
    x = 0.5 * (x + 1.)
    x = x.clamp(0, 1)
    x = x.view(x.shape[0], 1, 28, 28)
    return x


# train
def train_model(net,num,conv=False):
    criterion = nn.MSELoss(size_average=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    for epoch in range(num):
        for im,_ in train_data:
            if not conv:
                im = im.view(im.shape[0],-1)
            im = Variable(im)
            _,output = net(im)
            loss = criterion(output, im)/im.shape[0]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 20 == 0:  # 每 20 次，将生成的图片保存一下
            print('epoch: {}, Loss: {:.4f}'.format(epoch + 1, loss.data[0]))
            pic = to_img(output.cpu().data)
            if not os.path.exists('./simple_autoencoder'):
                os.mkdir('./simple_autoencoder')
            save_image(pic, './simple_autoencoder/image_{}.png'.format(epoch + 1))

def plot_encoder(net):
    import matplotlib.pyplot as plt
    from matplotlib import cm
    from mpl_toolkits.mplot3d import Axes3D
    view_data = Variable((train_set.train_data[:200].type(torch.FloatTensor).view(-1, 28 * 28) / 255. - 0.5) / 0.5)
    encode, _ = net(view_data)  # 提取压缩的特征值
    fig = plt.figure(2)
    ax = Axes3D(fig)  # 3D 图
    # x, y, z 的数据值
    X = encode.data[:, 0].numpy()
    Y = encode.data[:, 1].numpy()
    Z = encode.data[:, 2].numpy()
    values = train_set.train_labels[:200].numpy()  # 标签值
    for x, y, z, s in zip(X, Y, Z, values):
        c = cm.rainbow(int(255 * s / 9))  # 上色
        ax.text(x, y, z, s, backgroundcolor=c)  # 标位子
    ax.set_xlim(X.min(), X.max())
    ax.set_ylim(Y.min(), Y.max())
    ax.set_zlim(Z.min(), Z.max())
    plt.show()

if __name__ == "__main__":
    net = convAutoEncoder()
    train_model(net,40,True)
    plot_encoder(net)
