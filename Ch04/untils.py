o
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable


def train(net,train_data,valid_data,num_epoches,optimizer,criterion):
    length = len(train_data)
    for epoch in range(num_epoches):
        train_loss = 0
        train_acc = 0
        net = net.train()
        for iter,data in enumerate(train_data):
            im, label = data
            im = Variable(im)
            label = Variable(label)
            output = net(im)
            loss = criterion(output,label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred_label = torch.max(output.data,1)
            train_loss += loss.data[0]
            temp_loss = loss.data[0]
            train_acc += torch.sum(pred_label == label.data)

            temp_acc = (torch.sum(pred_label == label.data)) / label.size(0)
            if iter % 300 == 0 and iter > 0:
                print('Epoch {}/{},Iter {}/{} Loss: {:.4f},ACC:{:.4f}' \
                      .format(epoch, num_epoches - 1,iter,length,temp_loss,temp_acc))
        if valid_data is not None:
            valid_loss = 0
            valid_acc = 0
            net = net.eval()
            for data in valid_data:
                im,label = data
                im = Variable(im,volatile=True)
                label = Variable(label,volatile=True)
                output = net(im)
                _, pred_label = torch.max(output.data,1)
                loss = criterion(output, label)
                valid_loss += loss.data[0]
                valid_acc += torch.sum(pred_label == label.data)
            print('Epoch {}/{},complete! train_loss: {:.4f},train_acc:{:.4f}' \
                  .format(epoch, num_epoches - 1,train_loss, train_acc/60000),
                  'valid_loss: {:.4f},valid_acc:{:.4f}'.format(valid_loss,valid_acc/10000)
                  )
        else:
            pass

