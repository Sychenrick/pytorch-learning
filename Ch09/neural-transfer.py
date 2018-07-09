import PIL.Image as Image
import torchvision.transforms as transforms
import torch.nn as nn
import torch
import torchvision.models as models
import torch.optim as optim
from torch.autograd import Variable
'''
show img
'''
img_size = 512

def load_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0)
    return img


def show_img(img):
    img = img.squeeze(0)
    img = transforms.ToPILImage()(img)
    img.show()


'''
definte loss function
redo definte forward and backword
'''
class Content_Loss(nn.Module):
    def __init__(self,target,weight):
        super(Content_Loss, self).__init__()
        self.weight = weight
        self.target = target.detach()*self.weight
        self.criterion = nn.MSELoss()

    def forward(self,input):
        self.loss = self.criterion(input*self.weight,self.target)
        out = input.clone()
        return out

    def backward(self,retain_variabels=True):
        self.loss.backward(retain_graph=retain_variabels)
        return self.loss

'''
Gram = K * K.t
'''
class Gram(nn.Module):
    def __init__(self):
        super(Gram, self).__init__()

    def forward(self,input):
        a, b, c, d = input.size()
        feature = input.view(a * b, c * d)
        gram = torch.mm(feature, feature.t())
        gram /= (a * b * c * d)
        return gram

class Style_Loss(nn.Module):
    def __init__(self,target,wegiht):
        super(Style_Loss, self).__init__()
        self.weight = wegiht
        self.target = target.detach()*self.weight
        self.gram = Gram()
        self.criterion = nn.MSELoss()

    def forward(self,input):
        G = self.gram(input)*self.weight
        self.loss = self.criterion(G, self.target)
        out = input.clone()
        return out

    def backward(self, retain_variabels=True):
        self.loss.backward(retain_graph=retain_variabels)
        return self.loss


vgg = models.vgg19(pretrained=True).features
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_style_model_and_loss(style_img,
                             content_img,
                             cnn=vgg,
                             style_weight=1000,
                             content_weight=10,
                             content_layers=content_layers_default,
                             style_layers=style_layers_default):

    content_loss_list = []
    style_loss_list = []

    model = nn.Sequential()
    gram = Gram()

    i = 1
    for layer in cnn:
        if isinstance(layer, nn.Conv2d):
            name = 'conv_' + str(i)
            model.add_module(name, layer)

            if name in content_layers_default:
                target = model(content_img)
                content_loss = Content_Loss(target, content_weight)
                model.add_module('content_loss_' + str(i), content_loss)
                content_loss_list.append(content_loss)

            if name in style_layers_default:
                target = model(style_img)
                target = gram(target)
                style_loss = Style_Loss(target, style_weight)
                model.add_module('style_loss_' + str(i), style_loss)
                style_loss_list.append(style_loss)

            i += 1
        if isinstance(layer, nn.MaxPool2d):
            name = 'pool_' + str(i)
            model.add_module(name, layer)

        if isinstance(layer, nn.ReLU):
            name = 'relu' + str(i)
            model.add_module(name, layer)

    return model, style_loss_list, content_loss_list

def get_input_param_optimier(input_img):
    """
    input_img is a Variable
    """
    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer

def run_style_transfer(content_img, style_img, input_img, num_epoches=300):
    print('Building the style transfer model..')
    model, style_loss_list, content_loss_list = get_style_model_and_loss(
        style_img, content_img)
    input_param, optimizer = get_input_param_optimier(input_img)

    print('Opimizing...')
    epoch = [0]
    while epoch[0] < num_epoches:

        def closure():
            input_param.data.clamp_(0, 1)

            model(input_param)
            style_score = 0
            content_score = 0

            optimizer.zero_grad()
            for sl in style_loss_list:
                style_score += sl.backward()
            for cl in content_loss_list:
                content_score += cl.backward()

            epoch[0] += 1
            if epoch[0] % 50 == 0:
                print('run {}'.format(epoch))
                print('Style Loss: {:.4f} Content Loss: {:.4f}'.format(
                    style_score.data[0], content_score.data[0]))
                print()
            return style_score + content_score

        optimizer.step(closure)

        input_param.data.clamp_(0, 1)

    return input_param.data

style_img = load_img('./picture/style.png')
style_img = Variable(style_img)
content_img = load_img('./picture/content.png')
content_img = Variable(content_img)

input_img = content_img.clone()

out = run_style_transfer(content_img, style_img, input_img, num_epoches=200)

show_img(out.cpu())

save_pic = transforms.ToPILImage()(out.cpu().squeeze(0))
save_pic.save('./picture/saved_picture.png')
