from torchvision.datasets import ImageFolder
'''
同一类放在同一个文件下
'''

from torch.utils.data import Dataset


# 定义一个子类叫 custom_dataset，继承与 Dataset
# 重写__getitem__和__len__函数
class custom_dataset(Dataset):
    def __init__(self, txt_path, transform=None):
        self.transform = transform  # 传入数据预处理
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        self.img_list = [i.split()[0] for i in lines]  # 得到所有的图像名字
        self.label_list = [i.split()[1] for i in lines]  # 得到所有的 label

    def __getitem__(self, idx):  # 根据 idx 取出其中一个
        img = self.img_list[idx]
        label = self.label_list[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):  # 总数据的多少
        return len(self.label_list)

from torch.utils.data import DataLoader
train_data2 = DataLoader('data', 8, True)
im, label = next(iter(train_data2))
for im1, label1 in train_data2:
    pass


def collate_fn(batch):
    '''
    将上面一个 batch 输出的 label 补成相同的长度，短的 label 用 0 填充，
    我们就需要使用 collate_fn 来自定义我们 batch 的处理方式，
    :param batch:
    :return:
    '''
    batch.sort(key=lambda x: len(x[1]), reverse=True) # 将数据集按照 label 的长度从大到小排序
    img, label = zip(*batch) # 将数据和 label 配对取出
    # 填充
    pad_label = []
    lens = []
    max_len = len(label[0])
    for i in range(len(label)):
        temp_label = label[i]
        temp_label += '0' * (max_len - len(label[i]))
        pad_label.append(temp_label)
        lens.append(len(label[i]))
    # pad_label
    return img, pad_label, lens # 输出 label 的真实长度