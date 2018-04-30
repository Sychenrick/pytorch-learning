import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable




CONTEXT_SIZE = 2 # 依据的单词数
EMBEDDING_DIM = 10 # 词向量的维度
# 我们使用莎士比亚的诗
test_sentence = """When forty winters shall besiege thy brow,
And dig deep trenches in thy beauty's field,
Thy youth's proud livery so gazed on now,
Will be a totter'd weed of small worth held:
Then being asked, where all thy beauty lies,
Where all the treasure of thy lusty days;
To say, within thine own deep sunken eyes,
Were an all-eating shame, and thriftless praise.
How much more praise deserv'd thy beauty's use,
If thou couldst answer 'This fair child of mine
Shall sum my count, and make my old excuse,'
Proving his beauty by succession thine!
This were to be new made when thou art old,
And see thy blood warm when thou feel'st it cold.""".split()

trigram = [((test_sentence[i], test_sentence[i + 1]), test_sentence[i + 2])
           for i in range(len(test_sentence) - 2)]

vocb = set(test_sentence) # 使用 set 将重复的元素去掉
word_to_idx = {word: i for i, word in enumerate(vocb)}
idx_to_word = {word_to_idx[word]: word for word in word_to_idx}

#define model
class NGram(nn.Module): 
    def __init__(self,vocab_size, context_size, n_dim):
        super(NGram, self).__init__()
        #self.n_word = vocab_size
        self.embed = nn.Embedding(vocab_size, n_dim)
        self.classify = nn.Sequential(
            nn.Linear(context_size * n_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, vocab_size)
        )

    def forward(self, x):
        voc_embed = self.embed(x)  # 得到词嵌入
        voc_embed = voc_embed.view(1, -1)  # 将两个词向量拼在一起
        out = self.classify(voc_embed)

        return out


length = len(word_to_idx)
model = NGram(length,CONTEXT_SIZE,EMBEDDING_DIM)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-2,weight_decay=1e-5)

#training
for e in range(100):
    train_loss = 0
    for word, label in trigram[0:100]:  # 使用前 100 个作为训练集
        word = Variable(torch.LongTensor([word_to_idx[i] for i in word])) # 将两个词作为输入
        label = Variable(torch.LongTensor([word_to_idx[label]]))
        # 前向传播
        out = model(word)
        loss = criterion(out, label)
        train_loss += loss.data[0]
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (e + 1) % 50 == 0:
        print('epoch: {}, Loss: {:.6f}'.format(e + 1, train_loss / len(trigram)))

#valid

model.eval()
sum = 0
acc = 0
for word,label in trigram[100:]:
    #print('input: {}'.format(word))
    #print('label: {}'.format(label))
    sum += 1
    word = Variable(torch.LongTensor([word_to_idx[i] for i in word]))
    out = model(word)
    pred_label_idx = out.max(1)[1].data[0]
    predict_word = idx_to_word[pred_label_idx]
    #print('real word is {}, predicted word is {}'.format(label, predict_word))

    if predict_word == label:
        acc += 1

print("the valid acc: {:.4f}".format(float(acc)/float(sum)))