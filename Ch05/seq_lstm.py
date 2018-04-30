import torch
from torch import nn
from torch.autograd import Variable
import string

training_data = [("The dog ate the apple".split(),["DET", "NN", "V", "DET", "NN"]),
                 ("Everybody read that book".split(), ["NN", "V", "DET", "NN"]),
                 ("I hate you".split(),["NN","V","NN"]),
                 ("I love children".split(),["NN","V","NN"])]
word_to_idx = {}
tag_to_idx = {}
idx_to_tag  = {}
for context, tag in training_data:
    for word in context:
        if word.lower() not in word_to_idx:
            word_to_idx[word.lower()] = len(word_to_idx)
    for label in tag:
        if label.lower() not in tag_to_idx:
            tag_to_idx[label.lower()] = len(tag_to_idx)
            idx_to_tag[len(tag_to_idx)-1                                                                                ] = label.lower()
#print(idx_to_tag)
alphabet = string.ascii_lowercase
char_to_idx = {}
for i in alphabet:
    char_to_idx[i] = alphabet.index(i)


def make_seq(vocab,dicx):
    id_indx = [dicx[i.lower()] for i in vocab]
    id_indx = torch.LongTensor(id_indx)

    return id_indx


class char_lstm(nn.Module):
    '''
    构建当个字符分类器，对单词的字符判断词性，比如ly为副词
    将隐含状态的最后一个输出
    '''
    def __init__(self, n_char, char_dim, char_hidden):
        super(char_lstm, self).__init__()

        self.char_embed = nn.Embedding(n_char, char_dim)
        self.lstm = nn.LSTM(char_dim, char_hidden)

    def forward(self, x):
        x = self.char_embed(x)
        out, _ = self.lstm(x)  #out (seq,batch,hidden)
        return out[-1]  # (batch, hidden)

class LSTMTagger(nn.Module):
    def __init__(self,n_word, n_char, char_dim, word_dim,
                 char_hidden, word_hidden, n_tag):
        super(LSTMTagger, self).__init__()
        self.word_embed = nn.Embedding(n_word,word_dim)
        self.char_lstm = char_lstm(n_char,char_dim,char_hidden)
        self.word_lstm = nn.LSTM(word_dim+char_hidden,word_hidden)
        self.classify = nn.Linear(word_hidden,n_tag)

    def forward(self, x,word):
        char = []
        for w in word:
            char_list = make_seq(w,char_to_idx)
            char_list = char_list.unsqueeze(1)
            char_infor = self.char_lstm(Variable(char_list))
            char.append(char_infor)
        char = torch.stack(char,dim=0)
        x = self.word_embed(x) # batch,seq,word_dim
        x = x.permute(1,0,2)
        x = torch.cat((x,char),dim=2)
        x, _ = self.word_lstm(x) # seq,batch,word_hidden

        s,b,h= x.shape
        x = x.view(-1,h)
        out = self.classify(x)

        return out


net = LSTMTagger(len(word_to_idx), len(char_to_idx), 10, 100, 50, 128, len(tag_to_idx))
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=1e-2)

# 开始训练
for e in range(300):
    train_loss = 0
    for word, tag in training_data:
        word_list = make_seq(word, word_to_idx).unsqueeze(0) # 添加第一维 batch
        tag = make_seq(tag, tag_to_idx)
        word_list = Variable(word_list)
        tag = Variable(tag)
        # 前向传播
        out = net(word_list, word)
        loss = criterion(out, tag)
        train_loss += loss.data[0]
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if (e + 1) % 50 == 0:
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, train_loss / len(training_data)))


net.eval()
test_sent = 'I love you'
test = make_seq(test_sent.split(), word_to_idx).unsqueeze(0)
out = net(Variable(test), test_sent.split())
_,pred = torch.max(out,1)
print([idx_to_tag[int(i)] for i in pred])
