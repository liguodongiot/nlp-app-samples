

import pandas as pd
import numpy as np
import torch
import time
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import Adam, AdamW
from nlp_app_samples.algo.text_cnn.model import TextCNN
import jieba
import os
import random
from tqdm import tqdm


def seed_torch(seed=1029):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# 统计词库
def count_word(sentences, word_to_index):
    with open('/Users/liguodong/data/stopwords.txt', "r", encoding="utf-8") as f:
        stopwords = f.readlines()
    for i in range(len(stopwords)):
        stopwords[i] = stopwords[i].strip("\n")
    for sentence in sentences:
        for word in sentence:
            if word not in word_to_index and word not in stopwords:
                word_to_index[word] = len(word_to_index)

# 将word转换为token
def sentence_to_index(sentence, max_len, word_to_index):
    sentence = [word_to_index.get(word, 0) for word in sentence]
    if len(sentence) < max_len:
        sentence += (max_len - len(sentence)) * [0]
    else:
        sentence = sentence[:max_len]
    return sentence


# 定义数据类
class MyDataset(Dataset):

    def __init__(self, sent, label):
        self.sentence = np.array(sent).astype('float')
        self.label = np.array(label)

    def __getitem__(self, index):
        label, sentence = self.label[index], self.sentence[index]

        return {'label': label, 'sentence': torch.Tensor(sentence)}

    def __len__(self):
        return len(self.sentence)

# 加载训练数据
def load_train(path, type='char'):
    train = {}
    train['label'] = []
    train['sentence'] = []
    word_to_index = {'pad': 0}

    with open(path, encoding='utf-8') as f:
        for line in f:
            line = eval(line)
            train['label'].append(line['label'])
            train['sentence'].append(line['sentence'])

    train = pd.DataFrame(train)
    label_to_id = {}
    sents = []
    labels = []
    for i, label in enumerate(train['label'].unique()):
        label_to_id[label] = i
    if type == 'char':
        train['cut_sentence'] = train['sentence']
    else:
        train['cut_sentence'] = train['sentence'].map(jieba.lcut)
    count_word(train['cut_sentence'], word_to_index)

    for sent, label in zip(train['cut_sentence'], train['label']):
        sents.append(sentence_to_index(sent, max_len=max_len, word_to_index=word_to_index))
        labels.append(label_to_id[label])
    return sents, labels, label_to_id, word_to_index

# 加载验证数据
def load_val(val_path, label_to_id, word_to_index, type='char'):
    val = {}
    val['label'] = []
    val['sentence'] = []
    with open(val_path, encoding='utf-8') as f:
        for line in f:
            line = eval(line)
            val['label'].append(line['label'])
            val['sentence'].append(line['sentence'])

    val = pd.DataFrame(val)
    val_sents = []
    val_labels = []
    if type == 'char':
        val['cut_sentence'] = val['sentence']
    else:
        val['cut_sentence'] = val['sentence'].map(jieba.lcut)
    for sent, label in zip(val['cut_sentence'], val['label']):
        val_sents.append(sentence_to_index(sent, max_len=max_len, word_to_index=word_to_index))
        val_labels.append(label_to_id[label])

    return val_sents, val_labels

# 评估函数
def evaluate(val_loader, model, device):
    model.eval()
    corrects, avg_loss = 0, 0
    for data in val_loader:
        label = data['label']
        sentence = data['sentence']

        label = label.type(torch.LongTensor)
        sentence = sentence.type(torch.LongTensor)

        label = label.to(device)
        sentence = sentence.to(device)

        output = model(sentence)

        loss = criterion(output, label)

        avg_loss += loss.item()
        corrects += (output.argmax(1) == label).sum().item()

    size = len(val_dataset)
    avg_loss /= size
    accuracy = corrects / size
    print('\nEvaluation - loss: {:.3f} acc: {:.4f}\n'.format(avg_loss, accuracy))

    return accuracy


seed_torch()
train_path = '/Users/liguodong/data/iflytek_public/train.json'
val_path = '/Users/liguodong/data/iflytek_public/dev.json'

max_len = 64
# 加载数据
sents, labels, label_to_id, word_to_index = load_train(train_path, type='char')
val_sents, val_labels = load_val(val_path, label_to_id, word_to_index, type='char')
# 超参数
batch_size = 128
learn_rate = 3e-4
n_epochs = 64
embedding_dim = 300
windows_size = [2, 4, 3]
feature_size = 100
dropout = 0.5
vocab_size = len(word_to_index)
n_class = len(label_to_id)

# 训练
train_dataset = MyDataset(sents, labels)
val_dataset = MyDataset(val_sents, val_labels)

train_loader = DataLoader(train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=0)

val_loader = DataLoader(val_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0)

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

model = TextCNN(vocab_size, embedding_dim, windows_size, max_len, feature_size, n_class, dropout).to(device)

optimizer = Adam(model.parameters(), lr=learn_rate)

criterion = nn.CrossEntropyLoss()
best_acc = 0.0
early_times = 0

for epoch in range(1, n_epochs + 1):
    print("epoch: {}".format(epoch))
    running_loss = 0.0
    model.train()
    start = time.time()

    for batch_i, data in enumerate(train_loader):
        start = time.time()
        label = data['label']
        sentence = data['sentence']

        label = label.type(torch.LongTensor)
        sentence = sentence.type(torch.LongTensor)

        label = label.to(device)
        sentence = sentence.to(device)

        output = model(sentence)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if batch_i % 100 == 99:
            print('epoch: {}, batch: {}/{}, loss: {}'.format(epoch, batch_i + 1, len(train_loader),
                                                           round(running_loss / 100/64, 4)))
            running_loss = 0.0

    # 评估
    val_acc = evaluate(val_loader, model, device)
    if best_acc < val_acc:
        best_acc = val_acc
        early_times = 0
    else:
        early_times += 1
        if early_times > 5:
            print('EarlyStopping---best_acc: {}, '.format(best_acc))
            break