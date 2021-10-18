# 【深度学习】在PyTorch中使用Datasets和DataLoader来定制文本数据
# https://jishuin.proginn.com/p/763bfbd60d06

# 导入库
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

# 创建自定义数据集类
class CustomTextDataset(Dataset):
    def __init__(self, text, labels):
        self.labels = labels
        self.text = text

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = self.labels[idx]
        data = self.text[idx]
        sample = {"Text": data, "Class": label}
        return sample

# 定义数据和类标签
text = ['Happy', 'Amazing', 'Sad', 'Unhapy', 'Glum']
labels = ['Positive', 'Positive', 'Negative', 'Negative', 'Negative']

# 创建Pandas DataFrame
text_labels_df = pd.DataFrame({'Text': text, 'Labels': labels})

# 定义数据集对象
TD = CustomTextDataset(text_labels_df['Text'], text_labels_df['Labels'])

# 显示图像和标签
print('\nFirst iteration of data set: ', next(iter(TD)), '\n')

# 打印数据集中有多少项
print('Length of data set: ', len(TD), '\n')

# 打印整个数据集
print('Entire data set: ', list(DataLoader(TD)), '\n')


# 在机器学习或深度学习中，在训练之前需要对文本进行清理并将其转化为向量。
# DataLoader有一个方便的参数collate_fn。
# 此参数允许你创建单独的数据处理函数，并在输出数据之前将该函数中的处理应用于数据。
def collate_batch(batch):

    word_tensor = torch.tensor([[1.], [0.], [45.]])
    label_tensor = torch.tensor([[1.]])

    text_list, classes = [], []

    for (_text, _class) in batch:
        text_list.append(word_tensor)
        classes.append(label_tensor)

    text = torch.cat(text_list)
    classes = torch.tensor(classes)

    return text, classes

# 创建数据集对象的DataLoader对象
bat_size = 2
# Shuffle将在每个epoch对数据进行随机化
# DL_DS = DataLoader(TD, batch_size=bat_size, shuffle=True)
DL_DS = DataLoader(TD, batch_size=bat_size, collate_fn=collate_batch)

# 循环遍历DataLoader对象中的每个batch
for (idx, batch) in enumerate(DL_DS):
    # 打印“text”数据
    print(idx, 'Text data: ', batch, '\n')
    # 打印“Class”数据
    print(idx, 'Class data: ', batch, '\n')

