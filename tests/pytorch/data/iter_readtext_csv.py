
from torch.utils.data import IterableDataset


# https://blog.csdn.net/weixin_35757704/article/details/119241547
# pytorch构造可迭代的Dataset

class MyIterableDataset(IterableDataset):

    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, 'r') as file_obj:
            for line in file_obj:
                line_data = line.strip('\n').split(',')
                yield line_data


if __name__ == '__main__':
    dataset = MyIterableDataset('/Users/liguodong/work/data/bbc-text.csv')
    for data in dataset:
        print(type(data))
        print(data) # <class 'list'>
        print("----------------")
        print("----------------")



