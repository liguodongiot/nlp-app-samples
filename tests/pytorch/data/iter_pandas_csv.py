from torch.utils.data import IterableDataset
import pandas as pd
from torch.utils.data import DataLoader
import torch

# https://blog.csdn.net/weixin_35757704/article/details/119241547
# pytorch构造可迭代的Dataset

class PandasIterableDataset(IterableDataset):
    def __init__(self, file_path):
        # 对无表头的数据，则需设置 header=None，否则第一行数据被作为表头

        # header=[0] 表示第一行为表头
        # header=[0,1,3] 表示第一二四行为表头，数据从第五行开始。
        self.data_iter = pd.read_csv(file_path, iterator=True, header=[0], chunksize=1)
        # self.data_iter = pd.read_csv(file_path, iterator=False, header=None, chunksize=1)

    def __iter__(self):
        for dataframe in self.data_iter:
            data = torch.tensor(data=dataframe.values)
            yield data
            # yield dataframe


# if __name__ == '__main__':
#     dataset = PandasIterableDataset('/Users/liguodong/work/data/bbc-text.csv')
#     for data in dataset:
#         print(type(data)) # <class 'pandas.core.frame.DataFrame'>
#         print(data)
#         print("----------------")
#         print("----------------")

if __name__ == '__main__':
    dataset = PandasIterableDataset('/Users/liguodong/work/data/bbc-number.csv')
    dl = DataLoader(dataset, num_workers=0, batch_size=10)
    print(type(dl)) # <class 'torch.utils.data.dataloader.DataLoader'>
    
    # for data in dl:
    #     print(data)
    #     print("----------------")
    #     print("----------------")

    for batch_idx, data in enumerate(dl):
        print(f'batch idx: {batch_idx}, batch len:  {len(data)} type: {type(data)}')
        print("----------------")
        print("----------------")

        