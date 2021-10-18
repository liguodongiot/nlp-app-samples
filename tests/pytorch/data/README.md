



## DataSet

### IteratorDataSet



## Sampler

对于iterable-style类型的dataset来说，数据的加载顺序是完全由用户定义的迭代器来确定的。

而对于map-style类型的dataset来说，数据的索引的加载顺序由torch.utils.data.Sampler类确定。例如当使用SGD进行网络训练的时候，sampler会自动生成乱序的序列（相对数据读取的顺序），然后每次从中选择一个进行读取，或者选择一个batch进行mini-batch SGD训练。





## DataLoader


### 参考文章

- [pytorch::Dataloader中的迭代器和生成器应用详解](http://www.deiniu.com/article/177717.htm)
- []()
- []()
- []()
- []()
- []()



