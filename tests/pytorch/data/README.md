



## DataSet

Dataset负责对raw data source封装，将其封装成Python可识别的数据结构，其必须提供提取数据个体的接口。Dataset共有Map-style datasets和Iterable-style datasets两种：

map-style dataset：实现了__getitem__和__len__接口，表示一个从索引/key到样本数据的map。比如：datasets[10]，就表示第10个样本。
iterable-style dataset：实现了__iter__接口，表示在data samples上的一个Iterable（可迭代对象），这种形式的dataset非常不适合随机存取（代价太高），但非常适合处理流数据。比如：iter(datasets)获得迭代器，然后不断使用next迭代从而实现遍历。


### IteratorDataSet



## Sampler
Sampler负责提供一种遍历数据集所有元素索引的方式。



对于iterable-style类型的dataset来说，数据的加载顺序是完全由用户定义的迭代器来确定的。

而对于map-style类型的dataset来说，数据的索引的加载顺序由torch.utils.data.Sampler类确定。例如当使用SGD进行网络训练的时候，sampler会自动生成乱序的序列（相对数据读取的顺序），然后每次从中选择一个进行读取，或者选择一个batch进行mini-batch SGD训练。





## DataLoader

Dataloader负责加载数据，同时支持map-style和iterable-style Dataset，支持单进程/多进程，还可以设置loading order, batch size, pin memory等加载参数。





### 参考文章

- [pytorch::Dataloader中的迭代器和生成器应用详解](http://www.deiniu.com/article/177717.htm)
- []()
- []()
- []()
- []()
- []()



