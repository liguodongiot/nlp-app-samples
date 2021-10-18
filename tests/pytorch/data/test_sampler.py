from torch.utils.data import sampler

# pytorch源码阅读（三）Sampler类与4种采样方式
# https://zhuanlan.zhihu.com/p/100280685?utm_source=qq

## 顺序采样Sequential Sampler

# 定义数据和对应的采样器
data = list([17, 22, 3, 41, 8])
seq_sampler = sampler.SequentialSampler(data_source=data)

# 迭代获取采样器生成的索引
for index in seq_sampler:
    print("index: {}, data: {}".format(str(index), str(data[index])))

print("----------------")

## 随机采样RandomSampler
ran_sampler = sampler.RandomSampler(data_source=data)
print("----------------")
for index in ran_sampler:
    print("index: {}, data: {}".format(str(index), str(data[index])))


ran_sampler = sampler.RandomSampler(data_source=data, replacement=True)

print("----------------")
for index in ran_sampler:
    print("index: {}, data: {}".format(str(index), str(data[index])))

print("----------------")

# 子集随机采样

sub_sampler_train = sampler.SubsetRandomSampler(indices=data[0:2])
for index in sub_sampler_train:
    print("data: {}".format(str(index)))

print("----------------")

sub_sampler_val = sampler.SubsetRandomSampler(indices=data[2:])
for index in sub_sampler_val:
    print("data: {}".format(str(index)))


print("----------------")
