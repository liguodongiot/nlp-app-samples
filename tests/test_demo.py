
import numpy as np
import bisect

split_map = {"train": 0.334,
                     "eval": 0.333,
                  "test": 0.333}

probability_mass = np.cumsum(list(split_map.values()))
print(probability_mass)

max_value = probability_mass[-1]

# 从一个均匀分布[low,high)中随机采样，注意定义域是左闭右开，即包含low，不包含high，默认采集一个样本
print(np.random.uniform(0, max_value))


a = [1,2,3,4,5]
# 返回要插入元素在列表中的下标。假定列表是有序的
print(bisect.bisect(a, 2))
print(a)

print(bisect.bisect(probability_mass, np.random.uniform(0, max_value)))

print("over")

