import pandas as pd
from pandas.io.parsers.readers import TextFileReader


# TSV文件和CSV的文件的区别是：前者使用\t作为分隔符，后者使用,作为分隔符。
# train=pd.read_csv('test.tsv', sep='\t')


merge_dt:TextFileReader = pd.read_csv('/Users/liguodong/work/data/bbc-text.csv', iterator = True)
print(merge_dt)
merge_data = merge_dt.get_chunk(10)
print(merge_data)
print(type(merge_data))

print("=======================")

merge_dt = pd.read_csv('/Users/liguodong/work/data/bbc-simple.csv', iterator = True, chunksize = 5) 

loop = True
chunkSize = 10 # 每次读取的行数

while loop:
    try:
        chunk = merge_dt.get_chunk(chunkSize)
        print(merge_data)
        # 需要注意的是文件的列名
        # do something 
    except StopIteration:
        loop = False
        print("Iteration is stopped.")


