
# /Users/liguodong/work/data/similarity

import pandas as pd
import csv
from typing import List

import pandas as pd
import csv
from typing import List

quotechar=None
input_file = "/Users/liguodong/work/data/similarity/train.txt"
with open(input_file, "r", encoding="utf-8-sig") as f:
    corpus:List[List[str]] = list(csv.reader(f, delimiter=" ", quotechar=quotechar))
    corpus_result:List[List[str]] = []
    for line in corpus:
        corpus_line = []
        for temp in line:
            if temp is not None and temp != '':
                corpus_line.append(temp)
        if len(corpus_line) != 3:
            print(line)
        else:
            corpus_result.append(corpus_line)



texts_a = []
texts_b = []
labels = []
text_index_a = 0
text_index_b = 1
for (i, line) in enumerate(corpus_result):
    # 第一行表头不进行处理
    if i == 0:
        continue
    text_a = line[text_index_a]
    text_b = line[text_index_b]
    if len(line) > text_index_b + 1:
        label = line[len(line) - 1]
    else:
        label = None
    texts_a.append(text_a)
    texts_b.append(text_b)
    labels.append(label)

print("-----------")

df = pd.DataFrame()

df['sentence'] = texts_a
df['sentence2'] = texts_b
df['label'] = labels

print(df.head(5))

df.to_csv('/Users/liguodong/work/data/similarity/train.tsv', sep='\t', header=["sentence", "sentence2", "label"], index=False, columns=["sentence", "sentence2", "label"], mode="w")

print("-----------")



