
# /Users/liguodong/work/data/tnews

import pandas as pd
import csv
from typing import List

quotechar=None
input_file = "/Users/liguodong/work/data/tnews/temp/test.txt"
with open(input_file, "r", encoding="utf-8-sig") as f:
    corpus:List[str] = list(csv.reader(f, delimiter=" ", quotechar=quotechar))

texts = []
labels = []
text_index = 0
for (i, line) in enumerate(corpus):
    # 第一行表头不进行处理
    if i == 0:
        continue
    text_a = line[text_index]
    if len(line) > text_index + 1:
        label = line[len(line) - 1]
    else:
        label = None
    texts.append(text_a)
    labels.append(label)

print("-----------")

df = pd.DataFrame()

df['sentence'] = texts
df['label'] = labels

print(df.head(5))

df.to_csv('/Users/liguodong/work/data/tnews/temp/test.tsv', sep='\t', header=["sentence", "label"], index=False, columns=["sentence", "label"], mode="w")

print("-----------")

