

import json
import pandas as pd
import re

def read_text(data_path:str):
    with open(data_path, "r",encoding='utf-8') as f:
        reader = f.readlines()
        lines = []
        for line in reader:
            line_dict = json.loads(line.strip())
            line_dict['labels'] = line_dict['labels'][0]
            lines.append([line_dict['text'], line_dict["labels"]])
    return lines

data = read_text("/Users/liguodong/work/data/liantong/2000144.json")
df = pd.DataFrame(data, columns=['text', 'labels'])

# 字符串替换
df['text'] = df['text'].map(lambda x: re.sub('\s', '', x))

result = df['labels'].unique()
print(result)
print()




