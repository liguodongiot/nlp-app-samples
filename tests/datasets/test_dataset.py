from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

# Load a dataset from the Hugging Face Hub:
# cache_dir = str(Path("~/.cache/huggingface/datasets").expanduser())
# ds = load_dataset('rotten_tomatoes', cache_dir=cache_dir, split='train')
# text = ds['text'][:8]
# print("=========")

# Load a CSV file
ds = load_dataset('csv', data_files='/Users/liguodong/work/data/bbc-text.csv')

print(ds["train"][:3])
print(ds["train"]["category"][:3])
print(ds["train"][:3]["category"])
# ds = load_dataset('json', data_files='path/to/local/my_dataset.json')
print("=========")




