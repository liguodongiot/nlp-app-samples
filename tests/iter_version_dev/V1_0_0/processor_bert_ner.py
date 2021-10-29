
from typing import List, Optional, Union
from dataclasses import dataclass
import os


from tests.iter_version_dev.V1_0_0.common import Split
from tests.iter_version_dev.V1_0_0.porcessor_base import DataProcessor


@dataclass
class InputExample:
    """
    词分类
    A single training/test example for token classification.
    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """
    guid: str
    words: List[str]
    labels: Optional[List[str]]


@dataclass
class InputFeatures:
    """
    特征数据
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class TokenClassProcessor(DataProcessor):

    def get_examples(self, data_dir: str, mode: Split):
        mode = mode.value
        file_path = os.path.join(data_dir, f"{mode}.txt")
        guid_index = 1
        examples = []
        with open(file_path, encoding="utf-8") as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
                        guid_index += 1
                        words = []
                        labels = []
                else:
                    splits = line.split(" ")
                    words.append(splits[0])
                    if len(splits) > 1:
                        labels.append(splits[-1].replace("\n", ""))
                    else:
                        # Examples could have no label for mode = "test"
                        labels.append("O")
            if words:
                examples.append(InputExample(guid=f"{mode}-{guid_index}", words=words, labels=labels))
        return examples

    def get_label2id(self, data_dir: str):
        label_map = {"O": 0}
        label_id = 1
        train_set_path = os.path.join(data_dir, "train.txt")
        with open(train_set_path, encoding="utf-8") as f:
            words = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if words:
                        words = []
                else:
                    splits = line.split(" ")
                    if len(splits) > 1:
                        label = splits[-1].replace("\n", "")
                        if label not in label_map:
                            label_map[label] = label_id
                            label_id += 1
        return label_map

    def get_labels(self, data_dir):
        # NER 任务不需要 label list
        return []

    def get_id2label(self, data_dir):
        label2id = self.get_label2id(data_dir)
        return {idx: label for label, idx in label2id.items()}


