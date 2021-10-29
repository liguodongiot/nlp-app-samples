

import dataclasses
import json
from tests.iter_version_dev.V1_0_0.processor_bert_classification import ClassificationProcessor
import os 
from tests.iter_version_dev.V1_0_0.common import Split
from typing import List, Union, Optional, Dict
from transformers import InputExample, PreTrainedTokenizer
from dataclasses import dataclass

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
    label: Optional[List[int]] = None

    def to_json_string(self):
        return json.dumps(dataclasses.asdict(self))+"\n"


class MultiLabelProcessor(ClassificationProcessor):
    """
    多标签分类普通文本分类数据的处理器.

    支持的文件格式必须是 .tsv 格式，列名无要求，但数据规范是：
        第一列：text
        第二列：label1
        第三列：label2
        ……
        第 N列：labeln
    """

    def get_labels(self, data_dir):
        lines = self._read_tsv(os.path.join(data_dir, "train.tsv"))
        labels = set()
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            one_labels = line[1:]
            labels.update(one_labels)
        return sorted(list(labels))


    def _create_examples(self, lines: List[str], mode: Split):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (mode.value, i)
            text_a = line[0]
            if len(line) > 0:
                one_labels = line[1:]
            else:
                one_labels = None
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=one_labels))
        return examples

    def get_features(self,
                    #  examples: Union[List[InputExample], "tf.data.Dataset"],
                     examples,
                     tokenizer: PreTrainedTokenizer,
                     max_length: Optional[int] = None,
                     label_list=None,
                     output_mode=None,
                     ):
        """
        参考 transformers.data.processors.glue  glue_convert_examples_to_features
        主要修改点：label 从一个分类 id 改为 num_labels 个分类，每一个值为 0 或 1
        """

        if max_length is None:
            max_length = tokenizer.max_len

        if label_list is None:
            label_list = self.get_labels()

        label_map_id = {label: i for i, label in enumerate(label_list)}

        def label_from_example(example: InputExample):
            if example.label is None:
                return None
            one_labels = [0] * len(label_list)
            for label_str in example.label:
                if label_str in label_map_id:  # 对于测试集有，训练集未存在的数据，直接排除
                    one_labels[label_map_id[label_str]] = 1
            return one_labels

        labels = [label_from_example(example) for example in examples]

        batch_encoding = tokenizer(
            [(example.text_a, example.text_b) for example in examples],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )

        features = []
        for i in range(len(examples)):
            inputs = {k: batch_encoding[k][i] for k in batch_encoding}

            feature = InputFeatures(**inputs, label=labels[i])
            features.append(feature)

        return features
