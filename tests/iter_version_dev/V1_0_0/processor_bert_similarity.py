
from transformers import InputExample, PreTrainedTokenizer, InputFeatures
from typing import List, Optional, Union, Dict
from tests.iter_version_dev.V1_0_0.porcessor_base import DataProcessor
from tests.iter_version_dev.V1_0_0.common import Split
import os


class PairTextProcessor(DataProcessor):
    """文本对分类的处理器.

    支持的文件格式必须是 .tsv，列名无要求，但数据规范是：
        第一列：text_1
        第二列：text_2
        第三列：label
    """

    def get_examples(self, data_dir: str, mode: Split):
        file_name = "{}.tsv".format(mode.value)
        examples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), mode)
        return examples

    def get_labels(self, data_dir):
        """获取数据集中所有 label list."""
        lines = []
        files = ['{}.tsv'.format(v.value) for k,v in Split.__members__.items()]
        for f in files:
            file_dir = os.path.join(data_dir, f)
            if os.path.exists(file_dir):
                l = self._read_tsv(file_dir)
                lines.extend(l[1:])

        labels = set()
        for (i, line) in enumerate(lines):
            label = line[2]
            labels.add(label)
        return sorted(list(labels))

    def get_label2id(self, data_dir):
        label_list = self.get_labels(data_dir)
        return {label: idx for idx, label in enumerate(label_list)}

    def get_id2label(self, data_dir):
        label2id = self.get_label2id(data_dir)
        return {idx: label for label, idx in label2id.items()}

    def get_predict_examples(self, lines):
        examples = [
            InputExample(guid="predict-{}".format(i), text_a=line[0], text_b=line[1], label=None)
            for (i, line) in enumerate(lines)
        ]
        return examples

    def _create_examples(self, lines: List[str], mode: Split):
        """Creates examples for the training, dev and test sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (mode.value, i)
            try:
                text_a = line[0]
                text_b = line[1]
                label = line[2]
            except IndexError:
                continue
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples



