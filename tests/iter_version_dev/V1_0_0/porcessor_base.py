
import csv
from typing import List, Optional, Union
from transformers import InputExample, PreTrainedTokenizer
from transformers.data.processors.glue import glue_convert_examples_to_features
import json
from sklearn.feature_extraction.text import CountVectorizer


# 4、 数据预处理基类

class DataProcessor:
    """Base class for data converters for sequence classification data sets."""

    def get_examples(self, data_dir):
        """Gets a collection of :class:`InputExample` for the data set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    def get_features(self,
                     # examples: Union[List[InputExample], "tf.data.Dataset"],
                     examples,
                     tokenizer: PreTrainedTokenizer,
                     max_length: Optional[int] = None,
                     label_list=None,
                     output_mode=None,
                     ):
        """
        因为多标签分类需要使用，所以从dataset中通用逻辑抽取出来
        :param examples:
        :param tokenizer:
        :param max_length:
        :param label_list:
        :param output_mode:
        :return:
        """
        return glue_convert_examples_to_features(
            examples,
            tokenizer,
            max_length=max_length,
            label_list=label_list,
            output_mode='classification',
        )

    @classmethod
    def _read_json(cls, input_file):
        """Reads a json list file."""
        with open(input_file, "r") as f:
            reader = f.readlines()
            lines = []
            for line in reader:
                lines.append(json.loads(line.strip()))
            return lines

