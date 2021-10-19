
from torch.utils.data.dataset import Dataset
from typing import List, Optional, Union, Dict
from transformers import PreTrainedTokenizer, InputFeatures,InputExample
from transformers.data.processors.glue import glue_convert_examples_to_features
from enum import Enum
import logging
import csv
import json
import os 
from transformers import AutoConfig,BertTokenizer
from typing import NamedTuple
import numpy as np


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"



class ClassificationDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach
    soon.
    """
    features: List[InputFeatures]
    label_list: List[str]

    def __init__(
        self,
        data_dir,
        tokenizer: PreTrainedTokenizer,
        label_list: List[str],
        limit_length: Optional[int] = None,
        mode: Union[str, Split]=Split.train,
        max_seq_length: int = 128,
        task_name='calssification',
    ):
        """
        limit_length 限制语料集的长度
        """
        self.processor = ClassificationProcessor()
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        logger.info(f"从数据集文件创建特征： {data_dir}")


        examples = self.processor.get_examples(data_dir, mode)

        # 截取语料集
        if limit_length is not None:
            examples = examples[:limit_length]

        self.features = self.processor.get_features(
            examples,
            tokenizer,
            max_length=max_seq_length,
            label_list=self.label_list,
            output_mode=task_name)
        logger.info(f"（训练||校验||测试）特征集长度：{len(self.features)}")


    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list




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
        """
            Reads a tab separated value file.
            读取TSV文件
        """
        with open(input_file, "r", encoding="utf-8-sig") as f:
            # 说明：delimiter是分隔符，quotechar是引用符，当一段话中出现分隔符的时候，
            # 用引用符将这句话括起来，就能排除歧义。
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
            返回包含许多feature的一个列表
            输入example的列表
            返回feature类的列表
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

class ClassificationProcessor(DataProcessor):
    """普通文本分类数据的处理器.

    支持的文件格式必须是 .tsv 格式，列名无要求，但数据规范是：
        第一列：text
        第二列：label
    """

    def get_examples(self, data_dir: str, mode: Split):
        file_name = "{}.tsv".format(mode.value)
        logger.info(f"获取文本数据：{os.path.join(data_dir, file_name)}")
        examples = self._create_examples(
            self._read_tsv(os.path.join(data_dir, file_name)), mode)
        
        return examples

    def get_labels(self, data_dir):
        """
            获取数据集中所有 label list.
            提取训练集、校验集和测试集的label,并排重
        """
        lines = []
        files = ['{}.tsv'.format(v.value) for k,v in Split.__members__.items()]
        for f in files:
            file_dir = os.path.join(data_dir, f)
            if os.path.exists(file_dir):
                l = self._read_tsv(file_dir)
                lines.extend(l[1:])

        labels = set()
        for (i, line) in enumerate(lines):
            label = line[1]
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
            InputExample(guid="predict-{}".format(i), text_a=line, text_b=None, label=None)
            for (i, line) in enumerate(lines)
        ]
        return examples

    def _create_examples(self, lines: List[str], mode: Split):
        """Creates examples for the training, dev and test sets."""
        examples = []
        text_index = 0
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (mode.value, i)
            text_a = line[text_index]
            if len(line) > text_index + 1:
                label = line[text_index + 1]
            else:
                label = None
            # InputExample
            # text_a和text_b是文本对，可用于问答文本匹配，text_b可选
            # guid是唯一标识符
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class TaskMode:
    training = 0
    inference = 1

class PredictionOutput(NamedTuple):
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
    qps: Optional[float]
    id2label: Optional[Dict[str, str]]


class ClassificationTask():

    def __init__(self):
        pretrain_path = '/Users/liguodong/work/pretrain_model/robert_tiny'
        self.data_dir = '/Users/liguodong/work/data/tnews'

        path = pretrain_path
        task_mode = TaskMode.training
        self.tokenizer = self._init_tokenizer(task_mode, path)
        self.config = self._init_model_config(task_mode, path)
  



    def _init_tokenizer(self, task_mode: TaskMode, path: str):
        # init tokenizer
        t_func = BertTokenizer
        tokenizer = t_func.from_pretrained(path)
        return tokenizer


    def _init_model_config(self, task_mode: TaskMode, path: str):
        # init config
        if task_mode == TaskMode.training:
            processor = ClassificationProcessor()
            label2id = processor.get_label2id(self.data_dir)
            id2label = processor.get_id2label(self.data_dir)
            num_labels = len(label2id)
            logger.info(f"数据集所有标签数：{num_labels}")
            config = AutoConfig.from_pretrained(
                path,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                output_hidden_states=True
            )
        else:
            config = AutoConfig.from_pretrained(path)
            id2label = {
                int(i): label for i, label in config.id2label.items()}

        self.id2label = id2label
        return config


    def train(self):
        id2label = self.config.id2label
        label_list = [label for _, label in id2label.items() ]

        train_dataset = ClassificationDataset(data_dir=self.data_dir,
            tokenizer=self.tokenizer,
            label_list=label_list,
            mode=Split.train,
            max_seq_length=128,
            task_name = "classification"
        )

        print("---------")


classification = ClassificationTask()
classification.train()


