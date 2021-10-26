import logging
import os
from typing import List, Optional, Union, Dict
from torch.utils.data.dataset import Dataset, IterableDataset
from transformers import InputFeatures, PreTrainedTokenizer
from transformers.data.processors.glue import glue_convert_examples_to_features
import torch

from tests.iter_version_dev.V1_0_0.common import Split,TaskName
from tests.iter_version_dev.V1_0_0.processor_bert_classification import ClassificationProcessor


logger = logging.getLogger(__name__)

from tests.iter_version_dev.V1_0_0.processor_dict import DATA_PROCESSOR


class ClassificationPredictSet(Dataset):

    def __init__(
        self,
        texts: List[str],
        id2label: Dict[str, int],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 128,
        task_name='calssification',
    ):
        self.processor = DATA_PROCESSOR[task_name]()
        examples = self.processor.get_predict_examples(texts)
        self.label_list = [label for _, label in id2label.items()]
        self.features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            max_length=max_seq_length,
            label_list=self.label_list,
            output_mode='classification',
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list


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
        logger.info(f"分类数据集入参（ClassificationDataset）：\n data_dir: {data_dir}, mode:{mode}")
        self.processor = DATA_PROCESSOR[task_name]()
        if isinstance(mode, str):
            try:
                mode = Split[mode]
            except KeyError:
                raise KeyError("mode is not a valid split name")
        self.label_list = label_list

        # Make sure only the first process in distributed training processes the dataset,
        # and the others will use the cache.
        logger.info(
            f"Creating features from dataset file at {data_dir}")

        examples = self.processor.get_examples(data_dir, mode)

        # if limit_length is not None:
        #     examples = examples[:limit_length]
        examples = examples[:2000]

        self.features = self.processor.get_features(
            examples,
            tokenizer,
            max_length=max_seq_length,
            label_list=self.label_list,
            output_mode=task_name)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list


class ClassificationIterableDataset(torch.utils.data.IterableDataset):
    """
    This will be superseded by a framework-agnostic approach soon.
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
        logger.info(f"分类数据集入参（ClassificationIterableDataset）：\n data_dir: {data_dir}, mode:{mode}")
        self.processor = DATA_PROCESSOR[task_name]()
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

        if limit_length is not None:
            examples = examples[:limit_length]

        self.features = self.processor.get_features(
            examples = examples,
            tokenizer = tokenizer,
            max_length=max_seq_length,
            label_list=self.label_list,
            output_mode=task_name)
        logger.info(f"（训练||校验||测试）特征集长度：{len(self.features)}")

    def __len__(self):
        return len(self.features)

    def __iter__(self):
        for feature in self.features:
            yield feature

    def get_labels(self):
        return self.label_list


class ClassificationPredictSet(Dataset):

    def __init__(
        self,
        texts: List[str],
        id2label: Dict[str, int],
        tokenizer: PreTrainedTokenizer,
        max_seq_length: int = 128,
        task_name='calssification',
    ):
        self.processor = DATA_PROCESSOR[task_name]()
        examples = self.processor.get_predict_examples(texts)
        self.label_list = [label for _, label in id2label.items()]
        self.features = glue_convert_examples_to_features(
            examples,
            tokenizer,
            max_length=max_seq_length,
            label_list=self.label_list,
            output_mode='classification',
        )

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    def get_labels(self):
        return self.label_list


