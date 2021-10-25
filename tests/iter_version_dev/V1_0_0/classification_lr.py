#! -*- coding: utf-8 -*-
import logging
import random
import numpy as np
import os
import torch
from typing import NamedTuple, List, Union
import time
import pandas as pd
import shutil
import argparse

import math
from collections import defaultdict, Iterable
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from enum import Enum
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import pickle
from typing import Callable, Dict, Optional, List
import json
import socket
from datetime import datetime
from transformers.file_utils import cached_property, is_torch_available, is_torch_tpu_available, torch_required
import requests
import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

logger = logging.getLogger(__name__)

class TaskMode:
    training = 0
    inference = 1

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"

class LanguageType:
    """
    type of language
    """
    CN = 2
    EN = 1

"""
预处理
"""
class Preprocessor(object):

    @classmethod
    def texts_clean(cls, texts):
        """
        :param texts: list of str.
        :return: list str. 
        cleaned texts
        数据清洗
        """
        return texts

    @classmethod
    def train_vectorizers(cls, texts, labels, **kwargs):
        """
        :param texts: list of str
        :param labels: list of str
        :return: vectorizer, labelencoder, x, y
        训练文本转向量
        """
        raise NotImplementedError('not implement train_vetorizers method!')

    @classmethod
    def predict_vectorizer(cls, text, vectorizer, **kwargs):
        """
        :param text: str
        :param vectorizer: such as countvectorize  word2vec
        :return: numpy of shape
        预测文本清洗及文本转向量
        """
        clean_text = cls.texts_clean([text])[0]
        return vectorizer.transform([clean_text])

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """
            Reads a tab separated value file.
            读取以制表符分割的TSV文件
        """
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    """
        获取数据集
    """
    @classmethod
    def get_examples(cls, data_dir: str, mode: Split):
        file_name = "{}.tsv".format(mode.value)
        examples = cls._create_examples(cls._read_tsv(os.path.join(data_dir, file_name)), mode)
        return examples

    """
        将数据集处理成texts和labels
    """
    @classmethod
    def _create_examples(cls, lines: List[str], mode: Split):
        """Creates examples for the training, dev and test sets."""
        texts = []
        labels = []
        text_index = 0
        for (i, line) in enumerate(lines):
            # 第一行表头不进行处理
            if i == 0:
                continue
            text_a = line[text_index]
            if len(line) > text_index + 1:
                label = line[text_index + 1]
            else:
                label = None
            if label == '':
                print(f"脏数据：{text_a}")
                continue
            texts.append(text_a)
            labels.append(label)
        return (texts, labels)
    


class LrPreprocessor(Preprocessor):

    EN_TFIDF_VECTOR_PARAMS = {
            "ngram_range": (1, 3),
            "max_features": 200000,
            "analyzer": "char",
            "min_df": 3,
            "max_df": 0.9,
            "strip_accents": "unicode",
            "use_idf": True,
            "smooth_idf": True,
            "sublinear_tf": True,
            "stop_words": 'english',
            "norm": "l2",
            "lowercase": True
            }

    CN_TFIDF_VECTOR_PARAMS = {
            "ngram_range": (1, 3),
            "max_features": 100000,
            "analyzer": "char",
            "min_df": 1,
            "max_df": 1.0,
            "strip_accents": "unicode",
            "use_idf": True,
            "smooth_idf": True,
            "sublinear_tf": True,
            "norm": "l2",
            "lowercase": True
            }

    """
    训练数据向量化
    """
    @classmethod
    def train_vectorizers(cls, texts, labels, **kwargs):
        # 数据清洗
        clean_texts = cls.texts_clean(texts)

        language_type = kwargs.get("languageType", LanguageType.CN)

        # 获取模型超参数
        params = cls.EN_TFIDF_VECTOR_PARAMS if language_type == LanguageType.EN else cls.CN_TFIDF_VECTOR_PARAMS
        
        # 将原始文档集合转换为 TF-IDF特征矩阵
        vectorizer = TfidfVectorizer(**params)

        # 将离散型的数据转换成 0 到 n − 1 之间的数，这里 n 是一个列表的不同取值的个数，
        # 可以认为是某个特征的所有不同取值的个数。
        labelencoder = LabelEncoder()

        x = vectorizer.fit_transform(clean_texts)

        y = labelencoder.fit_transform(labels)

        logger.info(f"LabelEncoder: lable: {set(labels)}, y: {set(y)}")

        return (vectorizer, labelencoder, x, y)

    """
    评估数据向量化
    """
    @classmethod
    def eval_vectorizers(cls, texts, labels, vectorizer, labelencoder):
        
        clean_texts = cls.texts_clean(texts)

        x = vectorizer.transform(clean_texts)
        print(set(list(labels)))
        y = labelencoder.transform(labels)

        return (x, y)

    """
    文本数据清洗
    返回移除字符串头尾指定的字符序列生成的新字符串。
    """
    @classmethod
    def texts_clean(cls, texts):
       
        if not texts:
            return texts
        return [t.strip('\n\t\r，') for t in texts]


###################

"""
加载pkl和json
"""
def load_object(file_path: str):
    if file_path.endswith('pkl'):
        with open(file_path, "rb") as f:
            return pickle.load(f)
    elif file_path.endswith("json"):
        with open(file_path, "rb") as f:
            return json.load(f)
                
"""
加载模型以及向量
"""
def load_model(model_path, *args, **kwargs):
    model_data = {}
    try:
        
        for k, v in model_path.items():
            if isinstance(v, str):
                # 加载向量
                model_data[k] = load_object(v)
            elif isinstance(v, list):
                # 加载模型，如果是多个模型，采用K折交叉验证
                model_data[k] = []
                for sub in v:
                    model_data[k].append(load_object(sub))
            else:
                raise Exception("invalid model_path value: {}".format(v))
    except Exception as e:
        logger.error(f"load model failed. model_path={model_path}, exception={e}", exc_info=True)
        return None
    return model_data

# 下载模型到本地
def pickle_dump(obj, fp):
    """
    Dump model to local file system.
    Param
       obj: model
       fp: file path
    """
    if not obj:
        return
    # dump to local file system
    with open(fp, 'wb') as f:
        pickle.dump(obj, f)


# 默认的日志目录
def default_logdir() -> str:
    """
    Same default as PyTorch
    """
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    return os.path.join("runs", current_time + "_" + socket.gethostname())



@dataclass
class TrainingArguments:
    """
    TrainingArguments is the subset of the arguments we use in our example scripts

    **which relate to the training loop itself**.

    Using :class:`~transformers.HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on the command line.

    Parameters:
        output_dir (:obj:`str`):
            The output directory where the model predictions and checkpoints will be written.
        overwrite_output_dir (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, overwrite the content of the output directory. Use this to continue training if
            :obj:`output_dir` points to a checkpoint directory.
        do_train (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run training or not.
        do_eval (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run evaluation on the dev set or not.
        do_predict (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run predictions on the test set or not.
        evaluate_during_training (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to run evaluation during training at each logging step or not.
        per_device_train_batch_size (:obj:`int`, `optional`, defaults to 8):
            The batch size per GPU/TPU core/CPU for training.
        per_device_eval_batch_size (:obj:`int`, `optional`, defaults to 8):
            The batch size per GPU/TPU core/CPU for evaluation.
        gradient_accumulation_steps: (:obj:`int`, `optional`, defaults to 1):
            Number of updates steps to accumulate the gradients for, before performing a backward/update pass.
        learning_rate (:obj:`float`, `optional`, defaults to 5e-5):
            The initial learning rate for Adam.
        weight_decay (:obj:`float`, `optional`, defaults to 0):
            The weight decay to apply (if not zero).
        adam_epsilon (:obj:`float`, `optional`, defaults to 1e-8):
            Epsilon for the Adam optimizer.
        max_grad_norm (:obj:`float`, `optional`, defaults to 1.0):
            Maximum gradient norm (for gradient clipping).
        num_train_epochs(:obj:`float`, `optional`, defaults to 3.0):
            Total number of training epochs to perform.
        max_steps (:obj:`int`, `optional`, defaults to -1):
            If set to a positive number, the total number of training steps to perform. Overrides
            :obj:`num_train_epochs`.
        warmup_steps (:obj:`int`, `optional`, defaults to 0):
            Number of steps used for a linear warmup from 0 to :obj:`learning_rate`.
        logging_dir (:obj:`str`, `optional`):
            Tensorboard log directory. Will default to `runs/**CURRENT_DATETIME_HOSTNAME**`.
        logging_first_step (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Wheter to log and evalulate the first :obj:`global_step` or not.
        logging_steps (:obj:`int`, `optional`, defaults to 500):
            Number of update steps between two logs.
        save_steps (:obj:`int`, `optional`, defaults to 500):
            Number of updates steps before two checkpoint saves.
        save_total_limit (:obj:`int`, `optional`):
            If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in
            :obj:`output_dir`.
        no_cuda (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Wherher to not use CUDA even when it is available or not.
        seed (:obj:`int`, `optional`, defaults to 42):
            Random seed for initialization.
        fp16 (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to use 16-bit (mixed) precision training (through NVIDIA apex) instead of 32-bit training.
        fp16_opt_level (:obj:`str`, `optional`, defaults to 'O1'):
            For :obj:`fp16` training, apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. 
        local_rank (:obj:`int`, `optional`, defaults to -1):
            During distributed training, the rank of the process.
        tpu_num_cores (:obj:`int`, `optional`):
            When training on TPU, the mumber of TPU cores (automatically passed by launcher script).
        debug (:obj:`bool`, `optional`, defaults to :obj:`False`):
            When training on TPU, whether to print debug metrics or not.
        dataloader_drop_last (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size)
            or not.
        eval_steps (:obj:`int`, `optional`, defaults to 1000):
            Number of update steps between two evaluations.
        past_index (:obj:`int`, `optional`, defaults to -1):
            Some models like :doc:`TransformerXL <../model_doc/transformerxl>` or :doc`XLNet <../model_doc/xlnet>` can
            make use of the past hidden states for their predictions. 
            If this argument is set to a positive int, the
            ``Trainer`` will use the corresponding output (usually index 2) as the past state 
            and feed it to the model
            at the next training step under the keyword argument ``mems``.
    """
    output_dir: str = field(
        metadata={"help": "The output directory where the model predictions and checkpoints will be written."}
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory."
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    do_predict: bool = field(default=False, metadata={"help": "Whether to run predictions on the test set."})
    evaluate_during_training: bool = field(
        default=False, metadata={"help": "Run evaluation during training at each logging step."},
    )
    per_device_train_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=8, metadata={"help": "Batch size per GPU/TPU core/CPU for evaluation."}
    )
    per_gpu_train_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Deprecated, the use of `--per_device_train_batch_size` is preferred. "
            "Batch size per GPU/TPU core/CPU for training."
        },
    )
    per_gpu_eval_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Deprecated, the use of `--per_device_eval_batch_size` is preferred."
            "Batch size per GPU/TPU core/CPU for evaluation."
        },
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for Adam."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay if we apply some."})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for Adam optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})
    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})
    logging_dir: Optional[str] = field(default_factory=default_logdir, metadata={"help": "Tensorboard log dir."})
    logging_first_step: bool = field(default=False, metadata={"help": "Log and eval the first global_step"})
    logging_steps: int = field(default=500, metadata={"help": "Log every X updates steps."})
    save_steps: int = field(default=500, metadata={"help": "Save checkpoint every X updates steps."})
    save_total_limit: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Limit the total amount of checkpoints."
                "Deletes the older checkpoints in the output_dir. Default is unlimited checkpoints"
            )
        },
    )
    no_cuda: bool = field(default=False, metadata={"help": "Do not use CUDA even when it is available"})
    seed: int = field(default=42, metadata={"help": "random seed for initialization"})
    fp16: bool = field(
        default=False,
        metadata={"help": "Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit"},
    )
    fp16_opt_level: str = field(
        default="O1",
        metadata={
            "help": (
                "For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                "See details at https://nvidia.github.io/apex/amp.html"
            )
        },
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})
    tpu_num_cores: Optional[int] = field(
        default=None, metadata={"help": "TPU: Number of TPU cores (automatically passed by launcher script)"}
    )
    tpu_metrics_debug: bool = field(
        default=False,
        metadata={"help": "Deprecated, the use of `--debug` is preferred. TPU: Whether to print debug metrics"},
    )
    debug: bool = field(default=False, metadata={"help": "Whether to print debug metrics on TPU"})

    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    eval_steps: int = field(default=1000, metadata={"help": "Run an evaluation every X steps."})
    past_index: int = field(
        default=-1,
        metadata={"help": "If >=0, uses the corresponding part of the output as the past state for next step."},
    )

    # Custom codes
    # 以下属性是对源码的修改，以满足 SDK 的需求
    max_seq_length: int = field(default=128, metadata={'help': 'Dataset max seq length.'})

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training (may differ from :obj:`per_gpu_train_batch_size` in distributed training).
        """
        if self.per_gpu_train_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_train_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_train_batch_size` is preferred."
            )
        per_device_batch_size = self.per_gpu_train_batch_size or self.per_device_train_batch_size
        return per_device_batch_size * max(1, self.n_gpu)

    @property
    def eval_batch_size(self) -> int:
        """
        The actual batch size for evaluation (may differ from :obj:`per_gpu_eval_batch_size` in distributed training).
        """
        if self.per_gpu_eval_batch_size:
            logger.warning(
                "Using deprecated `--per_gpu_eval_batch_size` argument which will be removed in a future "
                "version. Using `--per_device_eval_batch_size` is preferred."
            )
        per_device_batch_size = self.per_gpu_eval_batch_size or self.per_device_eval_batch_size
        return per_device_batch_size * max(1, self.n_gpu)

    @cached_property
    @torch_required
    def _setup_devices(self) -> Tuple["torch.device", int]:
        logger.info("PyTorch: setting up devices")
        if self.no_cuda:
            device = torch.device("cpu")
            n_gpu = 0
        elif self.local_rank == -1:
            # if n_gpu is > 1 we'll use nn.DataParallel.
            # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
            # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
            # trigger an error that a device index is missing. Index 0 takes into account the
            # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
            # will use the first GPU in that env, i.e. GPU#1
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            n_gpu = torch.cuda.device_count()
        else:
            # Here, we'll use torch.distributed.
            # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
            torch.distributed.init_process_group(backend="nccl")
            device = torch.device("cuda", self.local_rank)
            n_gpu = 1

        if device.type == "cuda":
            torch.cuda.set_device(device)

        return device, n_gpu

    @property
    @torch_required
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        return self._setup_devices[0]

    @property
    @torch_required
    def n_gpu(self):
        """
        The number of GPUs used by this process.

        Note:
            This will only be greater than one when you have multiple GPUs available but are not using distributed
            training. For distributed training, it will always be 1.
        """
        return self._setup_devices[1]

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(dataclasses.asdict(self), indent=2)

    def to_sanitized_dict(self) -> Dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = dataclasses.asdict(self)
        valid_types = [bool, int, float, str]
        if is_torch_available():
            valid_types.append(torch.Tensor)
        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}


class PredictionOutput(NamedTuple):
    predictions: np.ndarray
    label_ids: Optional[np.ndarray]
    metrics: Optional[Dict[str, float]]
    qps: Optional[float]
    id2label: Optional[Dict[str, str]]


def default_train_args(ouput_dir):
    train_args = TrainingArguments(output_dir=ouput_dir)
    train_args.logging_steps = 5000
    train_args.save_steps = 5000
    train_args.learning_rate = 2e-5
    train_args.num_train_epochs = 5
    train_args.per_device_train_batch_size = 32
    train_args.max_seq_length = 128
    return train_args


def training_callback(
    progress: int,
    measure: Optional[Dict] = None,
    reason: Optional[str] = None,
    en_callback: bool = False,
    code: int = 10000
    ):
    """
    :param progress: 训练进度（-1: 表示训练失败, 大于等于0正常）
    :param measure: 验证集、测试集指标
    :param reason: 训练失败原因
    :param en_callback: 是否进行回调
    :param code: 训练状态码
    :return:
    """
    if not en_callback:
        return
    call_back_url = os.environ.get('MESSAGE_CALL_BACK_URI', None)
    if not call_back_url:
        logger.warning('callback url is None. ')
        return
    logger.info(f'callback url is : {call_back_url} ')

    try:
        request_data = {
            "code": code,
            "message": reason,
            "data": {
                "progress": progress, 
                "measureMetrics": measure 
            }
        }

        response = requests.request(
            "POST", call_back_url, timeout=5, json=request_data)
        if response.status_code == 200:
            logger.info(f'callback success.')
            result_json = response.json()
            logger.info(f"结果数据，响应码：{result_json.get('code', 'code码不存在')} ，响应消息：{result_json.get('message1', '响应消息不存在')}")
        else:
            logger.error(f'callback failed. msg: {response.reason}')
    except:
        logger.error('callback service error.', exc_info=True)

###################

class PredictProcessing:

    @classmethod
    def pre_processing(cls, texts, tokenizer, model_type, **kwargs) -> List[torch.Tensor]:
        """
        :param texts: list of sentences.
        :param tokenizer: word tokenizer
        :param labels: list of label.
        :param kwargs: model_params.
        :return:
        """
        raise NotImplementedError('not implement pre_processing method!')

    @classmethod
    def post_processing(
        cls, logits: np.ndarray,
        id2label: Dict[int, str],
        label_idlist: list=[],
        **kwargs) -> Dict[str, list]:
        """
        :param logits: list of tensor
        :param id2label: dict
        :param label_idlist: list of ground truth label id
        Return:
            {
                'prediction': list，预测结果 list，
                'label': list，真值结果 list
            }
        """
        raise NotImplementedError('not implement post_processing method!')


class LrPredictprocess(PredictProcessing):

    @classmethod
    def pre_processing(cls, texts, tokenizer, **kwargs) :
        """
        :param texts: list of sentences.
        :param labels: list of label.
        :param kwargs: model_params.
        :return:
        """
        try:
            # 预处理，数据清洗，然后文本转向量
            clean_input = [t.strip('\n\t\r，') for t in texts]
            return tokenizer.transform(clean_input)[0]
        except Exception as e:
            logger.error(f'lr preprocess failed!', exc_info=1)
        return None

    @classmethod
    def post_processing(
            cls, 
            prob_array: np.ndarray,
            labelencoder,
            label_idlist: List[int]=[],
            **kwargs
        ) -> Dict[str, list]:
        """
        将 model output 转换成 human readable 的数据结构.
        Params:
            prob_array: np.ndarray, 预测的概率array.
            labelencoder, 
            topn: int, 输出 top 多少的结果
            output_prob: bool, 是否输出 probability

        Return:
            {
                'prediction': list，预测结果 list，
                'label': list，真值结果 list
            }
        其中 prediction list 根据 topn 和 output_prob 的入参不同，输出不同的数据结构.
        - 当 topn == 1 且 output_prob == False 时 -> List[str]
            示例：['label1', 'label2', ...]
        - 当 topn == 1 且 output_prob == True 时 -> List[dict]
            示例：[{'label': 'xxx', 'probability': 0.4}, {}, ...]
        - 当 topn > 1 且 output_prob == False 时 -> List[List[str]]
            示例：[['label1', 'label2', ...], ['label3', 'label4', ...], [], ...]
        - 当 topn > 1 且 output_prob == True 时 -> List[List[Dict]]
            示例：[[{'label': 'xxx', 'probability': 0.4}, {}], [], ...]
        """

        topn = min(max(1, kwargs.get("topn", 1)), prob_array.shape[-1])
        output_prob = kwargs.get("output_prob", False)
        if topn == 1:
            # 输出 top1 结果
            pred_ids = np.argmax(prob_array, axis=1)
            pred_labels = labelencoder.inverse_transform(pred_ids)
            if output_prob:
                items = []
                for index, p_id in enumerate(pred_ids):
                    item = {'label': pred_labels[index], 'probability': prob_array[index][p_id]}
                    items.append(item)
            else:
                items = pred_labels
        else:
            # 输出多个 topn 结果
            pred_ids = np.argsort(prob_array, axis=1)[:, ::-1][:, :topn]
            if output_prob:
                items = []
                for index, p_ids in enumerate(pred_ids):
                    item = [{'label': labelencoder.inverse_transform([s_i])[0], 'probability': prob_array[index][s_i]} for s_i in p_ids]
                    items.append(item)
            else:
                items = [[labelencoder.inverse_transform([i_sort])[0] for i_sort in p_sort] for p_sort in pred_ids]

        # 获取 label 真值标签
        label_list = labelencoder.inverse_transform(label_idlist)
        return {'prediction': items, 'label': label_list}

###################

class LrClassification:
    """lr class for text classification."""

    def __init__(self, task_name: str, is_train: bool = True, **kwargs):
        """NLPTask 统一构造函数.
        Params
            task_name: NLP 任务名
            is_train: bool, 训练模式还是推理模式
            data_dir: NLP 任务对应的数据集
            model_path: NLP 任务的输出模型路径
            train_args: 训练参数
        """
        self.task_name = task_name
        self.data_dir = kwargs.pop('data_dir', '')
        self.model_path = kwargs.pop('model_path', '')
        self.train_args = kwargs.pop('train_args', self._init_training_args(self.model_path))
        self.train_args.kfold = 1
        self.train_args.dual = True
        self.train_args.C = 15.0
        self.train_args.max_iter = 20
        self.train_args.class_weight = 'balanced'
        self.train_args.RANDOM_SEED = 1

        if not self.model_path:
            raise ValueError('NLPTask 训练以及推理模式需要 model_path 参数.')
        
        if is_train:
            task_mode = TaskMode.training
        else:
            task_mode = TaskMode.inference

        self.model_data = self._init_model(task_mode, self.model_path, self.data_dir)

    # 模型初始化
    def _init_model(self, task_mode: TaskMode, model_path: str, data_dir: str):

        if task_mode == TaskMode.training:
            print("-------train before---------")
            # 加载数据
            texts, labels = LrPreprocessor.get_examples(data_dir, Split.train)
            # 特征工程
            # 转换为向量
            vectorizer, labelencoder, x, y = LrPreprocessor.train_vectorizers(texts, labels)
            # 创建模型
            models = []
            for i in range(self.train_args.kfold):
                model = LogisticRegression(
                        # dual=self.train_args.dual,
                        dual=False,
                        C=self.train_args.C,
                        max_iter=self.train_args.max_iter,
                        class_weight=self.train_args.class_weight)
                models.append(model)

            model_data = {
                "vectorizer":vectorizer,
                "labelencoder":labelencoder,
                "lr": models,
                "train_dataset":(x ,y)
            }
            print("-------train---------")
        else:
            
            # 对于推理
            # 对于多个模型遍历
            print("-------inference before---------")

            model_path = {
                "vectorizer": os.path.join(model_path,"vectorizer.pkl"),
                "labelencoder": os.path.join(model_path,"labelencoder.pkl"),
                "lr": [os.path.join(model_path,"lr_"+str(i+1)+".pkl") for i in range(self.train_args.kfold)],
            }
            # 加载模型
            model_data = load_model(model_path)
            print("-------inference---------")

        labelencoder:LabelEncoder = model_data["labelencoder"]
        labels = labelencoder.classes_.tolist()
        id_list = labelencoder.transform(labels).tolist()
        self.id2label = {id_list[i]:labels[i] for i in range(len(id_list))}
        logger.info(f"模型初始化：labels: {labels}, id_list: {id_list} , id2label: {self.id2label}")

        return model_data


    def save_models(self, model_data):
        try:
            if not os.path.exists(self.model_path):
                os.makedirs(self.model_path)
            for k, v in model_data.items():
                if isinstance(v, Iterable):
                    list_paths = []
                    for i, e in enumerate(v):
                        local_path = os.path.join(self.model_path, str(k)+'_'+str(i+1)+'.pkl')
                        pickle_dump(e, local_path)
                else:
                    local_path = os.path.join(self.model_path, str(k)+'.pkl')
                    pickle_dump(v, local_path)
        except Exception as e:
            logger.error(f'error_msg, exception: {e}', exc_info=1)
        return None    

    def _init_training_args(self, model_path: str) -> TrainingArguments:
        r"""
        构造训练参数.
        """
        train_args = default_train_args(model_path)
        return train_args


    def trainning(self,
            vectorizer,
            labelencoder,
            model_list,
            train_data,
            en_callback: bool=False) -> PredictionOutput:
        """
        :training
        :param tuple_data: preprocess data: tokenizer, labelencoder.
        :return: model_info
        """
        try:
            x, y = train_data
            accuracy_list = []
            precision_list = []
            recall_list = []
            f1_list = []
            k = 0
            train_percent = 0
            train_x, train_y = x, y
            val_x, val_y = x, y
            model = model_list[k]
            k += 1
            model.fit(train_x, train_y)
            val_y_pred = model.predict(val_x)
            accuracy = metrics.accuracy_score(val_y, val_y_pred)
            precision = metrics.precision_score(val_y, val_y_pred, average='macro')
            recall = metrics.recall_score(val_y, val_y_pred, average='macro')
            f1 = metrics.f1_score(val_y, val_y_pred, average='macro')
            accuracy_list.append(accuracy)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            train_percent += 30
            training_callback(
                progress=train_percent, 
                measure=None, 
                reason=None,
                en_callback=en_callback
            )
            return {
                'model_data': {
                    'vectorizer': vectorizer,
                    'labelencoder': labelencoder,
                    'lr': model_list
                },
                'metrics': {
                    'accuracy' : sum(accuracy_list) / self.train_args.kfold,
                    'precision': sum(precision_list) / self.train_args.kfold,
                    'recall': sum(recall_list) / self.train_args.kfold,
                    'f1Score': sum(f1_list) / self.train_args.kfold
                }
            }
        except Exception as e:
            logger.error(f'train failed exception: {e}', exc_info=True)
            return None


    # 推理入口
    def predict(self, texts: List[Union[str, List[str]]], **kwargs):
        r"""
        统一预测模块，输入文本，输出结果.

        Params:
            texts: : List[Union[str, List[str]]], 文本序列
            topn: int, 输出 top 多少的结果
            output_prob: bool, 是否输出 probability
        Return：Dict
            {
                'data': []
            }
        """
        predictions = []
        vectorizer = self.model_data["vectorizer"]
        labelencoder = self.model_data["labelencoder"]
        models = self.model_data["lr"]

        processor = LrPredictprocess
        features = processor.pre_processing(texts, vectorizer)
        prob_array = self.cv_pred(models, features)
        predictions = processor.post_processing(prob_array, labelencoder, **kwargs)['prediction']
        return {"data": predictions}

    # 训练入口
    def train(self, data_dir, seed=None, en_callback=False) -> PredictionOutput:

        vectorizer = self.model_data["vectorizer"]
        labelencoder = self.model_data["labelencoder"]
        model_list = self.model_data["lr"]
        train_data = self.model_data["train_dataset"]

        result = self.trainning(vectorizer, labelencoder, model_list, train_data)
        
        if not result:
            logger.error("train error")
            return None
        
        trained_model_data = result["model_data"]

        self.save_models(trained_model_data)

        print("马上要开始评估模型了---------------------------------")
        eval_result = self.test(data_dir, mode=Split.dev)

        return eval_result

    """
        多模型预测
    """
    def cv_pred(self, models, x):
        prob = []
        for model in models:
            prob.append(model.predict_proba(x))
        prob_array = np.array(prob[0])
        nfolds = len(models)
        for i in range(1, nfolds):
            prob_array += np.array(prob[i])
        prob_array /= nfolds
        pred_id = np.argmax(prob_array, axis=1)
        pred_prob = prob_array.max(axis=1)
        return prob_array

    def test(
        self,
        data_dir: str,
        is_log_preds: bool = False,
        mode = Split.test
    ) -> PredictionOutput:
        r"""
        统一测试模块.

        Params:
            data_dir: 数据集目录
            is_log_preds: bool, 是否记录每条测试集的预测结果，default = False
        Return
            PredictionOutput 数据结构中包含：predictions, true label_ids, metrics, eval qps
        """
        logging.info("*** Test ***")
        models = self.model_data["lr"]
        vectorizer = self.model_data["vectorizer"]
        labelencoder = self.model_data["labelencoder"]
        texts, labels = LrPreprocessor.get_examples(data_dir, mode)

        test_dataset = LrPreprocessor.eval_vectorizers(texts, labels, vectorizer, labelencoder)
        x, y = test_dataset
        starttime = time.time()

        prob_array = self.cv_pred(models, x)
        preds = np.argmax(prob_array, axis=1)
        qps = len(test_dataset) / (time.time() - starttime)

        accuracy = metrics.accuracy_score(y, preds)
        precision = metrics.precision_score(y, preds, average='macro')
        recall = metrics.recall_score(y, preds, average='macro')
        f1 = metrics.f1_score(y, preds, average='macro')
        metrics_dict =  {
                    'accuracy' : accuracy,
                    'precision': precision,
                    'recall':recall,
                    'f1Score': f1
                    }
        label2id = {label: idx for idx, label in self.id2label.items()}
        label_ids = [label2id[label] for label in labels]
        eval_result = PredictionOutput(
            predictions=prob_array,
            label_ids=label_ids,
            metrics=metrics_dict,
            qps=qps,
            id2label=self.id2label)
        # 保存预测结果
        if is_log_preds:
            self._log_preditions(texts, labels, eval_result)

        # 保存指标结果
        self._log_metrics(eval_result)

        return eval_result

    def _log_preditions(self, texts, labels, eval_result: PredictionOutput):
        """记录预测结果到文件中，便于做错误分析，列包括，index, origin_text, predictions, labels.
        Params:
            test_dataset: Dataset
            eval_result: PredictionOutput
        """
        pred_probs = eval_result.predictions
        pred_id = np.argmax(pred_probs, axis=1)
        labelencoder = self.model_data["labelencoder"]
        predictions = labelencoder.inverse_transform(pred_id)

        output_test_file = os.path.join(
            self.model_path,
            f"test_results_{self.task_name}.xlsx"
        )
        diff = [1 if str(predictions[_]) != str(labels[_]) else 0 for _ in range(len(labels))]
        df_dict = {'text': texts, 'pred': predictions, 'label': labels, 'diff': diff}
        df = pd.DataFrame(df_dict)
        df.to_excel(output_test_file, index=False)

    def _log_metrics(self, eval_result: PredictionOutput):
        """基于指标结果到日志文件."""
        metrics = eval_result.metrics
        output_eval_file = os.path.join(
            self.model_path, f"test_metrics_{self.task_name}.txt"
        )
        with open(output_eval_file, "w") as writer:
            logger.info(
                "***** Eval results {} *****"
                .format(self.task_name))
            for key, value in metrics.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))


################

TASK_DICT = {
   "lr": LrClassification,
}


class PretrainType:
    bert = "bert"
    roberta = "roberta"

# Supported pretrain type
SUPPORT_PRETRAIN_TYPE = [v for k, v in PretrainType.__dict__.items() if '__' not in k]


def get_pretrain_type(pretrain_name: str):
    max_match = ''
    for p_type in SUPPORT_PRETRAIN_TYPE:
        if p_type in pretrain_name and len(p_type) > len(max_match):
            max_match = p_type
    return max_match



#######################################################

class NLPTrainer:
    """Text classification task."""

    def __init__(self, task_name: str, **kwargs):
        """NLPTask 统一构造函数.
            Params
                task_name: NLP 任务名
                pretrain_name: 任务所需要的预训练模型
                pretrain_path: 需要加载的预训练模型的路径
                data_dir: 任务对应的数据集
                model_path: 输出的模型路径
                fp16: 是否使用混合精度模式
                train_args: 训练参数
        """
        if not get_pretrain_type(kwargs['pretrain_name']):
            raise ValueError(
                '{} not supported right now. '
                'You could use `MSPretrainedModel.support_models()`'
                ' to see which supported models.'.format(kwargs['pretrain_name']))

        if task_name not in TASK_DICT:
            raise ValueError(
                '{} not supported right now. '
                'You could use `NLPTrainer.support_tasks()`'
                ' to see which supported models.'.format(task_name))
        # nlptask 构造函数
        # self.nlptask = TASK_DICT[task_name](task_name, is_train=True, **kwargs)
        self.nlptask = LrClassification(task_name, is_train=True, **kwargs)


    @classmethod
    def support_models(cls):
        r"""
        获取能支持的预训练模型的信息.
        """
        return SUPPORT_PRETRAIN_TYPE

    @classmethod
    def support_tasks(cls):
        r"""
        获取能支持的预训练模型的信息.
        """
        return TASK_DICT.keys()

    def train(self, data_dir: str, model_path: str, en_callback: bool=False) -> PredictionOutput:
        eval_res = self.nlptask.train(data_dir, en_callback=en_callback)
        return eval_res


#######################################################

class NLPPredictor:
    """nlp predictor."""

    def __init__(self, task_name: str, **kwargs):
        """
        NLPTask 统一构造函数.
        Params
            task_name: NLP 任务名
            kwargs: 其他自定义入参
        """
        if task_name not in TASK_DICT:
            raise ValueError(
                '{} not supported right now. '
                'You could use `NLPPredictor.support_tasks()`'
                ' to see which supported models.'.format(task_name))

        # self.nlptask = TASK_DICT[task_name](task_name, is_train=False, **kwargs)
        self.nlptask = LrClassification(task_name, is_train=False, **kwargs)

    @classmethod
    def support_tasks(cls):
        r"""
        获取能支持的预训练模型的信息.
        """
        return TASK_DICT.keys()

    def predict_testdataset(
            self, 
            data_dir: str, 
            is_log_preds: bool=False) -> PredictionOutput:

        test_res = self.nlptask.test(data_dir, is_log_preds=is_log_preds)
        return test_res

    def predict_rest(
            self, 
            json_data: Dict[str, List[str]],
            topn: int = 1,
            output_prob: bool = False,
            bot_mode: str = 'inference') -> Dict[str, list]:
        """
        该接口用于文本流预测，输入是将文本流封装好的 dict，预测结果为 label list.

        Params:
            json_data: Dict[str, List[str]], {
                'text': [str, str, ...]
            }

        Return:
            根据任务和参数的不同，有不同的输出结构.
        """
        texts = json_data.get("text", [])

        pred_result = self.nlptask.predict(
            texts=texts, 
            topn=topn, 
            output_prob=output_prob,
            bot_mode=bot_mode)
        return pred_result


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task_name", help="任务名：task name: ner/text_classification/similarity", type=str, default="")
    parser.add_argument(
        "--pretrain_name", help="模型名：model name:bert-base-chinese", type=str, default="")
    parser.add_argument(
        "--data_dir", help="训练和评估数据集路径 rain and eval data set dir path", type=str, default="")
    parser.add_argument(
        "--model_path", help="模型输出路径：output model dir path", type=str, default="")
    parser.add_argument(
        "--pretrain_path", help="预训练模型路径：pretained model dir path", type=str, default="")
    parser.add_argument(
        "--is_log_preds", help="是否输出预测日志：whether output predictions", type=int, default=0)
    parser.add_argument(
        "--num_train_epochs", help="迭代次数：number of train epochs", type=int, default=3)
    parser.add_argument(
        "--overwrite_output_dir", help="是否重写输出目录：whether overwrite_output_dir", type=int, default=0)
    parser.add_argument(
        "--callback_url", help="训练回调URL callback url", type=str, default="")
    parser.add_argument(
        "--en_callback", help="是否回调训练任务", action='store_true', default=False)

    # 这里的bool是一个可选参数，返回给args的是 args.bool
    args = parser.parse_args()

    # task_name = args.task_name
    task_name = "lr"
    # pretrain_name = args.pretrain_name
    pretrain_name = "bert"
    # data_dir = args.data_dir
    data_dir = "/Users/liguodong/work/data/tnews/temp"
    # model_path = args.model_path
    model_path= "/Users/liguodong/work/train_model"

    pretrain_path = args.pretrain_path
    is_log_preds = args.is_log_preds
    callback_url = args.callback_url
    en_callback = args.en_callback
    train_args = default_train_args(model_path)

    train_args.num_train_epochs = args.num_train_epochs
    train_args.overwrite_output_dir = args.overwrite_output_dir

    trainner = NLPTrainer(
        task_name,
        pretrain_name=pretrain_name,
        pretrain_path=pretrain_path,
        data_dir=data_dir,
        model_path=model_path, 
        fp16=False,
        train_args=train_args,)

    # 设置环境变量
    os.environ['MESSAGE_CALL_BACK_URI'] = callback_url

    trainner.train(data_dir, model_path, en_callback=en_callback)

    print("***********************train done *****************************")


    predictor = NLPPredictor(
        task_name,
        model_path=model_path,
        pretrain_name=pretrain_name)
    res = predictor.predict_testdataset(data_dir, is_log_preds=is_log_preds)

    print(res)
    print("-----------------------------")
    json_data = {"text": ["更改银行卡绑定"]}
    result = predictor.predict_rest(json_data)
    print(result)
    print("over -----------------------------")






