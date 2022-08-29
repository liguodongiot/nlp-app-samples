#! -*- coding: utf-8 -*-
import logging
import random
import numpy as np
import os
import torch
from typing import NamedTuple, List, Union
import time
import pandas as pd
from transformers import EvalPrediction
from transformers import AutoConfig, AutoTokenizer, BertTokenizer
# from transformers import InputFeatures as InputFeatures_CLS
from typing import Callable, Dict, Optional, List
from torch.utils.data.dataset import Dataset
from transformers import set_seed
from textbrewer import TrainingConfig, DistillationConfig
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import onnxruntime

from pathlib import Path
import transformers
from transformers.onnx import FeaturesManager
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification


from tests.iter_version_dev.V1_0_0.common import (
    PretrainType, TaskMode, InferMode, 
    TrainType, DistilType, init_strategy,
    SUPPORT_PRETRAIN_TYPE, get_pretrain_type
)

from tests.iter_version_dev.V1_0_0.common_dict import (
    TOKENIZER_FUNC,
    MODEL_FUNC, PREDICT_PROCESSOR
)
from tests.iter_version_dev.V1_0_0.trainer import Trainer

from tests.iter_version_dev.V1_0_0.train_inference_io import (
    PredictionOutput, 
    TrainingArguments,
    default_train_args,
    default_distil_train_args,
    default_distil_config_args
)
from tests.iter_version_dev.V1_0_0.train_inference_io import InputFeatures as InputFeatures_NER
from tests.iter_version_dev.V1_0_0.train_inference_io import InputFeatures_CLS

from tests.iter_version_dev.V1_0_0.processor_dict import DATA_PROCESSOR



logger = logging.getLogger(__name__)

# 加载onnx模型
def load_onnx_model_session(path):
    onnx_path = path + "/bert.onnx"
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(onnx_path, sess_options)
    return session

# 3. 训练任务基类

class NLPTask:
    """Base class for nlp task."""

    def __init__(self, task_name: str, is_train: bool = True, **kwargs):
        """NLPTask 统一构造函数.
        Params
            task_name: NLP 任务名
            is_train: bool, 训练模式还是推理模式
            pretrain_name: 预训练模型名称
            pretrain_type: 任务所需要的预训练模型类型
            pretrain_path: 预训练模型的路径
            data_dir: NLP 任务对应的数据集
            model_path: NLP 任务的输出模型路径
            fp16: 是否使用混合精度模式
            use_cuda: 是否使用GPU
            train_type: 训练类型，常规训练模式(general)还是蒸馏训练模式(distil)
            distil_type: student蒸馏类型，支持6种结构: T3, T3_Tiny, T4, T4_Tiny, T6, T6_Tiny
            train_args: 训练参数
            distil_args: 蒸馏配置参数
            teacher_pretrain_path：teacher预训练模型的路径
        """
        self.task_name = task_name
        self.pretrain_name = kwargs.pop('pretrain_name', '')
        self.pretrain_path = kwargs.pop('pretrain_path', '')
        self.pretrain_type = kwargs.pop('pretrain_type', None)
        self.data_dir = kwargs.pop('data_dir', '')
        self.model_path = kwargs.pop('model_path', '')
        self.fp16 = kwargs.pop('fp16', False)
        self.use_cuda = kwargs.pop('use_cuda', True)
        self.train_type = kwargs.pop('train_type', TrainType.general)
        self.distil_type = kwargs.pop('distil_type', DistilType.T4)
        self.train_args = kwargs.pop('train_args', self._init_training_args(self.model_path, self.fp16))
        self.distil_args = kwargs.pop('distil_args', self._init_distilling_args(self.distil_type))
        self.teacher_pretrain_path = kwargs.pop('teacher_pretrain_path', '')
        self.trainer = None
        self.infer_mode = kwargs.pop('infer_mode', InferMode.general)

        if not self.pretrain_type:
            self.pretrain_type = get_pretrain_type(self.pretrain_name)

        if is_train:
            task_mode = TaskMode.training
            if self.train_type == TrainType.general:
                if not self.pretrain_path:
                    raise ValueError('NLPTask 常规训练模式需要 pretrain_path 参数.')
            else:
                if not self.teacher_pretrain_path:
                    raise ValueError('NLPTask 蒸馏训练模式需要 teacher_pretrain_path 参数.')
        else:
            task_mode = TaskMode.inference
            if not self.model_path:
                raise ValueError('NLPTask 推理模式需要 model_path 参数.')

        # 根据 task mode 选择不同路径
        if task_mode == TaskMode.training:
            if (
                not self.train_args.overwrite_output_dir
                and os.path.exists(self.model_path)
                and os.listdir(self.model_path)):
                path = self.model_path
            else:
                path = self.pretrain_path
        else:
            path = self.model_path

        # 判断是常规训练模式 or 蒸馏训练模式
        if self.train_type == TrainType.general:
            self.tokenizer = self._init_tokenizer(task_mode, path)
            self.config = self._init_model_config(task_mode, path)

            # 判断是常规推理模式 or onnx推理模式
            if self.infer_mode == InferMode.general:
                self.model = self._init_model(task_mode, path, self.config)
            else:
                self.model = load_onnx_model_session(path)

        else:
            # teacher model
            self.tokenizer = self._init_tokenizer(task_mode, self.teacher_pretrain_path)
            self.teacher_config = self._init_model_config(task_mode, self.teacher_pretrain_path)
            self.teacher_model = self._init_model(task_mode, self.teacher_pretrain_path, self.teacher_config)
            # student model
            base_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
            self.config_path = os.path.join(base_dir, 'msnlp', 'distil', 'config', self.distil_type)
            self.config = self._init_model_config(task_mode, self.config_path)
            if init_strategy[self.distil_type] == 'random':
                self.model = self._init_model(task_mode=task_mode, path='', config=self.config)
            else:
                self.model = self._init_model(
                    task_mode=task_mode,
                    path=self.teacher_pretrain_path,
                    config=self.config)

    @classmethod
    def support_models(cls):
        r"""
        获取能支持的预训练模型的信息.
        """
        return SUPPORT_PRETRAIN_TYPE

    def _init_tokenizer(self, task_mode: TaskMode, path: str):
        # init tokenizer
        # t_func = TOKENIZER_FUNC[self.pretrain_type]
        # tokenizer = t_func.from_pretrained(path)
        # add_special_tokens to False
        tokenizer = BertTokenizer.from_pretrained(pretrained_model_name_or_path = path, add_special_tokens = False)
        return tokenizer

    def _init_model_config(self, task_mode: TaskMode, path: str):
        # init config
        if task_mode == TaskMode.training:

            # text_classification_bert
            processor = DATA_PROCESSOR[self.task_name]()

            label2id = processor.get_label2id(self.data_dir)
            id2label = processor.get_id2label(self.data_dir)
            num_labels = len(label2id)
            config = AutoConfig.from_pretrained(
                path,
                num_labels=num_labels,
                id2label=id2label,
                label2id=label2id,
                output_hidden_states=True
            )
        else:
            config = AutoConfig.from_pretrained(path)
            id2label = { int(i): label for i, label in config.id2label.items() }

        self.id2label = id2label
        return config

    def _init_model(self, task_mode: TaskMode, path: str, config: str):
        m_func = MODEL_FUNC[self.task_name]
        if path:
            model = m_func.from_pretrained(path, config=config)
        else:
            model = m_func.from_config(config=config)

        if task_mode == TaskMode.inference:
            device = torch.device("cuda" if torch.cuda.is_available() and self.use_cuda else "cpu")
            model.to(device)
            # if self.fp16:
            #     from apex import amp
            #     model = amp.initialize(model, opt_level="O1")
            model.eval()

        return model

    def _init_training_args(self, model_path: str, fp16: bool) -> TrainingArguments:
        r"""
        构造训练参数.
        """
        if self.train_type == TrainType.general:
            train_args = default_train_args(model_path)
            train_args.fp16 = fp16
        else:
            train_args = default_distil_train_args(model_path)
        return train_args
    
    def _init_distilling_args(self, distil_type: str) -> DistillationConfig:
        r"""
        构造蒸馏参数.
        """
        distil_args = default_distil_config_args(distil_type)
        return distil_args

    def test(self, data_dir) -> PredictionOutput:
        raise NotImplementedError('nlp task not implement `test` method!')

    def train(self, data_dir, seed=None) -> PredictionOutput:
        raise NotImplementedError('nlp task not implement `train` method!')

    def uni_train(
        self,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset]=None,
        compute_metrics_fn: Optional[Callable[[EvalPrediction], Dict]]=None,
        seed: int=None,
        en_callback: bool=False) -> PredictionOutput:
        r"""
        统一训练模块.

        Params:
            train_dataset: 训练集
            eval_dataset: 验证集
            compute_metrics_fn: 任务指标计算 function

        Return
            PredictionOutput 数据结构中包含：predictions, true label_ids, metrics, eval qps
        """
        if self.train_type == TrainType.general:
            return self.general_train(train_dataset=train_dataset,
                                      eval_dataset=eval_dataset,
                                      compute_metrics_fn=compute_metrics_fn,
                                      seed=seed,
                                      en_callback=en_callback)
        else:
            return self.distil_train(train_dataset=train_dataset,
                                     eval_dataset=eval_dataset,
                                     compute_metrics_fn=compute_metrics_fn,
                                     seed=seed)

    def general_train(
            self,
            train_dataset: Dataset,
            eval_dataset: Optional[Dataset]=None,
            compute_metrics_fn: Optional[Callable[[EvalPrediction], Dict]]=None,
            seed: int=None,
            en_callback: bool=False) -> PredictionOutput:
        r"""
        统一训练模块.

        Params:
            train_dataset: 训练集
            eval_dataset: 验证集
            compute_metrics_fn: 任务指标计算 function

        Return
            PredictionOutput 数据结构中包含：predictions, true label_ids, metrics, eval qps
        """
        # 是否覆盖训练
        if (
            not self.train_args.overwrite_output_dir
            and os.path.exists(self.model_path)
            and os.listdir(self.model_path)
        ):
            # 不覆盖训练
            logger.info(f'Model already exists in ({self.model_path}), pass trainer, load model to evaluate.')
            # 根据已训练好的 model 初始化 trainer
            trainer = Trainer(
                model=self.model,
                args=self.train_args,
                compute_metrics=compute_metrics_fn,)
        else:
            logger.info(f"准备开始训练模型，训练参数：{self.train_args}")
            if not seed:
                seed = random.randint(0, 2020)
            set_seed(seed)

            # Initialize our Trainer
            trainer = Trainer(
                model=self.model,
                args=self.train_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics_fn,
            )
            # 训练
            trainer.train(self.model_path, en_callback=en_callback)
            trainer.save_model()
            self.tokenizer.save_pretrained(self.model_path)


            # 转换成ONNX
            # load config
            model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(self.model,
                                                                                           feature="default")
            onnx_config = model_onnx_config(self.model.config)

            # export
            onnx_inputs, onnx_outputs = transformers.onnx.export(
                preprocessor=self.tokenizer,
                model=self.model,
                config=onnx_config,
                opset=13,
                output=Path(self.model_path + "/" + "trfs-model.onnx")
            )



        # Evaluation
        logger.info("*** Evaluate ***")
        trainer.compute_metrics = compute_metrics_fn
        starttime = time.time()
        eval_result = trainer.predict(test_dataset=eval_dataset)
        qps = len(eval_dataset) / (time.time() - starttime)
        metrics = eval_result.metrics
        output_eval_file = os.path.join(
            self.model_path, f"eval_results_{self.task_name}.txt"
        )
        with open(output_eval_file, "w") as writer:
            logger.info(
                "***** Eval results {} *****"
                .format(self.task_name))
            for key, value in metrics.items():
                logger.info("  %s = %s", key, value)
                writer.write("%s = %s\n" % (key, value))

        self.trainer = trainer
        return PredictionOutput(
            predictions=eval_result.predictions,
            label_ids=eval_result.label_ids,
            metrics=eval_result.metrics,
            qps=qps,
            id2label=self.id2label)

    def distil_train(
            self,
            train_dataset: Dataset,
            eval_dataset: Optional[Dataset] = None,
            compute_metrics_fn: Optional[Callable[[EvalPrediction], Dict]] = None,
            seed: int = None) -> PredictionOutput:
        print("----distil_train-----")
        pass


    def uni_test(
        self,
        test_dataset: Dataset,
        compute_metrics_fn: Optional[Callable[[EvalPrediction], Dict]] = None,
        is_log_preds: bool = False,
    ) -> PredictionOutput:
        r"""统一测试模块.

        Params:
            test_dataset: 测试集
            compute_metrics_fn: 任务指标计算 function
            is_log_preds: bool, 是否记录每条测试集的预测结果，default = False

        Return
            PredictionOutput 数据结构中包含：predictions, true label_ids, metrics, eval qps
        """
        logging.info("*** Test ***")
        if not self.trainer:
            self.trainer = Trainer(
                model=self.model,
                args=self.train_args,
                compute_metrics=compute_metrics_fn,)

        starttime = time.time()
        eval_result = self.trainer.predict(test_dataset=test_dataset)
        qps = len(test_dataset) / (time.time() - starttime)

        # 保存预测结果
        if is_log_preds:
            self._log_preditions(test_dataset, eval_result)

        # 保存指标结果
        self._log_metrics(eval_result)

        return PredictionOutput(
            predictions=eval_result.predictions,
            label_ids=eval_result.label_ids,
            metrics=eval_result.metrics,
            qps=qps,
            id2label=self.id2label)

    def _log_preditions(self, test_dataset: Dataset, eval_result: PredictionOutput):
        """记录预测结果到文件中，便于做错误分析，列包括，index, origin_text, predictions, labels.

        Params:
            test_dataset: Dataset
            eval_result: PredictionOutput
        """
        processor = PREDICT_PROCESSOR[self.task_name]
        predictions_logits = eval_result.predictions
        input_list = []
        text_list = []
        label_idlist = []
        filter_tokens = ['[SEP]', '[CLS]', '[PAD]']
        for f in test_dataset.features:
            input_list.append(f.input_ids)
            tokens = self.tokenizer.convert_ids_to_tokens(f.input_ids)
            tokens = [t for t in tokens if t not in filter_tokens]
            text_list.append(''.join(tokens))
            if isinstance(f, InputFeatures_CLS):
                label_idlist.append(f.label)
            elif isinstance(f, InputFeatures_NER):
                label_idlist.append(f.label_ids)
        post_res = processor.post_processing(
            predictions_logits,
            self.config.id2label,
            label_idlist=label_idlist,
            tokenizer=self.tokenizer,
            input_texts=text_list,
            input_list=input_list)
        predictions = post_res['prediction']
        labels = post_res['label']
        if not labels:
            labels = [None] * len(predictions)

        output_test_file = os.path.join(
            self.model_path,
            f"test_results_{self.task_name}.xlsx"
        )
        diff = [1 if str(predictions[_]) != str(labels[_]) else 0 for _ in range(len(labels))]
        df_dict = {'text': text_list, 'pred': predictions, 'label': labels, 'diff': diff}
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
        processor = PREDICT_PROCESSOR[self.task_name]
        # 推理阶段使用动态长度
        if isinstance(texts[0],list):
            # 文本匹配任务，title content等类型的数据场景
            text_len = []
            for text in texts:
                text_len.extend([len(text[0]), len(text[1])])
            max_dyna_length = max(text_len)
        else:
            max_dyna_length = max([len(text) for text in texts])
        max_seq_length = max_dyna_length+2 if max_dyna_length+2 < self.train_args.max_seq_length \
            else self.train_args.max_seq_length
        features = processor.pre_processing(
            texts,
            self.tokenizer,
            self.pretrain_type,
            id2label=self.config.id2label,
            max_seq_length=max_seq_length,
            task_name=self.task_name)
        device = torch.device(
            "cuda" if torch.cuda.is_available() and self.use_cuda else "cpu")
        features = tuple(t.to(device) for t in features)

        if self.infer_mode == InferMode.general:
            inputs = {
                'input_ids': features[0],
                'attention_mask': features[1],
                'token_type_ids': features[2]}
            with torch.no_grad():
                outputs = self.model(**inputs)[0]
                logits = outputs.detach().cpu().numpy()
        else:
            inputs = {
                'input_ids':  features[0].cpu().numpy(),
                'attention_mask': features[1].cpu().numpy(),
                'token_type_ids': features[2].cpu().numpy()}
            with torch.no_grad():
                logits = self.model.run(None, inputs)[0]

        input_list = features[0].cpu().numpy().tolist()
        predictions = processor.post_processing(
            logits,
            self.id2label,
            tokenizer=self.tokenizer,
            input_list=input_list,
            input_texts=texts,
            **kwargs)['prediction']
        return {"data": predictions}



