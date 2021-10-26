import logging

from tests.iter_version_dev.V1_0_0.task_bask import NLPTask
from tests.iter_version_dev.V1_0_0.train_inference_io import PredictionOutput
from tests.iter_version_dev.V1_0_0.dataset_custom import ClassificationIterableDataset,ClassificationDataset
from tests.iter_version_dev.V1_0_0.common import Split
from tests.iter_version_dev.V1_0_0.common_dict import METRICS_DICT
from tests.iter_version_dev.V1_0_0.metric import compute_metrics_fn

logger = logging.getLogger(__name__) 

# 3. 训练任务类（Bert文本分类）

class ClassificationTask(NLPTask):
    """
    Text classification task.
    """
    def __init__(self, task_name: str, is_train: bool = True, **kwargs):
        super().__init__(task_name, is_train=is_train, **kwargs)
    

    def train(self, data_dir, seed=None, en_callback=False) -> PredictionOutput:
        # 构造数据集
        id2label = self.config.id2label
        label_list = [label for _, label in id2label.items()]
        train_dataset = ClassificationDataset(
            data_dir,
            tokenizer=self.tokenizer,
            label_list=label_list,
            mode=Split.train,
            max_seq_length=self.train_args.max_seq_length,
            task_name=self.task_name)
        # train_dataset = ClassificationIterableDataset(
        #     data_dir,
        #     tokenizer=self.tokenizer,
        #     label_list=label_list,
        #     mode=Split.train,
        #     max_seq_length=self.train_args.max_seq_length,
        #     task_name=self.task_name)

        eval_dataset = ClassificationDataset(
            data_dir,
            tokenizer=self.tokenizer,
            label_list=label_list,
            mode=Split.dev,
            max_seq_length=self.train_args.max_seq_length,
            task_name=self.task_name)
        # eval_dataset = ClassificationIterableDataset(
        #     data_dir,
        #     tokenizer=self.tokenizer,
        #     label_list=label_list,
        #     mode=Split.dev,
        #     max_seq_length=self.train_args.max_seq_length,
        #     task_name=self.task_name)

        # 评估指标
        metrics_fn = METRICS_DICT[self.task_name] if self.task_name in METRICS_DICT else compute_metrics_fn
        
        # 交付统一训练
        eval_result = self.uni_train(
            train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics_fn=metrics_fn,
            seed=seed,
            en_callback=en_callback)

        return eval_result


    def test(
        self,
        data_dir:str,
        is_log_preds: bool = False,
        mode = Split.test) -> PredictionOutput:
        
        # 构造 test 数据集
        id2label = self.config.id2label
        label_list = [label for _, label in id2label.items()]
        test_dataset = ClassificationDataset(
            data_dir,
            tokenizer=self.tokenizer,
            label_list=label_list,
            max_seq_length=self.train_args.max_seq_length,
            task_name=self.task_name,
            mode=mode)

        # test_dataset = ClassificationIterableDataset(
        #     data_dir,
        #     tokenizer=self.tokenizer,
        #     label_list=label_list,
        #     max_seq_length=self.train_args.max_seq_length,
        #     task_name=self.task_name,
        #     mode=mode)
        metrics_fn = METRICS_DICT[self.task_name] if self.task_name in METRICS_DICT else compute_metrics_fn
        # 提交统一测试
        eval_result = self.uni_test(test_dataset,
            compute_metrics_fn=metrics_fn,
            is_log_preds=is_log_preds)

        return eval_result















