
from tests.iter_version_dev.V1_0_0.task_bask import NLPTask
from tests.iter_version_dev.V1_0_0.train_inference_io import PredictionOutput
from tests.iter_version_dev.V1_0_0.dataset_ner import NerIterableDataset, NerDataset
from tests.iter_version_dev.V1_0_0.common import Split
from tests.iter_version_dev.V1_0_0.metric import compute_metrics_fn, ner_metrics

class TokenClassification(NLPTask):
    """Text classification task."""

    def __init__(self, task_name: str, is_train: bool = True, **kwargs):
        super().__init__(task_name, is_train=is_train, **kwargs)

    def train(self, data_dir, seed=None, en_callback=False) -> PredictionOutput:

        # 构造数据集
        label2id = self.config.label2id

        train_dataset = NerDataset(
            data_dir=data_dir,
            tokenizer=self.tokenizer,
            label_map=label2id,
            pretrain_type=self.pretrain_type,
            max_seq_length=self.train_args.max_seq_length,
            mode=Split.train,
        )

        eval_dataset = NerDataset(
            data_dir=data_dir,
            tokenizer=self.tokenizer,
            label_map=label2id,
            pretrain_type=self.pretrain_type,
            max_seq_length=self.train_args.max_seq_length,
            mode=Split.dev,
        )

        # 交付统一训练
        eval_result = self.uni_train(
            train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics_fn=compute_metrics_fn,
            seed=seed,
            en_callback=en_callback)

        return eval_result

    def test(self, data_dir:str, is_log_preds: bool = False) -> PredictionOutput:
        # 构造 test 数据集
        label2id = self.config.label2id

        test_dataset = NerDataset(
            data_dir,
            tokenizer=self.tokenizer,
            label_map=label2id,
            pretrain_type=self.pretrain_type,
            max_seq_length=self.train_args.max_seq_length,
            mode=Split.test,
        )

        # 提交统一测试
        eval_result = self.uni_test(
            test_dataset,
            compute_metrics_fn=compute_metrics_fn,
            is_log_preds=is_log_preds)

        predictions = eval_result.predictions
        label_ids = eval_result.label_ids
        metrics = ner_metrics(predictions, label_ids, self.config.id2label)
        for key, value in metrics.items():
            eval_result.metrics[key] = value
        return eval_result


