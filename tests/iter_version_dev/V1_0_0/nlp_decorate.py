from typing import List, Dict

from tests.iter_version_dev.V1_0_0.common import PretrainType, TaskName, SUPPORT_PRETRAIN_TYPE, get_pretrain_type
from tests.iter_version_dev.V1_0_0.common_dict import TASK_DICT
from tests.iter_version_dev.V1_0_0.train_inference_io import PredictionOutput
from tests.iter_version_dev.V1_0_0.task_classification_bert import ClassificationTask
from tests.iter_version_dev.V1_0_0.task_ner import TokenClassification
# 2.训练预测装饰类

TASK_DICT = {
    TaskName.text_classification_bert: ClassificationTask,
    TaskName.text_similarity_bert: ClassificationTask,
    TaskName.ner: TokenClassification,
    TaskName.text_classification_multi_bert: ClassificationTask,
}

class NLPTrainer:
    """
        Text classification task.
    """

    def __init__(self, task_name: str, **kwargs):
        """
        NLPTask 统一构造函数.
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
        self.nlptask = TASK_DICT[task_name](task_name, is_train=True, **kwargs)


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
    """
    nlp predictor.
    """

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

        self.nlptask = TASK_DICT[task_name](task_name, is_train=False, **kwargs)

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



