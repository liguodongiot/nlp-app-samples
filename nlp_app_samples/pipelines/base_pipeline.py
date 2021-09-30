
from nlp_app_samples.logger import get_logger
from nlp_app_samples.default_args import default_model_parameter

logger = get_logger(logger_name = __file__, logging_level = 3)

class BasePipeline:
    """
    基础训练任务类
    """

    def __init__(self, task_name:str, **kwargs):
        self.task_name = task_name
        self.datasource = kwargs.pop('datasource', None)
        self.model_ouput_path = kwargs.pop('model_ouput_path', './')
        self.hyperparameter = default_model_parameter(model_ouput_path = self.model_ouput_path)
        self.dataset = None
        self.training_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.pretrain_model = None
        self.model = None


    def get_data(self):
        pass

    def split_dataset(self):
        pass

    def preprocesser(self):
        """
        训练任务预处理
        """
        raise NotImplementedError("训练任务没有实现前置处理方法（preprocesser）")

    def model_trainer(self, pre_process_result):
            """
            模型训练
            """
            raise NotImplementedError("训练任务没有实现前置处理方法（preprocesser）")

    def postprocesser(self, eval_result):
        """
        训练任务后置处理
        """
        return eval_result

    
    def model_evaluator(self):
        eval_result = {}
        return eval_result

    def run(self):
        logger.info(f"任务名为【{self.task_name}】的模型训练任务开始。")
        self.get_data()
        self.split_dataset()
        pre_process_result = self.preprocesser()
        logger.info("模型前置处理开始。")
        self.model_trainer(pre_process_result)
        eval_result = self.model_evaluator()
        logger.info("模型后置处理开始。")
        self.postprocesser(eval_result)
        logger.info(f"任务名为【{self.task_name}】的模型训练任务结束。")