
from nlp_app_samples.logger import get_logger

logger = get_logger(logger_name = __file__, logging_level = 3)

class BasePipeline:
    """基础训练任务类"""

    def __init__(self, task_name:str, **kwargs):
        self.task_name = task_name
        self.datasource = kwargs.pop('datasource', '')
        
    def get_data(self):
        pass

    def split_dataset(self):
        pass

    def preprocesser(self):
        """
        训练任务预处理
        """
        raise NotImplementedError("训练任务没有实现前置处理方法（preprocesser）")


    def postprocesser(self, eval_result):
        """
        训练任务后置处理
        """
        return eval_result

    def model_trainer(self):
        """
        模型训练
        """
        raise NotImplementedError("训练任务没有实现前置处理方法（preprocesser）")

    
    def model_evaluator(self):
        eval_result = {}
        return eval_result

    def run(self):
        logger.info("模型训练开始。")
        self.get_data()
        self.split_dataset()
        self.preprocesser()
        logger.info("模型前置处理开始。")
        self.model_trainer()
        eval_result = self.model_evaluator()
        logger.info("模型后置处理开始。")
        self.postprocesser(eval_result)
        logger.info("训练任务结束")