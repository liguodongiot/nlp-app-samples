

from nlp_app_samples import logger
from nlp_app_samples.logger import get_logger
from nlp_app_samples.pipelines.base_pipeline import BasePipeline
from nlp_app_samples.default_args import MetricParameter

logger.get_logger(__file__)

class MlpClassificationPipeline(BasePipeline):

    def __init__(self, task_name: str, **kwargs):
        super().__init__(task_name, **kwargs)

    def preprocesser(self):
        pre_process_result = {}
        return pre_process_result

    def model_trainer(self, pre_process_result):
        logger.info()


