
from nlp_app_samples.pipelines.base_pipeline import BasePipeline
from os.path import join
from sklearn.utils import Bunch
from sklearn.datasets import load_iris
from nlp_app_samples.logger import get_logger

logger = get_logger(__file__)

class LrClassificationPipeline(BasePipeline):

    def __init__(self, task_name:str, **kwargs):
        super().__init__(task_name, **kwargs)

    def preprocesser(self):
        # list(data.target_names)
        print()

    def model_trainer(self):
    
        print()
        

    def get_data(self):
        logger.info(f"从数据源【 {self.datasource} 】获取数据。")
        data = load_iris()
        print(data.target[[10, 25, 50]])
        return data