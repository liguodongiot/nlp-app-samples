

from nlp_app_samples import logger
from nlp_app_samples.logger import get_logger
from nlp_app_samples.pipelines.base_pipeline import BasePipeline
from nlp_app_samples.default_args import MetricParameter
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


logger.get_logger(__file__)

# TODO Pytorch实现多层感知机
class MlpClassificationPipeline(BasePipeline):

    def __init__(self, task_name: str, **kwargs):
        # number of subprocesses to use for data loading
        self.num_workers = 0
        # how many samples per batch to load
        self.batch_size = 20
        # percentage of training set to use as validation
        self.valid_size = 0.2
        super().__init__(task_name, **kwargs)

    def get_data(self):
        logger.info(f"从数据源【 {self.datasource} 】获取数据。")
        # choose the training and testing datasets
        train_data = datasets.MNIST(root = '/Users/liguodong/work/data', train = True, download = True, transform = transform)
        test_data = datasets.MNIST(root = '/Users/liguodong/work/data', train = False, download = True, transform = transform)

    def split_dataset(self):
        if self.dataset is None:
            logger.warn("数据集为空，请先确定是否存在数据集。")
            return

            
    def preprocesser(self):
        pre_process_result = {}
        return pre_process_result

    def model_trainer(self, pre_process_result):
        logger.info()


