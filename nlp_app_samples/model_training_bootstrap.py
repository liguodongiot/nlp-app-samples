from nlp_app_samples.pipelines.lr_classification_pipeline import LrClassificationPipeline
from nlp_app_samples.configs import TASK_DICT
from nlp_app_samples.logger import get_logger
logger = get_logger(logger_name = __file__, logging_level = 3)

args = {'datasource': 'sklearn.datasets.data',
        'model_ouput_path': '/Users/liguodong/work/data/model', 
        }

lr = TASK_DICT['LR'](task_name='lr_classification', **args)


lr.run()

