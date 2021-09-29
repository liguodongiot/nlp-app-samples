from nlp_app_samples.pipelines.lr_classification_pipeline import LrClassificationPipeline

from nlp_app_samples.logger import get_logger
logger = get_logger(logger_name = __file__, logging_level = 3)

args = {'datasource': 'sklearn.datasets.data',
        'model_ouput_path': '/Users/liguodong/work/data/model', 
        }

lr = LrClassificationPipeline(task_name='lr_classification', **args)

result = lr.get_data()

df = result['data'].join(result['target'])

logger.info(df.sample(5))


lr.run()

