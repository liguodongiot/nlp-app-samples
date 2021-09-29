from nlp_app_samples.pipelines.lr_classification_pipeline import LrClassificationPipeline

from nlp_app_samples.logger import get_logger
logger = get_logger(logger_name = __file__, logging_level = 3)

args = {'datasource': 'sklearn.datasets.data'}

lr = LrClassificationPipeline(task_name='lr_classification', **args)

result = lr.get_data()

print(result)

lr.run()

