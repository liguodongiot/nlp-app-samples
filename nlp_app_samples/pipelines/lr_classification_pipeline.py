from os.path import join
from nlp_app_samples.logger import get_logger
from nlp_app_samples.pipelines.base_pipeline import BasePipeline
from nlp_app_samples.default_args import MetricParameter
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import joblib

logger = get_logger(__file__)

class LrClassificationPipeline(BasePipeline):

    def __init__(self, task_name:str, **kwargs):
        super().__init__(task_name, **kwargs)

    def get_data(self):
        logger.info(f"从数据源【 {self.datasource} 】获取数据。")
        iris = load_iris(as_frame=True)
        print(iris.target[[10, 25, 50]])
        df = iris['data'].join(iris['target'])
        logger.info(f"数据集样例输出：{df.sample(5)}")
        self.dataset = df

    def split_dataset(self):
        if self.dataset is None:
            logger.warn("数据集为空，请先确定是否存在数据集。")
            return
        logger.info(f"数据集大小: { self.dataset.shape }")
        self.training_dataset, valid_test_dataset = train_test_split(self.dataset, train_size=0.7, stratify= self.dataset['target'])
        self.valid_dataset, self.test_dataset = train_test_split(valid_test_dataset, test_size = 0.33, stratify= valid_test_dataset['target'])
        logger.info(f"训练集大小: {self.training_dataset.shape} ，校验集大小: {self.valid_dataset.shape} ，测试集大小: {self.test_dataset.shape} ")

    def preprocesser(self):
        X, y = self.training_dataset.iloc[:,[0,1,2,3]], self.training_dataset.loc[:,('target')]
        return X,y

    def model_trainer(self, pre_process_result):
        X, y = pre_process_result
        self.model = LogisticRegression(multi_class = 'ovr', solver='liblinear') 
        self.model.fit(X, y)
    

    def model_evaluator(self):
        X_valid_true, y_valid_true = self.valid_dataset.iloc[:,[0,1,2,3]], self.valid_dataset.iloc[:,[4]]
        X_test_true, y_test_true  = self.test_dataset.iloc[:,[0,1,2,3]], self.test_dataset.iloc[:,[4]]
        
        y_valid_predict = self.model.predict(X_valid_true)
        y_test_predict = self.model.predict(X_test_true)

        eval_result = {}

        valid_accuracy, valid_precision, valid_recall, valid_f1 = accuracy_score(y_valid_true, y_valid_predict),\
                precision_score(y_valid_true, y_valid_predict,average='micro'), \
                recall_score(y_valid_true, y_valid_predict, average='micro'),\
                f1_score(y_valid_true, y_valid_predict, average='micro')

        valid_metrics = []
                
        # 序列化时对中文默认使用的ascii编码，因此转换时需要ensure_ascii = True
        valid_metrics.append(MetricParameter(key = '准确率', value = round(valid_accuracy, 3), description = '准确率，值越高表示效果越好').to_json(ensure_ascii = False))
        valid_metrics.append(MetricParameter(key = '精确率', value = round(valid_precision, 3), description = '精确率，值越高表示效果越好').to_json(ensure_ascii = False))
        valid_metrics.append(MetricParameter(key = '召回率', value = round(valid_recall, 3), description = '召回率，值越高表示效果越好').to_json(ensure_ascii = False))
        valid_metrics.append(MetricParameter(key = 'F1值', value = round(valid_f1, 3), description = 'F1值，值越高表示效果越好').to_json(ensure_ascii = False))
        eval_result['valid_metrics'] = valid_metrics

        test_accuracy, test_precision, test_recall, test_f1 = accuracy_score(y_test_true, y_test_predict),\
                precision_score(y_test_true, y_test_predict,average='micro'), \
                recall_score(y_test_true, y_test_predict, average='micro'),\
                f1_score(y_test_true, y_test_predict, average='micro')

        test_metrics = [] 
        test_metrics.append(MetricParameter(key = '准确率', value = round(test_accuracy, 3), description = '准确率，值越高表示效果越好').to_json(ensure_ascii = False))
        test_metrics.append(MetricParameter(key = '精确率', value = round(test_precision, 3), description = '精确率，值越高表示效果越好').to_json(ensure_ascii = False))
        test_metrics.append(MetricParameter(key = '召回率', value = round(test_recall, 3), description = '召回率，值越高表示效果越好').to_json(ensure_ascii = False))
        test_metrics.append(MetricParameter(key = 'F1值', value = round(test_f1, 3), description = 'F1值，值越高表示效果越好').to_json(ensure_ascii = False))

        eval_result['test_metrics'] = test_metrics

        return eval_result

    
    def postprocesser(self, eval_result):
        logger.info(f"评估指标结果：{eval_result}")
        joblib.dump(self.model, join(self.model_ouput_path, "lr_classification.joblib"))
