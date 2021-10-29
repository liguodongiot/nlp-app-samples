
from torch.utils.data.dataset import T
from transformers import AutoTokenizer, BertTokenizer
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification
)
from tests.iter_version_dev.V1_0_0.classification_lr import TASK_DICT

from tests.iter_version_dev.V1_0_0.processor_bert_classification import ClassificationProcessor
from tests.iter_version_dev.V1_0_0.common import TaskName, PretrainType
from tests.iter_version_dev.V1_0_0.processor_bert_classification_predict import ClassificationPredictprocess
from tests.iter_version_dev.V1_0_0.metric import compute_metrics_fn, multi_label_classification_metrics_fn
from tests.iter_version_dev.V1_0_0.automodel_multilabel import AutoModelForMultiLabelClassification


TOKENIZER_FUNC = {
    PretrainType.bert: AutoTokenizer,
    PretrainType.roberta: BertTokenizer
}




MODEL_FUNC = {
    TaskName.text_classification_bert: AutoModelForSequenceClassification,
    TaskName.ner: AutoModelForTokenClassification,
    TaskName.text_similarity_bert: AutoModelForSequenceClassification,
    TaskName.text_classification_multi_bert: AutoModelForMultiLabelClassification
}

PREDICT_PROCESSOR = {
    TaskName.text_classification_bert: ClassificationPredictprocess,
    # TaskName.ner: TokenPredictprocess,
    # TaskName.similarity: ClassificationPredictprocess,
    # TaskName.lr: LrPredictprocess,
}


METRICS_DICT = {
    TaskName.text_classification_bert : compute_metrics_fn,
    TaskName.text_classification_multi_bert : multi_label_classification_metrics_fn,
}

