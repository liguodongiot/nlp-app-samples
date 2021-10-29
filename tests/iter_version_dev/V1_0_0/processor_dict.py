

from tests.iter_version_dev.V1_0_0.common import Split,TaskName
from tests.iter_version_dev.V1_0_0.processor_bert_classification import ClassificationProcessor
from tests.iter_version_dev.V1_0_0.processor_bert_similarity import PairTextProcessor
from tests.iter_version_dev.V1_0_0.processor_bert_ner import TokenClassProcessor

DATA_PROCESSOR = {
    TaskName.text_classification_bert : ClassificationProcessor,
    TaskName.text_similarity_bert: PairTextProcessor,
    TaskName.ner: TokenClassProcessor
}


