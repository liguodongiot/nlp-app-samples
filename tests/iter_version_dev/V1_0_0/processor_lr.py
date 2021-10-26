import os
import csv
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from typing import List, Optional, Union, Dict
from enum import Enum
import logging
import pprint

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')

logger = logging.getLogger(__name__)

class LanguageType:
    """
    type of language
    """
    CN = 2
    EN = 1


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class Preprocessor(object):

    
    @classmethod
    def texts_clean(cls, texts):
        """
        :param texts: list of str.
        :return: list str. 
        cleaned texts
        """
        return texts

    @classmethod
    def train_vectorizers(cls, texts, labels, **kwargs):
        """
        :param texts: list of str
        :param labels: list of str
        :return: vectorizer, labelencoder, x, y
        """
        raise NotImplementedError('not implement train_vetorizers method!')

    @classmethod
    def predict_vectorizer(cls, text, vectorizer, **kwargs):
        """
        :param text: str
        :param vectorizer: such as countvectorize  word2vec
        :return: numpy of shape
        """
        clean_text = cls.texts_clean([text])[0]
        return vectorizer.transform([clean_text])

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """
            Reads a tab separated value file.
            读取以制表符分割的TSV文件
        """
        with open(input_file, "r", encoding="utf-8-sig") as f:
            return list(csv.reader(f, delimiter="\t", quotechar=quotechar))

    """
        获取数据集
    """
    @classmethod
    def get_examples(cls, data_dir: str, mode: Split):
        file_name = "{}.tsv".format(mode.value)
        examples = cls._create_examples(cls._read_tsv(os.path.join(data_dir, file_name)), mode)
        return examples

    """
        将数据集处理成texts和labels
    """
    @classmethod
    def _create_examples(cls, lines: List[str], mode: Split):
        """Creates examples for the training, dev and test sets."""
        texts = []
        labels = []
        text_index = 0
        for (i, line) in enumerate(lines):
            # 第一行表头不进行处理
            if i == 0:
                continue
            text_a = line[text_index]
            if len(line) > text_index + 1:
                label = line[text_index + 1]
            else:
                label = None
            texts.append(text_a)
            labels.append(label)
        return (texts, labels)
    


class LrPreprocessor(Preprocessor):

    EN_TFIDF_VECTOR_PARAMS = {
            "ngram_range": (1, 3),
            "max_features": 200000,
            "analyzer": "char",
            "min_df": 3,
            "max_df": 0.9,
            "strip_accents": "unicode",
            "use_idf": True,
            "smooth_idf": True,
            "sublinear_tf": True,
            "stop_words": 'english',
            "norm": "l2",
            "lowercase": True
            }

    CN_TFIDF_VECTOR_PARAMS = {
            "ngram_range": (1, 3),
            "max_features": 100000,
            "analyzer": "char",
            "min_df": 1,
            "max_df": 1.0,
            "strip_accents": "unicode",
            "use_idf": True,
            "smooth_idf": True,
            "sublinear_tf": True,
            "norm": "l2",
            "lowercase": True
            }

    """
    训练数据向量化
    """
    @classmethod
    def train_vectorizers(cls, texts, labels, **kwargs):
        # 数据清洗
        clean_texts = cls.texts_clean(texts)

        language_type = kwargs.get("languageType", LanguageType.CN)

        params = cls.EN_TFIDF_VECTOR_PARAMS if language_type == LanguageType.EN else cls.CN_TFIDF_VECTOR_PARAMS
        
        # 将原始文档集合转换为 TF-IDF特征矩阵
        vectorizer = TfidfVectorizer(**params)

        labelencoder = LabelEncoder()

        x = vectorizer.fit_transform(clean_texts)

        y = labelencoder.fit_transform(labels)

        logger.info(f"LabelEncoder: lable: {labels}, y: {y}")

        return (vectorizer, labelencoder, x, y)

    """
    评估数据向量化
    """
    @classmethod
    def eval_vectorizers(cls, texts, labels, vectorizer, labelencoder):
        
        clean_texts = cls.texts_clean(texts)

        x = vectorizer.transform(clean_texts)

        y = labelencoder.transform(labels)

        return (x, y)

    @classmethod
    def texts_clean(cls, texts):
        """
        文本数据清洗
        返回移除字符串头尾指定的字符序列生成的新字符串。
        """
        if not texts:
            return texts
        return [t.strip('\n\t\r，') for t in texts]


if __name__ == '__main__':
    texts = ['你好啊 ', '你是什么东西，', ' 日本人哦，']
    labels = ['a', 'a' ,'c']
    vectorizer, labelencoder, x ,y = LrPreprocessor.train_vectorizers(texts, labels)
    pprint.pprint(vectorizer.__dict__)
    print("---------------")
    result = LrPreprocessor.predict_vectorizer('你东西', vectorizer)
    pprint.pprint(result)

    # 将稀疏矩阵转换为密集（即常规的numpy矩阵），然后打印密集表示。
    trans_result = result.todense()
    pprint.pprint(trans_result)

    print("---------------")
    print(result.shape)