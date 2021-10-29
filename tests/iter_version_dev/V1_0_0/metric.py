
import numpy as np
from transformers import EvalPrediction
from typing import Dict
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score as sk_f1_score
from sklearn.metrics import recall_score as sk_recall_score
from sklearn.metrics import precision_score as sk_precision_score
from torch import nn


def compute_metrics_fn(p: EvalPrediction) -> Dict:
    preds = np.argmax(p.predictions, axis=1)
    return {"acc": (preds == p.label_ids).astype(int).mean()}


def multi_label_classification_metrics_fn(p: EvalPrediction) -> Dict:
    """
        多标签分类模型指标计算
    :param p:
        p.label_ids: 真实标签，shape batch_size * num_labels
        p.predictions：预测 logits 值，sigmoid 之前的值
    :return:
    """

    y_true = p.label_ids
    y_pre = p.predictions

    micro_auc = roc_auc_score(y_true.ravel(), y_pre.ravel())

    # 这里是 sigmod 前的logits，所以大于 0 即表示正例
    y_pre_binary = (y_pre >= 0).astype(int)
    micro_acc = (y_true == y_pre_binary).astype(int).mean()

    total_y_true = np.sum(y_true)
    y_pre_sort_idx = np.argsort(-y_pre, axis=-1) # 各个样本按值从大到小排序，得到位置
    def get_topn_pr(topn):
        """
            获取 topn 的 precision 和 recall
        :param topn: 最高的几个
        :return:
            {
            'precision': right / num_labels * topn,
            'recall': right / all_label_num
        }
        """
        y_topn_idx = y_pre_sort_idx[:, 0:topn]
        y_topn_ids_label = [y[topn_idx] for y, topn_idx in zip(y_true, y_topn_idx)]
        topn_recall = np.sum(y_topn_ids_label) / total_y_true
        topn_precision = np.mean(y_topn_ids_label)
        return {
            'precision': topn_precision,
            'recall': topn_recall
        }

    top1_pr = get_topn_pr(1)
    top3_pr = get_topn_pr(3)
    top5_pr = get_topn_pr(5)

    return {'micro_auc': micro_auc,
            'micro_acc': micro_acc,
            'top1_precision': top1_pr['precision'],
            'top1_recall': top1_pr['recall'],
            'top3_precision': top3_pr['precision'],
            'top3_recall': top3_pr['recall'],
            'top5_precision': top5_pr['precision'],
            'top5_recall': top5_pr['recall']
            }


##########


def f1_score(y_true, y_pred, average='micro', suffix=False):
    """Compute the F1 score.

    The F1 score can be interpreted as a weighted average of the precision and
    recall, where an F1 score reaches its best value at 1 and worst score at 0.
    The relative contribution of precision and recall to the F1 score are
    equal. The formula for the F1 score is::

        F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import f1_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> f1_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)
    nb_true = len(true_entities)

    p = nb_correct / nb_pred if nb_pred > 0 else 0
    r = nb_correct / nb_true if nb_true > 0 else 0
    score = 2 * p * r / (p + r) if p + r > 0 else 0

    return score


def accuracy_score(y_true, y_pred):
    """Accuracy classification score.

    In multilabel classification, this function computes subset accuracy:
    the set of labels predicted for a sample must *exactly* match the
    corresponding set of labels in y_true.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import accuracy_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> accuracy_score(y_true, y_pred)
        0.80
    """
    if any(isinstance(s, list) for s in y_true):
        y_true = [item for sublist in y_true for item in sublist]
        y_pred = [item for sublist in y_pred for item in sublist]

    nb_correct = sum(y_t == y_p for y_t, y_p in zip(y_true, y_pred))
    nb_true = len(y_true)

    score = nb_correct / nb_true

    return score


def end_of_chunk(prev_tag, tag, prev_type, type_):
    """
    检查块是否在上一个单词和当前单词之间结束。
    Checks if a chunk ended between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_end: boolean.
    """
    chunk_end = False

    if prev_tag == 'E':
        chunk_end = True
    if prev_tag == 'S':
        chunk_end = True

    if prev_tag == 'B' and tag == 'B':
        chunk_end = True
    if prev_tag == 'B' and tag == 'S':
        chunk_end = True
    if prev_tag == 'B' and tag == 'O':
        chunk_end = True
    if prev_tag == 'I' and tag == 'B':
        chunk_end = True
    if prev_tag == 'I' and tag == 'S':
        chunk_end = True
    if prev_tag == 'I' and tag == 'O':
        chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    """
    检查一个块是否在上一个单词和当前单词之间开始
    Checks if a chunk started between the previous and current word.

    Args:
        prev_tag: previous chunk tag.
        tag: current chunk tag.
        prev_type: previous type.
        type_: current type.

    Returns:
        chunk_start: boolean.
    """
    chunk_start = False

    if tag == 'B':
        chunk_start = True
    if tag == 'S':
        chunk_start = True

    if prev_tag == 'E' and tag == 'E':
        chunk_start = True
    if prev_tag == 'E' and tag == 'I':
        chunk_start = True
    if prev_tag == 'S' and tag == 'E':
        chunk_start = True
    if prev_tag == 'S' and tag == 'I':
        chunk_start = True
    if prev_tag == 'O' and tag == 'E':
        chunk_start = True
    if prev_tag == 'O' and tag == 'I':
        chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    return chunk_start


def get_entities(seq, suffix=False):
    """
    从序列中获取实体
    Gets entities from sequence.

    Args:
        seq (list): sequence of labels.

    Returns:
        list: list of (chunk_type, chunk_start, chunk_end).

    Example:
        >>> from seqeval.metrics.sequence_labeling import get_entities
        >>> seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
        >>> get_entities(seq)
        [('PER', 0, 1), ('LOC', 3, 3)]
    """
    # for nested list
    if any(isinstance(s, list) for s in seq):
        seq = [item for sublist in seq for item in sublist + ['O']]

    prev_tag = 'O'
    prev_type = ''
    begin_offset = 0
    chunks = []
    for i, chunk in enumerate(seq + ['O']):
        if suffix:
            tag = chunk[-1]
            type_ = chunk.split('-')[0]
        else:
            tag = chunk[0]
            type_ = chunk.split('-')[-1]

        if end_of_chunk(prev_tag, tag, prev_type, type_):
            chunks.append((prev_type, begin_offset, i-1))
        if start_of_chunk(prev_tag, tag, prev_type, type_):
            begin_offset = i
        prev_tag = tag
        prev_type = type_

    return chunks

def precision_score(y_true, y_pred, average='micro', suffix=False):
    """Compute the precision.

    The precision is the ratio ``tp / (tp + fp)`` where ``tp`` is the number of
    true positives and ``fp`` the number of false positives. The precision is
    intuitively the ability of the classifier not to label as positive a sample.

    The best value is 1 and the worst value is 0.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import precision_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> precision_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_pred = len(pred_entities)

    score = nb_correct / nb_pred if nb_pred > 0 else 0

    return score


def recall_score(y_true, y_pred, average='micro', suffix=False):
    """Compute the recall.

    The recall is the ratio ``tp / (tp + fn)`` where ``tp`` is the number of
    true positives and ``fn`` the number of false negatives. The recall is
    intuitively the ability of the classifier to find all the positive samples.

    The best value is 1 and the worst value is 0.

    Args:
        y_true : 2d array. Ground truth (correct) target values.
        y_pred : 2d array. Estimated targets as returned by a tagger.

    Returns:
        score : float.

    Example:
        >>> from seqeval.metrics import recall_score
        >>> y_true = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> y_pred = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        >>> recall_score(y_true, y_pred)
        0.50
    """
    true_entities = set(get_entities(y_true, suffix))
    pred_entities = set(get_entities(y_pred, suffix))

    nb_correct = len(true_entities & pred_entities)
    nb_true = len(true_entities)

    score = nb_correct / nb_true if nb_true > 0 else 0

    return score


##############

def compute_metrics_fn(p: EvalPrediction) -> Dict:
    predictions = p.predictions
    label_ids = p.label_ids
    preds = np.argmax(predictions, axis=2)

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(label_ids[i][j])
                preds_list[i].append(preds[i][j])
    token_out_label_list = []
    token_preds_list = []
    for i in out_label_list:
        token_out_label_list.extend(i)
    for i in preds_list:
        token_preds_list.extend(i)
    return {
        # "precision": precision_score(out_label_list, preds_list),
        # "recall": recall_score(out_label_list, preds_list),
        # "f1": f1_score(out_label_list, preds_list),
        "token_precision": sk_precision_score(
            token_out_label_list,
            token_preds_list, average='micro'),
        "token_recall": sk_recall_score(
            token_out_label_list,
            token_preds_list, average='micro'),
        "raw_f1": sk_f1_score(
             token_out_label_list,
            token_preds_list, average='micro'),
    }


def ner_metrics(predictions, label_ids, id2label, cal_entity=False):
    if not cal_entity:
        preds = np.argmax(predictions, axis=2)
    else:
        # cal_entity=True，则为计算单一实体指标，例如A
        # 需在函数外部执行np.argmax，并将除A外的实体处理成O
        preds = predictions

    batch_size, seq_len = preds.shape

    out_label_list = [[] for _ in range(batch_size)]
    preds_list = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        for j in range(seq_len):
            if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                out_label_list[i].append(id2label[label_ids[i][j]])
                preds_list[i].append(id2label[preds[i][j]])
    
    token_out_label_list = []
    token_preds_list = []
    for i in out_label_list:
        token_out_label_list.extend(i)
    for i in preds_list:
        token_preds_list.extend(i)
    return {
        "precision": precision_score(out_label_list, preds_list),
        "recall": recall_score(out_label_list, preds_list),
        "f1": f1_score(out_label_list, preds_list)
    }

