
import numpy as np
from transformers import EvalPrediction
from typing import Dict
from sklearn.metrics import roc_auc_score


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

