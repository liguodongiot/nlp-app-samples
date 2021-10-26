
import numpy as np
from typing import Dict, List 
import torch

from tests.iter_version_dev.V1_0_0.dataset_custom import ClassificationPredictSet


class PredictProcessing:

    @classmethod
    def pre_processing(cls, texts, tokenizer, model_type, **kwargs) -> List[torch.Tensor]:
        """
        :param texts: list of sentences.
        :param tokenizer: word tokenizer
        :param labels: list of label.
        :param kwargs: model_params.
        :return:
        """
        raise NotImplementedError('not implement pre_processing method!')

    @classmethod
    def post_processing(
        cls, logits: np.ndarray,
        id2label: Dict[int, str],
        label_idlist: list=[],
        **kwargs) -> Dict[str, list]:
        """
        :param logits: list of tensor
        :param id2label: dict
        :param label_idlist: list of ground truth label id
        Return:
            {
                'prediction': list，预测结果 list，
                'label': list，真值结果 list
            }
        """
        raise NotImplementedError('not implement post_processing method!')






class ClassificationPredictprocess(PredictProcessing):

    @classmethod
    def pre_processing(cls, texts, tokenizer, model_type, **kwargs) -> List[torch.Tensor]:
        """
        :param texts: list of sentences.
        :param labels: list of label.
        :param kwargs: model_params.
        :return:
        """
        task_name = kwargs.get("task_name")
        id2label = kwargs.get("id2label")
        max_seq_length = kwargs.get("max_seq_length")
        features_raw = ClassificationPredictSet(
            texts,
            id2label,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            task_name=task_name)
        features = []
        input_ids_list = []
        attention_mask_list = []
        token_ids_list = []
        for feature in features_raw:
            input_ids = feature.input_ids
            input_ids_list.append(input_ids)
            attention_mask_list.append(feature.attention_mask)
            token_ids_list.append(feature.token_type_ids)
        input_ids_list = torch.LongTensor(input_ids_list)
        attention_mask_list = torch.LongTensor(attention_mask_list)
        token_ids_list = torch.LongTensor(token_ids_list)
        features = [input_ids_list, attention_mask_list, token_ids_list]
        return features

    @classmethod
    def post_processing(
        cls, logits: np.ndarray,
        id2label: Dict[int, str],
        label_idlist: List[int]=[],
        **kwargs) -> Dict[str, list]:
        """将 model output tensor 转换成 human readable 的数据结构.

        Params:
            logits: np.ndarray, 模型输出 tensor 转换成的 ndarray.
            id2label: Dict[int, str], 标签字典.
            label_idlist: List[List[int]]，真值 label id list
            topn: int, 输出 top 多少的结果
            output_prob: bool, 是否输出 probability

        Return:
            {
                'prediction': list，预测结果 list，
                'label': list，真值结果 list
            }

            其中 prediction list 根据 topn 和 output_prob 的入参不同，输出不同的数据结构.
            - 当 topn == 1 且 output_prob == False 时 -> List[str]
                示例：['label1', 'label2', ...]
            - 当 topn == 1 且 output_prob == True 时 -> List[dict]
                示例：[{'label': 'xxx', 'probability': 0.4}, {}, ...]
            - 当 topn > 1 且 output_prob == False 时 -> List[List[str]]
                示例：[['label1', 'label2', ...], ['label3', 'label4', ...], [], ...]
            - 当 topn > 1 且 output_prob == True 时 -> List[List[Dict]]
                示例：[[{'label': 'xxx', 'probability': 0.4}, {}], [], ...]
        """
        topn = min(max(1, kwargs.get("topn", 1)), logits.shape[-1])
        output_prob = kwargs.get("output_prob", False)
        if topn == 1:
            # 输出 top1 结果
            pred_ids = np.argmax(logits, axis=1)
            if output_prob:
                items = []
                for index, p_id in enumerate(pred_ids):
                    prob_tensor = torch.nn.Softmax(dim=-1)(torch.from_numpy(logits))
                    item = {'label': id2label[p_id], 'probability': prob_tensor[index][p_id].item()}
                    items.append(item)
            else:
                items = [id2label[p] for p in pred_ids]
        else:
            # 输出多个 topn 结果
            pred_ids = np.argsort(logits, axis=1)[:, ::-1][:, :topn]
            if output_prob:
                items = []
                for index, p_ids in enumerate(pred_ids):
                    prob_tensor = torch.nn.Softmax(dim=-1)(torch.from_numpy(logits))
                    item = [{'label': id2label[s_i], 'probability': prob_tensor[index][s_i].item()} for s_i in p_ids]
                    items.append(item)
            else:
                items = [[id2label[i_sort] for i_sort in p_sort] for p_sort in pred_ids]

        # 获取 label 真值标签
        label_list = [id2label[l_id] for l_id in label_idlist]
        return {'prediction': items, 'label': label_list}
