
from enum import Enum


class TaskMode:
    training = 0
    inference = 1

class TaskName:
    text_classification_bert = "text_classification_bert"
    text_classification_lr = "text_classification_lr"
    ner = "ner"
    text_similarity_bert = "text_similarity_bert"

class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class PretrainType:
    bert = "bert"
    roberta = "roberta"


class InferMode:
    general = "general"
    onnx = "onnx"


class TrainType:
    general = "general"
    distil = "distil"


class DistilType:
    T3 = "T3"
    T3_Tiny = "T3_Tiny"
    T4 = "T4"
    T4_Tiny = "T4_Tiny"
    T6 = "T6"
    T6_Tiny = "T6_Tiny"



# student模型初始化方式
# [T6, T4, T3]: 使用teacher模型的前n层初始化
# [T6_Tiny, T4_Tiny, T3_Tiny]: 随机初始化
init_strategy = {'T6': 'teacher init',
                 'T6_Tiny': 'random',
                 'T4': 'teacher init',
                 'T4_Tiny': 'random',
                 'T3': 'teacher init',
                 'T3_Tiny': 'random'}




### T3, T3-tiny
L3_attention_mse = [{"layer_T": 4, "layer_S": 1, "feature": "attention", "loss": "attention_mse", "weight": 1},
                    {"layer_T": 8, "layer_S": 2, "feature": "attention", "loss": "attention_mse", "weight": 1},
                    {"layer_T": 12, "layer_S": 3, "feature": "attention", "loss": "attention_mse", "weight": 1}]

L3_attention_ce = [{"layer_T": 4, "layer_S": 1, "feature": "attention", "loss": "attention_ce", "weight": 1},
                   {"layer_T": 8, "layer_S": 2, "feature": "attention", "loss": "attention_ce", "weight": 1},
                   {"layer_T": 12, "layer_S": 3, "feature": "attention", "loss": "attention_ce", "weight": 1}]

L3_attention_mse_sum = [{"layer_T": 4, "layer_S": 1, "feature": "attention", "loss": "attention_mse_sum", "weight": 1},
                        {"layer_T": 8, "layer_S": 2, "feature": "attention", "loss": "attention_mse_sum", "weight": 1},
                        {"layer_T": 12, "layer_S": 3, "feature": "attention", "loss": "attention_mse_sum", "weight": 1}]

L3_attention_ce_mean = [{"layer_T": 4, "layer_S": 1, "feature": "attention", "loss": "attention_ce_mean", "weight": 1},
                        {"layer_T": 8, "layer_S": 2, "feature": "attention", "loss": "attention_ce_mean", "weight": 1},
                        {"layer_T": 12, "layer_S": 3, "feature": "attention", "loss": "attention_ce_mean", "weight": 1}]

L3_hidden_smmd = [{"layer_T": [0, 0], "layer_S": [0, 0], "feature": "hidden", "loss": "mmd", "weight": 1},
                  {"layer_T": [4, 4], "layer_S": [1, 1], "feature": "hidden", "loss": "mmd", "weight": 1},
                  {"layer_T": [8, 8], "layer_S": [2, 2], "feature": "hidden", "loss": "mmd", "weight": 1},
                  {"layer_T": [12, 12], "layer_S": [3, 3], "feature": "hidden", "loss": "mmd", "weight": 1}]

L3n_hidden_mse = [
    {"layer_T": 0, "layer_S": 0, "feature": "hidden", "loss": "hidden_mse", "weight": 1, "proj": ["linear", 384, 768]},
    {"layer_T": 4, "layer_S": 1, "feature": "hidden", "loss": "hidden_mse", "weight": 1, "proj": ["linear", 384, 768]},
    {"layer_T": 8, "layer_S": 2, "feature": "hidden", "loss": "hidden_mse", "weight": 1, "proj": ["linear", 384, 768]},
    {"layer_T": 12, "layer_S": 3, "feature": "hidden", "loss": "hidden_mse", "weight": 1, "proj": ["linear", 384, 768]}]

L3_hidden_mse = [{"layer_T": 0, "layer_S": 0, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                 {"layer_T": 4, "layer_S": 1, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                 {"layer_T": 8, "layer_S": 2, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                 {"layer_T": 12, "layer_S": 3, "feature": "hidden", "loss": "hidden_mse", "weight": 1}]

L3t_hidden_mse = [
    {"layer_T": 0, "layer_S": 0, "feature": "hidden", "loss": "hidden_mse", "weight": 1, "proj": ["linear", 312, 768]},
    {"layer_T": 4, "layer_S": 1, "feature": "hidden", "loss": "hidden_mse", "weight": 1, "proj": ["linear", 312, 768]},
    {"layer_T": 8, "layer_S": 2, "feature": "hidden", "loss": "hidden_mse", "weight": 1, "proj": ["linear", 312, 768]},
    {"layer_T": 12, "layer_S": 3, "feature": "hidden", "loss": "hidden_mse", "weight": 1, "proj": ["linear", 312, 768]}]

### T4, T4-tiny
L4_attention_mse = [{"layer_T": 3, "layer_S": 1, "feature": "attention", "loss": "attention_mse", "weight": 1},
                    {"layer_T": 6, "layer_S": 2, "feature": "attention", "loss": "attention_mse", "weight": 1},
                    {"layer_T": 9, "layer_S": 3, "feature": "attention", "loss": "attention_mse", "weight": 1},
                    {"layer_T": 12, "layer_S": 4, "feature": "attention", "loss": "attention_mse", "weight": 1}]

L4_attention_ce = [{"layer_T": 3, "layer_S": 1, "feature": "attention", "loss": "attention_ce", "weight": 1},
                   {"layer_T": 6, "layer_S": 2, "feature": "attention", "loss": "attention_ce", "weight": 1},
                   {"layer_T": 9, "layer_S": 3, "feature": "attention", "loss": "attention_ce", "weight": 1},
                   {"layer_T": 12, "layer_S": 4, "feature": "attention", "loss": "attention_ce", "weight": 1}]

L4_attention_mse_sum = [{"layer_T": 3, "layer_S": 1, "feature": "attention", "loss": "attention_mse_sum", "weight": 1},
                        {"layer_T": 6, "layer_S": 2, "feature": "attention", "loss": "attention_mse_sum", "weight": 1},
                        {"layer_T": 9, "layer_S": 3, "feature": "attention", "loss": "attention_mse_sum", "weight": 1},
                        {"layer_T": 12, "layer_S": 4, "feature": "attention", "loss": "attention_mse_sum", "weight": 1}]

L4_attention_ce_mean = [{"layer_T": 3, "layer_S": 1, "feature": "attention", "loss": "attention_ce_mean", "weight": 1},
                        {"layer_T": 6, "layer_S": 2, "feature": "attention", "loss": "attention_ce_mean", "weight": 1},
                        {"layer_T": 9, "layer_S": 3, "feature": "attention", "loss": "attention_ce_mean", "weight": 1},
                        {"layer_T": 12, "layer_S": 4, "feature": "attention", "loss": "attention_ce_mean", "weight": 1}]

L4_hidden_smmd = [{"layer_T": [0, 0], "layer_S": [0, 0], "feature": "hidden", "loss": "mmd", "weight": 1},
                  {"layer_T": [3, 3], "layer_S": [1, 1], "feature": "hidden", "loss": "mmd", "weight": 1},
                  {"layer_T": [6, 6], "layer_S": [2, 2], "feature": "hidden", "loss": "mmd", "weight": 1},
                  {"layer_T": [9, 9], "layer_S": [3, 3], "feature": "hidden", "loss": "mmd", "weight": 1},
                  {"layer_T": [12, 12], "layer_S": [4, 4], "feature": "hidden", "loss": "mmd", "weight": 1}]

L4t_hidden_sgram = [{"layer_T": [0, 0], "layer_S": [0, 0], "feature": "hidden", "loss": "gram", "weight": 1,
                     "proj": ["linear", 312, 768]},
                    {"layer_T": [3, 3], "layer_S": [1, 1], "feature": "hidden", "loss": "gram", "weight": 1,
                     "proj": ["linear", 312, 768]},
                    {"layer_T": [6, 6], "layer_S": [2, 2], "feature": "hidden", "loss": "gram", "weight": 1,
                     "proj": ["linear", 312, 768]},
                    {"layer_T": [9, 9], "layer_S": [3, 3], "feature": "hidden", "loss": "gram", "weight": 1,
                     "proj": ["linear", 312, 768]},
                    {"layer_T": [12, 12], "layer_S": [4, 4], "feature": "hidden", "loss": "gram", "weight": 1,
                     "proj": ["linear", 312, 768]}]

L4t_hidden_mse = [
    {"layer_T": 0, "layer_S": 0, "feature": "hidden", "loss": "hidden_mse", "weight": 1, "proj": ["linear", 312, 768]},
    {"layer_T": 3, "layer_S": 1, "feature": "hidden", "loss": "hidden_mse", "weight": 1, "proj": ["linear", 312, 768]},
    {"layer_T": 6, "layer_S": 2, "feature": "hidden", "loss": "hidden_mse", "weight": 1, "proj": ["linear", 312, 768]},
    {"layer_T": 9, "layer_S": 3, "feature": "hidden", "loss": "hidden_mse", "weight": 1, "proj": ["linear", 312, 768]},
    {"layer_T": 12, "layer_S": 4, "feature": "hidden", "loss": "hidden_mse", "weight": 1, "proj": ["linear", 312, 768]}]

L4_hidden_mse = [{"layer_T": 0, "layer_S": 0, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                 {"layer_T": 3, "layer_S": 1, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                 {"layer_T": 6, "layer_S": 2, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                 {"layer_T": 9, "layer_S": 3, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                 {"layer_T": 12, "layer_S": 4, "feature": "hidden", "loss": "hidden_mse", "weight": 1}]

### T6, T6-Tiny
L6_hidden_smmd = [{"layer_T": [0, 0], "layer_S": [0, 0], "feature": "hidden", "loss": "mmd", "weight": 1},
                  {"layer_T": [2, 2], "layer_S": [1, 1], "feature": "hidden", "loss": "mmd", "weight": 1},
                  {"layer_T": [4, 4], "layer_S": [2, 2], "feature": "hidden", "loss": "mmd", "weight": 1},
                  {"layer_T": [6, 6], "layer_S": [3, 3], "feature": "hidden", "loss": "mmd", "weight": 1},
                  {"layer_T": [8, 8], "layer_S": [4, 4], "feature": "hidden", "loss": "mmd", "weight": 1},
                  {"layer_T": [10, 10], "layer_S": [5, 5], "feature": "hidden", "loss": "mmd", "weight": 1},
                  {"layer_T": [12, 12], "layer_S": [6, 6], "feature": "hidden", "loss": "mmd", "weight": 1}]

L6_hidden_mse = [{"layer_T": 0, "layer_S": 0, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                 {"layer_T": 2, "layer_S": 1, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                 {"layer_T": 4, "layer_S": 2, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                 {"layer_T": 6, "layer_S": 3, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                 {"layer_T": 8, "layer_S": 4, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                 {"layer_T": 10, "layer_S": 5, "feature": "hidden", "loss": "hidden_mse", "weight": 1},
                 {"layer_T": 12, "layer_S": 6, "feature": "hidden", "loss": "hidden_mse", "weight": 1}]

L6t_hidden_mse = [
    {"layer_T": 0, "layer_S": 0, "feature": "hidden", "loss": "hidden_mse", "weight": 1, "proj": ["linear", 312, 768]},
    {"layer_T": 2, "layer_S": 1, "feature": "hidden", "loss": "hidden_mse", "weight": 1, "proj": ["linear", 312, 768]},
    {"layer_T": 4, "layer_S": 2, "feature": "hidden", "loss": "hidden_mse", "weight": 1, "proj": ["linear", 312, 768]},
    {"layer_T": 6, "layer_S": 3, "feature": "hidden", "loss": "hidden_mse", "weight": 1, "proj": ["linear", 312, 768]},
    {"layer_T": 8, "layer_S": 4, "feature": "hidden", "loss": "hidden_mse", "weight": 1, "proj": ["linear", 312, 768]},
    {"layer_T": 10, "layer_S": 5, "feature": "hidden", "loss": "hidden_mse", "weight": 1, "proj": ["linear", 312, 768]},
    {"layer_T": 12, "layer_S": 6, "feature": "hidden", "loss": "hidden_mse", "weight": 1, "proj": ["linear", 312, 768]}]


# student模型与teacher模型的中间层匹配方式
match_strategy = {'T6': L6_hidden_mse + L6_hidden_smmd,
                  'T6_Tiny': L6t_hidden_mse + L6_hidden_smmd,
                  'T4': L4_hidden_mse + L4_hidden_smmd,
                  'T4_Tiny': L4t_hidden_mse + L4_hidden_smmd,
                  'T3': L3_hidden_mse + L3_hidden_smmd,
                  'T3_Tiny': L3t_hidden_mse + L3_hidden_smmd}



SUPPORT_PRETRAIN_TYPE = [ v for k, v in PretrainType.__dict__.items() if '__' not in k]

# # [('__module__', '__main__'), ('bert', 'bert'), ('roberta', 'roberta'), ('__dict__', <attribute '__dict__' of 'PretrainType' objects>), ('__weakref__', <attribute '__weakref__' of 'PretrainType' objects>), ('__doc__', None)]
# print(PretrainType.__dict__.items())
# # ['bert', 'roberta']
# print(SUPPORT_PRETRAIN_TYPE)
# print("-----------")

def get_pretrain_type(pretrain_name: str):
    max_match = ''
    for p_type in SUPPORT_PRETRAIN_TYPE:
        if p_type in pretrain_name and len(p_type) > len(max_match):
            max_match = p_type
    return max_match

    

