
from enum import Enum
from dataclasses import dataclass


class StateCode(Enum):
    SUCCESS = (10000, '成功')
    READ_DATASET_ERROR = (21001, "读取数据集异常")
    DATA_SPLIT_ERROR = (22001, "数据切分异常")
    PREHANDLE_ERRPR = (23001, "预处理异常")
    MODEL_TRAIN_ERROR = (24001, "模型训练异常")
    MODEL_EVAL_ERROR = (25001, "模型评估异常")

    def __init__(self, code:int, message:str):
        self.code = code      
        self.message = message


code, message  = StateCode.PREHANDLE_ERRPR.value
print(f"code: {code} , message: {message}")
print("------------")


@dataclass
class CodeItem:
    code: int
    message: str

class BussinessCode(Enum):
    SUCCESS = CodeItem(10000, '成功')
    COMMON_ERROR = CodeItem(20001, "模型训练通用异常")
    READ_DATASET_ERROR = CodeItem(21001, "读取数据集异常")
    DATA_SPLIT_ERROR = CodeItem(22001, "数据切分异常")
    PREHANDLE_ERRPR = CodeItem(23001, "预处理异常")
    MODEL_TRAIN_ERROR = CodeItem(24001, "模型训练异常")
    MODEL_EVAL_ERROR = CodeItem(25001, "模型评估异常")


result:CodeItem = BussinessCode.DATA_SPLIT_ERROR.value
print(f"code: {result.code} , message: {result.message}")

print(f"{BussinessCode.DATA_SPLIT_ERROR.value.message}")
print("------------")

