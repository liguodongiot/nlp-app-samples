
from dataclasses import dataclass, field
from dataclasses_json import dataclass_json


def default_model_parameter(model_ouput_path: str):
    model_parameter = TrainingParameter(model_ouput_path = model_ouput_path)
    model_parameter.epoch_num = 2
    return model_parameter


@dataclass(init=True)
class TrainingParameter:
    """
    模型训练默认参数
    """
    model_ouput_path: str = field(default="./", metadata={'描述': '模型输出路径'})
    learning_rate:float = field(default=0.01, metadata={'描述': '学习率'})
    epoch_num:int = field(default=10, metadata={'描述': '迭代次数'})
    train_batch_size:int = field(default= 32, metadata={'描述': '模型训练每一批次读取数据量的大小'})
    eval_batch_size:int = field(default=32, metadata={'描述': '模型评估练每一批次读取数据量的大小'})
    max_seq_length:int = field(default=128, metadata={'描述': '数据集语料最大长度'})

@dataclass_json
@dataclass(init=True)
class MetricParameter:
    key:str = field(metadata={'描述': '指标的名字'})
    value:object = field(metadata={'描述': '指标的值'})
    description:str = field(default= '',metadata={'描述': '指标的详细描述'})
    show_precentage:bool = field(default= True, metadata={'描述': '指标的名字'})


# metric = MetricParameter(key = '准确率', value = 0.11)
# print(metric.to_json())
