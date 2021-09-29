


from enum import Enum


# 流水线状态
class PipelineStatusTypes(Enum):
    NotStarted = 1
    Failed = 2
    Succeeded = 3
    Running = 4

# 流水线的步骤类型
class StepTypes(Enum):
    base = 1 # 基础
    data = 2 # 获取数据
    split = 3 # 数据切分
    preprocessor = 4 # 预处理
    trainer = 5 # 训练
    evaluator = 6 # 评估
    postprocessor = 7 # 后处理





