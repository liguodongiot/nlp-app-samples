# nlp-app-samples
机器学习训练脚手架




## 依赖库

| 库 | 版本 |
| - | - |
| PyTorch | 1.9.1 |
| Transformers | 4.11.0|
| scikit-learn | 1.0|




## 创建项目及初始化

### 新建项目
>poetry new nlp-app-samples

### 查看目录结构

> cd nlp-app-samples
> tree
```
.
├── README.rst
├── nlp_app_samples
│   └── __init__.py
├── pyproject.toml
└── tests
    ├── __init__.py
    └── test_nlp_app_samples.py

```

### 配置默认分支
git config --global init.defaultBranch main

### 初始化git
> git init

### 添加代码到暂存区
> git add *

### 提交到本地仓库
> git commit -m '初始化'

### 在github上新建nlp-app-samples
> git@github.com:liguodongIOT/nlp-app-samples.git


### 添加远程仓库
> git remote add origin git@github.com:liguodongIOT/nlp-app-samples.git

### 拉取远程仓库代码
可以允许不相关历史提交，强制合并。

> git pull origin main --allow-unrelated-histories

### 将本地仓库push远程仓库，并将origin设为默认远程仓库
> git push -u origin main


## 虚拟环境

### 创建虚拟环境
> poetry env use /Users/dev/miniconda3/bin/python3

### 激活虚拟环境

> poetry shell


### 安装依赖

> sudo poetry install
> sudo poetry add torch==1.9.1
> sudo poetry add transformers==4.11.0
> sudo -H pip install scikit-learn==1.0
> sudo poetry add pandas


### 查看依赖

> poetry show --tree




## 参考文档

- [dataclasses-json](https://pypi.org/project/dataclasses-json/)

