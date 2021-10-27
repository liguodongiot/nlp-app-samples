
import logging
import os
import numpy as np
import random
import sys
import argparse

from tests.iter_version_dev.V1_0_0.nlp_decorate import NLPTrainer, NLPPredictor
from tests.iter_version_dev.V1_0_0.task_bask import default_train_args

# 1. 启动类
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task_name", help="task name: ner/text_classification/similarity", type=str, default="")
    parser.add_argument(
        "--pretrain_name", help="model name:bert-base-chinese", type=str, default="")
    parser.add_argument(
        "--data_dir", help="--train and eval data set dir path", type=str, default="")
    parser.add_argument(
        "--model_path", help="output model dir path", type=str, default="")
    parser.add_argument(
        "--pretrain_path", help="pretained model dir path", type=str, default="")
    parser.add_argument(
        "--is_log_preds", help="whether output predictions", type=int, default=0)
    parser.add_argument(
        "--num_train_epochs", help="number of train epochs", type=int, default=5)
    parser.add_argument(
        "--overwrite_output_dir", help="whether overwrite_output_dir", type=int, default=0)
    parser.add_argument(
        "--callback_url", help="self_training callback url", type=str, default="")
    parser.add_argument(
        "--en_callback", help="whether do self_training callback", action='store_true', default=False)
    # 这里的bool是一个可选参数，返回给args的是 args.bool
    args = parser.parse_args()

    # task_name = args.task_name
    # task_name = "text_classification_bert"
    task_name = "text_similarity_bert"

    # pretrain_name = args.pretrain_name
    pretrain_name = "bert"

    # data_dir = args.data_dir
    # data_dir = "/Users/liguodong/work/data/tnews/temp"

    data_dir = "/Users/liguodong/work/data/similarity"


    # model_path = args.model_path
    # model_path = "/Users/liguodong/work/train_model/roberta"
    model_path = "/Users/liguodong/work/train_model/similarity"

    # pretrain_path = args.pretrain_path
    pretrain_path = "/Users/liguodong/work/pretrain_model/robert_tiny"

    is_log_preds = args.is_log_preds
    callback_url = args.callback_url
    en_callback = args.en_callback

    train_args = default_train_args(model_path)

    # train_args.num_train_epochs = args.num_train_epochs
    train_args.num_train_epochs = 1

    train_args.overwrite_output_dir = args.overwrite_output_dir

    trainner = NLPTrainer(
        task_name,
        pretrain_name=pretrain_name,
        pretrain_path=pretrain_path,
        data_dir=data_dir,
        model_path=model_path,
        fp16=False,
        train_args=train_args,)

    os.environ['MESSAGE_CALL_BACK_URI'] = callback_url
    trainner.train(data_dir, model_path, en_callback=en_callback)

    predictor = NLPPredictor(
        task_name,
        model_path=model_path,
        pretrain_name=pretrain_name)
    # res = predictor.predict_testdataset(args.data_dir, is_log_preds=is_log_preds)
    
    res = predictor.predict_testdataset(data_dir, is_log_preds=is_log_preds)
    print(res)





