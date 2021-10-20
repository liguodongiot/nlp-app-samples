import os 
from typing import Dict, Optional
import requests
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(filename)s - %(levelname)s - %(message)s')
logger = logging.getLogger()


def training_callback(
    progress: int,
    measure: Optional[Dict] = None,
    reason: Optional[str] = None,
    en_callback: bool = False,
    code: int = 10000
    ):
    """
    :param progress: 训练进度（-1: 表示训练失败, 大于等于0正常）
    :param measure: 验证集、测试集指标
    :param reason: 训练失败原因
    :param en_callback: 是否进行回调
    :param code: 训练状态码
    :return:
    """
    if not en_callback:
        return
    call_back_url = os.environ.get('MESSAGE_CALL_BACK_URI', None)
    if not call_back_url:
        logger.warning('callback url is None. ')
        return
    logger.info(f'callback url is : {call_back_url} ')

    try:
        request_data = {
            "code": code,
            "message": reason,
            "data": {
                "progress": progress, 
                "measureMetrics": measure 
            }
        }

        response = requests.request(
            "POST", call_back_url, timeout=5, json=request_data)
        if response.status_code == 200:
            logger.info(f'callback success.')
            result_json = response.json()
            logger.info(f"结果数据，响应码：{result_json.get('code', 'code码不存在')} ，响应消息：{result_json.get('message1', '响应消息不存在')}")
        else:
            logger.error(f'callback failed. msg: {response.reason}')
    except:
        logger.error('callback service error.', exc_info=True)




