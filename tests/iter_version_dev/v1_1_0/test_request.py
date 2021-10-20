

import requests
import pprint

call_back_url = "https://www.fastmock.site/mock/d1797ff17268515632f9b6ee459a68c6/api/v1/train/xx/yy"

request_data = {
    "code": 10000,
    "message": "ok",
    "data": {
        "progress": 100, 
		"measureMetrics": {
            "metrics": [
                {
                    "key": "精确率", 
                    "value": 0.5, 
                    "description": "精确率(Precision)，查准率，表示正确预测为正的占全部预测为正的比例。"
                }, 
                {
                    "key": "召回率", 
                    "value": 0.5, 
                    "description": "召回率(Recall)，查全率，表示正确预测为正的占全部实际为正的比例。"
                }, 
                {
                    "key": "F1值", 
                    "value": 0.5, 
                    "description": "F1值为精确率和召回率的调和平均数，值越大越好。"
                }
            ]
        }
    }
}

response = requests.request("POST", call_back_url, timeout=5, json=request_data) 

if response.status_code == 200:
    result = response.text
    pprint.pprint(result)
    result_json = response.json()
    pprint.pprint(result_json)
    print(f"结果数据，响应码：{result_json.get('code', 'code码不存在')} ，响应消息：{result_json.get('message1', '响应消息不存在')}")

    print(f"结果数据，响应码：{result_json['code']} ，响应消息：{result_json.get('message1', '响应消息不存在')}")
    pprint.pprint("---------")




