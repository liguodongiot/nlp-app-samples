
import logging
from math import log 
import os
import pprint

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')


dirs = "file:///Users/liguodong/work/data/temp/classify/one, file:///Users/liguodong/work/data/temp/classify/two ,file:///Users/liguodong/work/data/temp/classify/three,/Users/liguodong/work/data/temp/classify/two"
dirs = dirs.replace("file://", "")
dir_list = dirs.split(',')
dir_list = [ dir.strip() for dir in dir_list ]

print("---------")
print(dir_list)

result = ','.join(dir_list)

print("---------")
print(result)

print("---------")

def read_data_multidir(dirs: str):
    dir_list = dirs.split(',')
    logging.info(f"语料列表：{dir_list}")

    file_list = []
    for temp in dir_list:
        if os.path.isdir(temp):
            get_file_path(temp, file_list, 1)
        else:
            # 过滤掉以下划线开头的文件
            # 仅保留以.txt文件结尾的文件
            filename = os.path.basename(temp)
            if (not filename.startswith('_')) and filename.endswith(('.dev', '.test', '.train', '.txt')):
                file_list.append(temp)
            
    pprint.pprint(f"待处理的所有语料文件：{set(file_list)}")
        


def get_file_path(root_path, file_list, level:int):
    logging.info(f"遍历目录：{root_path}，层级：{level}")
    # 获取该目录下所有的文件名称和目录名称
    dir_or_files = os.listdir(root_path)
    for dir_file in dir_or_files:
        
        # 获取目录或者文件的路径
        dir_file_path = os.path.join(root_path, dir_file)
        # 判断该路径为文件还是路径
        if os.path.isdir(dir_file_path):
            if level==3:
                return
            # 递归获取所有文件和目录的路径
            get_file_path(dir_file_path,file_list, level+1)
        else:
            # 过滤掉以下划线开头的文件
            filename = os.path.basename(dir_file_path)
            if (not filename.startswith('_')) and filename.endswith(('.dev', '.test', '.train', '.txt')):
                file_list.append(dir_file_path)

# 根目录路径
root_path = r"/Users/liguodong/work/data/temp/ner"
# 用来存放所有的文件路径
file_list = []
# 用来存放所有的目录路径
get_file_path(root_path, file_list, 1)
print(file_list)

print("==========")


dirs = '/Users/liguodong/work/data/temp/ner,/Users/liguodong/work/data/temp/ner/output/test.txt,/Users/liguodong/work/data/temp/sim/one/one.txt'
read_data_multidir(dirs)
print("==========")


dirs = '/Users/liguodong/work/data/temp/ner/output/test.txt,/Users/liguodong/work/data/temp/sim/one/one.txt'
read_data_multidir(dirs)
