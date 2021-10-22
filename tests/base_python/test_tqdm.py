
from time import sleep
from tqdm import tqdm, trange
import time
from tqdm.auto import tqdm, trange

def test_tqdm():
    for i in tqdm(range(10)):
        # 需要循环或者多次执行的代码
        print('\n the value of i is %d ---'%i)  # 正常的整数值：i
        print(range(i))
        sleep(0.1)

# test_tqdm()
# tqdm显示进度条解释：注意参数不一定要是数字，可迭代对象即可


def test_tqdm2():
    for ss in tqdm(range(10)):
        time.sleep(1)
        print('this is test for tqdm with:',ss)

    list_ = ['高楼','湖水','大海']  #可迭代对象即可，不一定要数字
    for ss in tqdm(list_):
        time.sleep(1)
        print('this is test for tqdm with:', ss)


# 可以很明显看到：
# （1）想看进度，直接看百分比即可，表示完成了多少，例如80%，当然也可以看分数8/10。
# （2）第一个时间表示已经执行的时间，第二个时间表示还剩多少时间。
# （3）速度：s/it，表示每次迭代需要的时间。

# test_tqdm2()


def test_tqdm3():
    list_ = ['高楼','湖水','大海']
    for ss in tqdm(list_, desc='---', postfix='***'):
        time.sleep(2)
        print('this is test for tqdm with:', ss)

# test_tqdm3()


# trange 同python中的range,区别在于trange在循环执行的时候会输出打印进度条

def test_trange():
    for i in trange(1, 4):
        print('第%d次执行'%i)
        time.sleep(1)

# test_trange()

def test_trange2():
    for i in trange(1, 4, desc = "Epoch"):
        print('第%d次执行'%i)
        time.sleep(1)
    print("----------------")
    for i in trange(1, 4, desc = "Epoch", disable = True):
        print('第%d次执行'%i)
        time.sleep(1)
     
# 进度条后面的 5.00s/it 是说循环一次耗时为5s


test_trange2()


