
from progressbar import ProgressBar, Percentage, Bar, Timer, ETA, FileTransferSpeed
from time import sleep


aa = [1,2,3,4,5,6,7,8,9]
total = len(aa)

def dowith_i(i):
    print(list(range(i)))
        
 
widgets = ['当前进度: ',Percentage(), ' ', Bar('=>'),' ', Timer(),
           ' ', ETA(), ' ', FileTransferSpeed()]

bar_object = ProgressBar(widgets=widgets, maxval=10*total).start()

# 其中 'widgets' 参数可以自己设置。
# Timer：表示经过的秒（时间）
# Bar：设置进度条形状
# Percentage：显示百分比进度
# ETA：预估剩余时间
# FileTransferSpeed：文件传输速度

for i in range(total):
    print("\n")
    dowith_i(i)  #做自己的任务
    bar_object.update(10 * i + 1)
    sleep(1)
    print("---------")



bar_object.finish()



