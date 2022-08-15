



import numpy as np


class ADTangent:

    # 自变量 x，对自变量进行求导得到的 dx
    def __init__(self, x, dx):
        self.x = x
        self.dx = dx

    # 重载 str 是为了方便打印的时候，看到输入的值和求导后的值
    def __str__(self):
        context = f'value:{self.x:.4f}, grad:{self.dx}'
        return context

    def __add__(self, other):
        if isinstance(other, ADTangent):
            x = self.x + other.x
            dx = self.dx + other.dx
        elif isinstance(other, float):
            x = self.x + other
            dx = self.dx
        else:
            return NotImplementedError
        return ADTangent(x, dx)


    def __sub__(self, other):
        if isinstance(other, ADTangent):
            x = self.x - other.x
            dx = self.dx - other.dx
        elif isinstance(other, float):
            x = self.x - other
            dx = self.dx
        else:
            return NotImplementedError
        return ADTangent(x, dx)

    def __mul__(self, other):
        if isinstance(other, ADTangent):
            x = self.x * other.x
            dx = self.x * other.dx + self.dx * other.x
        elif isinstance(other, float):
            x = self.x * other
            dx = self.dx * other
        else:
            return NotImplementedError
        return ADTangent(x, dx)

    def log(self):
        x = np.log(self.x)
        dx = 1 / self.x * self.dx
        return ADTangent(x, dx)

    def sin(self):
        x = np.sin(self.x)
        dx = self.dx * np.cos(self.x)
        return ADTangent(x, dx)


x_1 = ADTangent(x=2., dx=1)
x_2 = ADTangent(x=5., dx=0)
f = ADTangent.log(x_1) + x_1 * x_2 - ADTangent.sin(x_2)
print(f)
