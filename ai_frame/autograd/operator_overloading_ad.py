from typing import List, NamedTuple, Callable, Dict, Optional

import numpy as np

_name = 1


def fresh_name():
    global _name
    name = f'v{_name}'
    _name += 1
    return name

# 操作符重载
class Variable:
    def __init__(self, value, name=None):
        self.value = value
        self.name = name or fresh_name()

    def __repr__(self):
        return repr(self.value)

    # 我们需要从一些张量开始，这些张量的值没有在autograd中计算。此函数用于构造叶节点。
    # We need to start with some tensors whose values were not computed
    # inside the autograd. This function constructs leaf nodes.
    @staticmethod
    def constant(value, name=None):
        var = Variable(value, name)
        print(f'{var.name} = {value}')
        return var

    # 变量(Variable)的乘法，跟踪梯度
    # Multiplication of a Variable, tracking gradients
    def __mul__(self, other):
        return ops_mul(self, other)

    def __add__(self, other):
        return ops_add(self, other)

    def __sub__(self, other):
        return ops_sub(self, other)

    def sin(self):
        return ops_sin(self)

    def log(self):
        return ops_log(self)

# 接下来需要跟踪 Variable 所有计算，以便向后应用链式规则。那么数据结构 Tape 有助于实现这一点
class Tape(NamedTuple):
    inputs: List[str]
    outputs: List[str]
    # apply chain rule
    propagate: 'Callable[[List[Variable]], List[Variable]]'

# 输入 inputs 和输出 outputs 是原始计算的输入和输出变量的唯一名称。
# 反向传播使用链式规则，将函数的输出梯度传播给输入。
# 其输入为 dL/dOutputs，输出为 dL/dinput。Tape只是一个记录所有计算的累积 List 列表。




# 下面提供了一种重置 Tape 的方法 reset_tape，方便运行多次自动微分，每次自动微分过程都会产生 Tape List。
gradient_tape: List[Tape] = []

# reset tape
def reset_tape():
    global _name
    _name = 1
    gradient_tape.clear()

# 看看具体运算操作符是如何定义的，以乘法为例子啦，
# 首先需要计算正向结果并创建一个新变量来表示，也就是 x = Variable(self.value * other.value)。
# 然后定义了反向传播闭包 propagate，使用链规则来反向支撑梯度。
def ops_mul(self, other):
    # forward
    x = Variable(self.value * other.value)
    print(f'{x.name} = {self.name} * {other.name}')

    # backward
    def propagate(dl_doutputs):
        dl_dx, = dl_doutputs
        # 求偏导
        dx_dself = other  # partial derivate of r = self*other
        dx_dother = self  # partial derivate of r = self*other
        dl_dself = dl_dx * dx_dself
        dl_dother = dl_dx * dx_dother

        dl_dinputs = [dl_dself, dl_dother]
        return dl_dinputs

    # record the input and output of the op
    tape = Tape(inputs=[self.name, other.name], outputs=[x.name], propagate=propagate)
    gradient_tape.append(tape)
    return x


def ops_add(self, other):
    x = Variable(self.value + other.value)
    print(f'{x.name} = {self.name} + {other.name}')

    def propagate(dl_doutputs):
        dl_dx, = dl_doutputs
        # 求偏导
        dx_dself = Variable(1.)
        dx_dother = Variable(1.)

        dl_dself = dl_dx * dx_dself
        dl_dother = dl_dx * dx_dother
        return [dl_dself, dl_dother]

    # record the input and output of the op
    tape = Tape(inputs=[self.name, other.name], outputs=[x.name], propagate=propagate)
    gradient_tape.append(tape)
    return x


def ops_sub(self, other):
    x = Variable(self.value - other.value)
    print(f'{x.name} = {self.name} - {other.name}')

    def propagate(dl_doutputs):
        dl_dx, = dl_doutputs

        # 求偏导
        dx_dself = Variable(1.)
        dx_dother = Variable(-1.)

        dl_dself = dl_dx * dx_dself
        dl_dother = dl_dx * dx_dother
        return [dl_dself, dl_dother]

    # record the input and output of the op
    tape = Tape(inputs=[self.name, other.name], outputs=[x.name], propagate=propagate)
    gradient_tape.append(tape)
    return x


def ops_sin(self):
    x = Variable(np.sin(self.value))
    print(f'{x.name} = sin({self.name})')

    def propagate(dl_doutputs):
        dl_dx, = dl_doutputs

        # 求偏导
        dx_dself = Variable(np.cos(self.value))
        dl_dself = dl_dx * dx_dself

        return [dl_dself]

    # record the input and output of the op
    tape = Tape(inputs=[self.name], outputs=[x.name], propagate=propagate)
    gradient_tape.append(tape)
    return x


def ops_log(self):
    x = Variable(np.log(self.value))
    print(f'{x.name} = log({self.name})')

    def propagate(dl_doutputs):
        dl_dx, = dl_doutputs
        # 求偏导
        dx_dself = Variable(1 / self.value)
        dl_dself = dl_dx * dx_dself

        return [dl_dself]

    # record the input and output of the op
    tape = Tape(inputs=[self.name], outputs=[x.name], propagate=propagate)
    gradient_tape.append(tape)
    return x

# grad 呢是将变量运算放在一起的梯度函数，函数的输入是 l 和对应的梯度结果 results。
def grad(l, results):

    dl_d = {} # map dL/dX for all values X

     # 节点 -> 梯度
    dl_d[l.name] = Variable(1.)
    print("dl_d", dl_d)

    def gather_grad(entries):
        return [dl_d[entry] if entry in dl_d else None for entry in entries]

    for entry in reversed(gradient_tape):
        print(entry)

        # 取出反向传播已经计算好的梯度
        dl_doutputs = gather_grad(entry.outputs)

        dl_dinputs = entry.propagate(dl_doutputs)

        for input, dl_dinput in zip(entry.inputs, dl_dinputs):
            if input not in dl_d:
                dl_d[input] = dl_dinput
            else:
                dl_d[input] += dl_dinput
    print("~~~~~~~~~~~~`")
    for name, value in dl_d.items():
        print(f'd{l.name}_d{name} = {value.name}')

    return gather_grad(result.name for result in results)


reset_tape()

x = Variable.constant(2., name='v-1')
y = Variable.constant(5., name='v0')

f = Variable.log(x) + x * y - Variable.sin(y)
print(f)

print("------------------")

dx, dy = grad(f, [x, y])
print("dx", dx)
print("dy", dy)