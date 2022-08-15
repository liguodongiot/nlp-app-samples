

from typing import List, NamedTuple, Callable



# 最简单的函数
def print_name(name: str):
    print(name)


# 判断函数是否可调用
print(isinstance(print_name, Callable))

x = 1
print(isinstance(x, Callable))

print("------------------")




# Callable 作为函数参数

# 看看 Callable 的源码
# Callable[[int], str] is a function of (int) -> str
# 第一个类型(int)代表参数类型
# 第二个类型(str)代表返回值类型

def print_name(name: str):
    print(name)


# Callable 作为函数参数使用，其实只是做一个类型检查的作用，检查传入的参数值 get_func 是否为可调用对象
def get_name(get_func: Callable[[str], None]):
    return get_func


vars = get_name(print_name)
vars("test")


# 等价写法，其实就是将函数作为参数传入
def get_name_test(func):
    return func


vars2 = get_name_test(print_name)
vars2("小菠萝")


print("------------------")

# Callable 作为函数返回值

# Callable  作为函数返回值使用，其实只是做一个类型检查的作用，看看返回值是否为可调用对象
def get_name_return() -> Callable[[str], None]:
    return print_name


vars = get_name_return()
vars("test")


# 等价写法，相当于直接返回一个函数对象
def get_name_test():
    return print_name


vars2 = get_name_test()
vars2("小菠萝")






