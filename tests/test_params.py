
# 用来发送一个非键值对的可变数量的参数列表给一个函数
def test_var_args(f_arg, *argv):
    print("first normal arg:", f_arg)
    for arg in argv:
        print("another arg through *argv:", arg)

test_var_args('yasoob', 'python', 'eggs', 'test')


# **kwargs 允许你将不定长度的键值对, 作为参数传递给一个函数。 
# 如果你想要在一个函数里处理带名字的参数, 你应该使用**kwargs。
def greet_me(**kwargs):
    for key, value in kwargs.items():
        print("{0} == {1}".format(key, value))


greet_me(name="yasoob")

print("------使用*args和**kwargs 来调用一个函数-------")

def test_args_kwargs(arg1, arg2, arg3):
    print("arg1:", arg1)
    print("arg2:", arg2)
    print("arg3:", arg3)

args = ("two", 3, 5)
test_args_kwargs(*args)

kwargs = {"arg3": 3, "arg2": "two", "arg1": 5}
test_args_kwargs(**kwargs)


