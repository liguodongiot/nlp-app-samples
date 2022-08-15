# 将字符串str当成有效的表达式来求值并返回计算结果。eval函数可以实现list、dict、tuple与str之间的转化


a = "[[1,2], [3,4], [5,6], [7,8], [9,0]]"

print(type(a))

b = eval(a)

print(type(b))

print(b)

print("======================")

a = "{1: 'a', 2: 'b'}"

print(type(a))

b = eval(a)

print(type(b))

print(b)

print("======================")

a = "([1,2], [3,4], [5,6], [7,8], (9,0))"

print(type(a))

b = eval(a)

print(type(b))

print(b)

print("======================")

a = "{1, 2, 3}"

print(type(a))

b = eval(a)

print(type(b))

print(b)
