

import torch
from torch.autograd import Variable

x = Variable(torch.Tensor([2.]), requires_grad=True)
y = Variable(torch.Tensor([5.]), requires_grad=True)

f = torch.log(x) + x * y - torch.sin(y)
f.backward()

print(f)
print(x.grad)
print(y.grad)
