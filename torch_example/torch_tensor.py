from __future__ import  print_function
import torch
'''torch 基础实例'''
# x = torch.empty(5, 3)
# print(x)
# x = torch.rand(5, 3)
# print(x)
# x = torch.zeros(5, 3, dtype=torch.long)
# print(x)
# x = torch.tensor([5.5, 3])
# print(x)
# x = x.new_ones(5, 3, dtype=torch.double)
# print(x)
# x = torch.rand_like(x, dtype=torch.float)
# print(x)
# print(x.size())
#
# y = torch.rand(5, 3)
# print(x+y)

# x = torch.randn(1)
# print(x)
# print(x.item())

'''自动微分 '''
# x = torch.ones(2, 2, requires_grad=True)
# print(x)
# y = x + 2
# print(y)
# print(y.grad_fn)
#
# z = y * y * 3
# out = z.mean()
# print(z, out)
#
# a = torch.rand(2, 2)
# a = ((a*3)/(a-1))
# print(a.requires_grad)
# a.requires_grad_(True)
# print(a.requires_grad)
# b = (a*a).sum()
# print(b.grad_fn)
#
# out.backward(torch.tensor(1.))
# print(x.grad)

'''雅可比向量积'''
x = torch.rand(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:
    y = y * 2
print(y)

v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(v)
print(x.grad)








