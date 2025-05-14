import numpy as np
import matplotlib.pyplot as plt
a = np.arange(15).reshape(3,5)
print(a.dtype.name)
print(a.size)
print(type(a))
b = np.array([1.2, 2.2])
print(type(b))
print(b.size)
print(b.itemsize)
a = np.array([1, 2, 3, 4], dtype=int)
b = np.array([2,3,4,5])
a = a.reshape(2,2)
b = b.reshape(2,2)
tich = a * b
matrix = a.dot(b)
print(tich)
print(matrix)

rg = np.random.default_rng(1)
a = rg.random((2,3))
b = np.arange(12).reshape(3,4)
c = np.linspace(1,4,5)
print(b)
print(b.sum(axis=0))
print(b.sum(axis=1))
def f(x,y):
    return 10 *x + y

z = np.fromfunction(f, (5,4), dtype=int)
print(z[1:3, 2]) 
g = rg.random((2,2,3))
print(g.flatten().reshape(-1, 6))
def double(a):
  '''Return a * 2'''
  return a * 2
a = np.linspace(0,5,20)
b = np.linspace(0,10,20)
# plt.plot(a,b,'red')
# plt.plot(a,b,'o','blue')
z = np.array([2, 1, 5, 7, 4, 6, 8, 14, 10, 9, 18, 20, 22,29])

z = z.reshape(-1,2)
z_res = np.flip(z)
print(z_res)
a = np.array((1,2))


# print(a)
# print(a.sum())
# print(a.min())

# c = np.ones((3,3,4))
# d = np.empty((2,3))
# print(np.arange(10,24,5))

# e = np.linspace(0,1.8*3.14, 100)
# f = np.sin(e)
