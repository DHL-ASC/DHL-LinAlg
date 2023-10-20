from ASCsoft.bla import Matrix
from ASCsoft.bla import Vector
import pickle

x = Vector(3)

for i in range(len(x)):
    x[i] = i

m = Matrix(3, 3)
n = Matrix(3, 3)
m[1, 2] = 1.0

for i in range(3):
    n[i, i] = 5
print(m)
print(m.shape)

print(m[0, 0])
print(m[1, 2])
print(m * x)
print(m * n)
print(m.ncols)
