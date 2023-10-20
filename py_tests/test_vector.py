from ASCsoft.bla import Vector


x = Vector(3)
y = Vector(3)

for i in range(len(x)):
    x[i] = i
y[:] = 2

print(f"x = {x}")
print(f"y = {y}")
print(f"x+3*y = {x + 3 * y}")
print(f"vector x length: {len(x)}")


x = Vector(10)
x[0:] = 1
print(x)

x[3:7] = 2
print(x)

x[0:10:2] = 3
print(x)
