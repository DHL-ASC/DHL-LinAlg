import time
from ASCsoft.bla import Matrix
from ASCsoft.bla import Vector
from ASCsoft.bla import StartWorkers
from ASCsoft.bla import StopWorkers

s = 2000
numThreads = 8
print(f"initializing {s}x{s} matrices...\n")
m = Matrix(s, s)
n = Matrix(s, s)
for i in range(s):
    for j in range(s):
        m[i, j] = i+j
        n[i, j] = 2*i+j

print("done.\n")
print("Measuring with 1 thread...\n")
start = time.time()
c = m*n
end = time.time()
print(end - start)
print("\ndone.\n")

print(f"Measuring with {numThreads} thread\n")
StartWorkers(numThreads)
start = time.time()
d = m*n
end = time.time()
StopWorkers()
print(end - start)
print("\ndone.\n")