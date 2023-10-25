import time

from ASCsoft.bla import Matrix, ParallelComputing, NumThreads

s = 1000
print(f"initializing {s}x{s} matrices...\n")
m = Matrix(s, s)
n = Matrix(s, s)
for i in range(s):
    for j in range(s):
        m[i, j] = i + j
        n[i, j] = 2 * i + j

print("done.\n")
print("Measuring with 1 thread...\n")
start = time.time()
c = m * n
end = time.time()
print(end - start)
print("\ndone.\n")


with ParallelComputing():
    print(f"Measuring with {NumThreads()} thread\n")
    start = time.time()
    d = m * n
    end = time.time()
print(end - start)
print("\ndone.\n")
