import time
import numpy as np
import sys

from dhllinalg.bla import Matrix, ParallelComputing, NumThreads

s = 150
numTestsPerS = 20
maxS = 750
incS = 25

print("Opening file results.csv...\t")
resFile = open("results.csv", "x")
print("done.\n")

resFile.write(f"iterations\tthreads\ttime in ns\tmatrix size\tGMAC/s\n")

while s <= maxS:
    print(f"initializing {s}x{s} matrices...\t")
    m = Matrix(s, s)
    n = Matrix(s, s)
    for i in range(s):
        for j in range(s):
            m[i, j] = i + j
            n[i, j] = 2 * i + j

    print("done.\n")

    singleThreadResults = np.empty(numTestsPerS)
    multiThreadResults = np.empty(numTestsPerS)
    nThreads = 1
    for i in range(numTestsPerS):
        print(f"{i}:")
        sys.stdout.write("\tMeasuring with 1 thread...\t")
        sys.stdout.flush()
        start = time.time_ns()
        c = m * n
        end = time.time_ns()
        print("done.")
        t = end - start
        singleThreadResults[i] = t
        print(f"\tt={t/1e9}s")

        with ParallelComputing():
            nThreads = NumThreads()
            sys.stdout.write(f"\tMeasuring with {NumThreads()} threads...\t")
            sys.stdout.flush()
            start = time.time_ns()
            d = m * n
            end = time.time_ns()
            print("done.")
            t = end - start
            multiThreadResults[i] = t
            print(f"\tt={t/1e9}s")

        resFile.write(
            f"{i}\tsingle.1\t{singleThreadResults[i]}\t{s}\t{s*s*s/singleThreadResults[i]}\n"
        )
        resFile.write(
            f"{i}\tmulti.{nThreads}\t{multiThreadResults[i]}\t{s}\t{s*s*s/multiThreadResults[i]}\n"
        )

    # resFile.write(f"{numTestsPerS}\tsingle.1\t{np.median(singleThreadResults)}\t{s}\t{s*s*s/np.median(singleThreadResults)}\n")
    # resFile.write(f"{numTestsPerS}\tmulti.{nThreads}\t{np.median(multiThreadResults)}\t{s}\t{s*s*s/np.median(multiThreadResults)}\n")

    s += incS

resFile.close()
