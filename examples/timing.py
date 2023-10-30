from pathlib import Path

import time
import pandas as pd
import sys

from dhllinalg.bla import Matrix, ParallelComputing, NumThreads

s = 150
numTestsPerS = 20
maxS = 750
incS = 25

file_name = "results.csv"
if Path(file_name).is_file():
    print(
        f"File with name: {file_name} already exists. Either change the name or delete the old file"
    )
    sys.exit()

iterations = []
threads = []
time_in_ns = []
matrix_size = []
gmacs = []

while s <= maxS:
    print(f"initializing {s}x{s} matrices...\t")
    m = Matrix(s, s)
    n = Matrix(s, s)
    for i in range(s):
        for j in range(s):
            m[i, j] = i + j
            n[i, j] = 2 * i + j

    print("done.\n")

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
        iterations.append(i)
        threads.append(nThreads)
        time_in_ns.append(t)
        matrix_size.append(s)
        gmacs.append(s**3 / t)
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
            iterations.append(i)
            threads.append(nThreads)
            time_in_ns.append(t)
            matrix_size.append(s)
            gmacs.append(s**3 / t)
            print(f"\tt={t/1e9}s")

    s += incS

df = pd.DataFrame(
    {
        "iterations": iterations,
        "threads": threads,
        "time_in_ns": time_in_ns,
        "matrix_size": matrix_size,
        "gmacs": gmacs,
    }
)

df.to_csv(file_name, index=False)
