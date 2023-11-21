import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import numpy as np
from tqdm import tqdm

import dhllinalg.bla as dhlbla


def check_file_exists(args: argparse.Namespace, file_name: str):
    file = Path(file_name)
    if file.is_file():
        if args.overwrite_all:
            print(f"Deleting {file_name}..")
            file.unlink()
        else:
            print(
                f"File with name: {file_name} already exists. Either change the name, delete the old file or use --overwrite_all."
            )
            sys.exit()


def run_dhl(
    args,
    iterations,
    labels,
    time_in_ns,
    matrix_size,
    gmacs,
):
    print("Running DHL")
    loop = tqdm(
        range(args.initial_size, args.max_size + args.step_size, args.step_size)
    )
    with dhlbla.ParallelComputing(2):
        for size in loop:
            m = dhlbla.Matrix(size, size)
            n = dhlbla.Matrix(size, size)
            for i in range(args.iterations_per_step):
                start = time.time_ns()
                c = dhlbla.InnerProduct(m, n)
                end = time.time_ns()
                t = end - start
                iterations.append(i)
                labels.append("DHL 2 Core")
                time_in_ns.append(t)
                matrix_size.append(size)
                gmacs.append(size**3 / t)
                loop.set_postfix_str(f"Iteration: {i}")
            loop.set_description(f"Matrix size {size}")

    return {
        "iterations": iterations,
        "labels": labels,
        "time_in_ns": time_in_ns,
        "matrix_size": matrix_size,
        "gmacs": gmacs,
    }


def run_dhl_parallel(
    args,
    iterations,
    labels,
    time_in_ns,
    matrix_size,
    gmacs,
):
    print("Running DHL-Parallel")
    loop = tqdm(
        range(args.initial_size, args.max_size + args.step_size, args.step_size)
    )
    for size in loop:
        m = dhlbla.Matrix(size, size)
        n = dhlbla.Matrix(size, size)
        for i in range(args.iterations_per_step):
            start = time.time_ns()
            c = dhlbla.InnerProduct(m, n)
            end = time.time_ns()
            t = end - start
            iterations.append(i)
            labels.append("DHL 1 Core")
            time_in_ns.append(t)
            matrix_size.append(size)
            gmacs.append(size**3 / t)
            loop.set_postfix_str(f"Iteration: {i}")
        loop.set_description(f"Matrix size {size}")

    return {
        "iterations": iterations,
        "labels": labels,
        "time_in_ns": time_in_ns,
        "matrix_size": matrix_size,
        "gmacs": gmacs,
    }


def run_numpy(
    args,
    iterations,
    labels,
    time_in_ns,
    matrix_size,
    gmacs,
):
    print("Running Numpy")
    loop = tqdm(
        range(args.initial_size, args.max_size + args.step_size, args.step_size)
    )
    for size in loop:
        m = np.zeros((size, size))
        n = np.zeros((size, size))
        for i in range(args.iterations_per_step):
            start = time.time_ns()
            c = np.dot(m, n)
            end = time.time_ns()
            t = end - start
            iterations.append(i)
            labels.append("Numpy")
            time_in_ns.append(t)
            matrix_size.append(size)
            gmacs.append(size**3 / t)
            loop.set_postfix_str(f"Iteration: {i}")
        loop.set_description(f"Matrix size {size}")

    return {
        "iterations": iterations,
        "labels": labels,
        "time_in_ns": time_in_ns,
        "matrix_size": matrix_size,
        "gmacs": gmacs,
    }


def main(args: argparse.Namespace):
    if args.overwrite_all:
        print(f"Running with --overwrite_all. All existing files will be overwritten.")
    for library in args.libraries:
        file_name = f"results_{library}.csv"
        check_file_exists(args, file_name)

        iterations = []
        labels = []
        time_in_ns = []
        matrix_size = []
        gmacs = []
        ret = globals()[f"run_{library}"](
            args, iterations, labels, time_in_ns, matrix_size, gmacs
        )

        df = pd.DataFrame(ret)
        df.to_csv(file_name, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark for matrix matrix multiplication"
    )
    parser.add_argument(
        "--initial_size", type=int, default=24, help="Initial matrix size (default: 24)"
    )
    parser.add_argument(
        "--max_size", type=int, default=1200, help="Maximum matrix size (default: 1200)"
    )
    parser.add_argument(
        "--step_size", type=int, default=24, help="Steps size (default: 24)"
    )
    parser.add_argument(
        "--iterations_per_step",
        type=int,
        default=5,
        help="Iterations per step (default: 5)",
    )

    parser.add_argument(
        "--libraries",
        nargs="*",
        default=["all"],
        choices=["dhl", "dhl_parallel", "numpy", "all"],
        help="List of libraries to run (default: all)",
    )
    parser.add_argument(
        "--overwrite_all",
        action="store_true",
        default=False,
        help="Overwrite all existing benchmarks (default: True)",
    )

    args = parser.parse_args()
    if "all" in args.libraries:
        args.libraries = ["dhl", "dhl_parallel", "numpy"]
    main(args)
