# Benchmark

DHL-LinAlg provides a simple cli to run benchmarks against Numpy and NGSolve. To view all available options run 

```bash
$ python3 benchmark.py --help
usage: benchmark.py [-h] [--initial_size INITIAL_SIZE] [--max_size MAX_SIZE] [--step_size STEP_SIZE] [--iterations_per_step ITERATIONS_PER_STEP]
                    [--libraries [{dhl,numpy,ngsolve,all} ...]] [--override_all]

Benchmark for matrix matrix maultiplication

options:
  -h, --help            show this help message and exit
  --initial_size INITIAL_SIZE
                        Initial size (default: 24)
  --max_size MAX_SIZE   Maximum size (default: 1200)
  --step_size STEP_SIZE
                        Steps size (default: 24)
  --iterations_per_step ITERATIONS_PER_STEP
                        Iterations per step (default: 5)
  --libraries [{dhl,numpy,ngsolve,all} ...]
                        List of libraries to run (default: all)
  --override_all        Override all existing benchmarks (default: True)
```

All benchmarks will be stored separately in a csv file. You can compare all benchmarks inside **benchmark_visualisation.ipynb**.