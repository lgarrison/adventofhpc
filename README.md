# Advent of HPC

Some Advent of Code solutions using brute-force HPC methods. Minimum thinking, maximum computing!

## 2024 Day 17
For [2024 Day 17](https://adventofcode.com/2024/day/17), we have a search space of about 2^48 integers and just need to perform some bitwise operations on each. This is eminently brute-forceable on a GPU! With a CUDA implementation running on 1 H100 PCIe, we can test 0.73 T inputs per second and get the answer in 124 seconds. We see about 200 / 350 W of GPU usage in `nvtop`, and 98% occupancy and 86% SM throughput in `ncu`. Pretty good!

For fun, we also implement three Python versions:
1) a CuPy `RawKernel` (0.73 T / sec),
2) a CuPy experimental JIT kernel (*0.48 T / sec),
3) a Numba CUDA kernel (0.48 T / sec).

The CuPy `RawKernel` gets the same performance as the native CUDA implementation, as we would hope/expect given that the kernel itself it still written in CUDA C++, and we don't have much host-side work.

The CuPy experimental JIT kernel is kind of hideous, as the JIT is very limited at present. Indexing arrays with non-constants is not supported, nor are break statements, so we have to hack around this. It ends up running at about 2/3rds the speed of the native kernel, but doesn't get the right answer unless the initial guess is sufficiently close. My best guess is that this is a mis-compile (or perhaps use of an unsupported feature), but it's also possible I simply have a bug.

The Numba kernel is very pleasant and natural to write. It achieves the same speed as the CuPy JIT kernel—to 2 decimal places, in fact—and gets the right answer, too.
