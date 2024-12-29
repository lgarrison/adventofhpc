# Advent of HPC

Some Advent of Code solutions using brute-force HPC methods. Minimum thinking, maximum computing!

## 2024 Day 17
For [2024 Day 17](https://adventofcode.com/2024/day/17), we have a search space of about 2^48 integers and just need to perform some bitwise operations on each. This is eminently brute-forceable on a GPU! With 1 H100 PCIe, we can test 0.73 T inputs per second and get the answer in 124 seconds. We see about 200 / 350 W of GPU usage in `nvtop`, and 98% occupancy and 86% SM throughput in `ncu`. Pretty good!
