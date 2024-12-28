# Advent of HPC

Some Advent of Code solutions using brute-force HPC methods. Minimum thinking, maximum computing!

## 2024 Day 17
For [2024 Day 17](https://adventofcode.com/2024/day/17), we have a search space of about 2^48 integers and just need to perform some bitwise operations on each. This is eminently brute-forceable on a GPU! With 1 H100 PCIe, we can test 1.1 T inputs per second and get the answer in 124 seconds.
