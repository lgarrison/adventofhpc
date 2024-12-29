from timeit import default_timer as timer

import cupy as cp

ITERATIONS_PER_THREAD = 4096
THREADS_PER_BLOCK = 1024
BLOCKS_PER_GRID = 65536


def main():
    kernel = cp.RawKernel(
        f"""#define ITERATIONS_PER_THREAD {ITERATIONS_PER_THREAD}ULL
        """
        + r"""
#include <array>
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

extern "C" __global__ void compute(uint64_t start, unsigned long long *answer){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    // printf("tid: %d, start: %llu\n", tid, start);

    std::array<uint8_t, 16> target = {2,4,1,6,7,5,4,6,1,4,5,5,0,3,3,0};

    unsigned long long Astart = start + ITERATIONS_PER_THREAD * tid;
    unsigned long long Aend = start + ITERATIONS_PER_THREAD * (tid + 1);
    for(unsigned long long Aseed = Astart; Aseed < Aend; Aseed++){
        
        uint64_t A = Aseed;
        uint64_t i = 0;
        for (i = 0; i < target.size() && A != 0; i++){
            uint64_t B = A & 0b111;
            B ^= 0b110;
            uint64_t C = A >> B;
            B ^= C;
            B ^= 0b100;
            uint8_t out = B & 0b111;

            if (out != target[i]){
                break;
            }
            
            A = A >> 3;
        }
        
        if (i == target.size()){
            atomicMin(answer, Aseed);
        }
    }
}
""",
        "compute",
        backend='nvcc',
        options=('--expt-relaxed-constexpr','-std=c++17'),
        jitify=False,
    )

    NO_ANSWER = cp.iinfo(cp.uint64).max
    answer = cp.array([NO_ANSWER])
    i = 0

    iterations_per_grid = BLOCKS_PER_GRID * THREADS_PER_BLOCK * ITERATIONS_PER_THREAD

    tlast = timer()
    while answer[0] == NO_ANSWER:
        
        kernel((BLOCKS_PER_GRID,), (THREADS_PER_BLOCK,), (i, answer,))
        i += iterations_per_grid

        tnow = timer()
        print(f'Rate: {iterations_per_grid / (tnow - tlast) / 1e12:.4g} T / sec')
        tlast = tnow

    print(answer[0])


if __name__ == "__main__":
    main()
