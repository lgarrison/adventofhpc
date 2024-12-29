#include <array>
#include <chrono>
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>

const uint64_t ITERATIONS_PER_THREAD = 4096;
const uint64_t THREADS_PER_BLOCK = 1024;
const uint64_t BLOCKS_PER_GRID = 65536;

__managed__ unsigned long long answer = ULLONG_MAX;

__global__ void compute(uint64_t start){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    std::array<uint8_t, 16> target = {2,4,1,6,7,5,4,6,1,4,5,5,0,3,3,0};

    unsigned long long Astart = start + ITERATIONS_PER_THREAD * tid;
    unsigned long long Aend = start + ITERATIONS_PER_THREAD * (tid + 1);
    for(unsigned long long Aseed = Astart; Aseed < Aend; Aseed++){
        
        uint64_t A = Aseed;
        uint64_t i;
        for (i = 0; i < target.size() && A != 0; i++){
            uint64_t B = A & 0b111;
            B ^= 0b110;
            uint64_t C = A >> B;
            B ^= C;
            B ^= 0b100;
            uint8_t out = B & 0b111;
            
            // 19m 49s with no early exit
            // 2m 4s with early exit
            if (out != target[i]){
                break;
            }
            
            A = A >> 3;
        }
        
        if (i == target.size()){
            atomicMin(&answer, Aseed);
        }
    }
}

int main(){
    uint64_t iterations_per_grid = BLOCKS_PER_GRID * THREADS_PER_BLOCK * ITERATIONS_PER_THREAD;
    
    uint64_t progress = 0;
    auto last = std::chrono::high_resolution_clock::now();
    for(uint64_t i = 0; answer == ULLONG_MAX; i += iterations_per_grid){

        if (i - progress > 2e12){
            auto now = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = now - last;
            auto rate = (double) (i - progress) / duration.count();
            printf("rate: %.4g T/sec\n", rate / 1e12);

            progress = i;
            last = now;
        }

        compute<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(i);
        cudaDeviceSynchronize();
    }

    printf("answer: %llu\n", answer);

    return 0;
}
