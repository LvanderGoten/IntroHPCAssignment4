#include <stdio.h>

__global__ void cuda_hello_world() {
    printf("Hello World from GPU! [ThreadID = %d, BlockID = %d]\n",
           threadIdx.x,
           blockIdx.x);
}

int main() {
    cuda_hello_world<<<1, 256>>>();
    cudaDeviceSynchronize();
    return 0;
}
