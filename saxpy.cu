#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <assert.h>

#define NUM_REPS 100
#define MAX_ERR 1e-6

float get_uniform_sample(void) {
    return (float) rand() / (float) RAND_MAX;
}

void saxpy_cpu(float *a, float *x, float *y, int n) {
    for(int i = 0; i < n; i++) {
        y[i] = (*a) * x[i] + y[i];
    }
}

__global__ void saxpy_gpu(float *a, float *x, float *y, int n) {
    int thread_num = blockDim.x * gridDim.x;
    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = thread_id; i < n; i += thread_num) {
        y[i] = (*a) * x[i] + y[i];
    }
}

double elapsed_time_ms(struct timeval t1, struct timeval t2) {
    double elapsed_time;
    elapsed_time = (t2.tv_sec - t1.tv_sec) * 1000.0;
    elapsed_time += (t2.tv_usec - t1.tv_usec) / 1000.0;
    return elapsed_time;
}

int main(int argc, char *argv[]){
    if (argc == 1) {
        printf("Missing ARRAY_SIZE in input specification!\n");
        return 1;
    }

    // Parse ARRAY_SIZE
    int n;
    sscanf(argv[1], "%i", &n);

    // Random seed
    srand(time(NULL));

    struct timeval t1_cpu, t2_cpu;
    struct timeval t1_gpu, t2_gpu;
    double elapsed_time_cpu, elapsed_time_gpu;

    float *a, *x, *y, *y_hat;
    float *d_a, *d_x, *d_y;

    // Allocate host memory
    a     = (float*) malloc(sizeof(float));
    x     = (float*) malloc(sizeof(float) * n);
    y     = (float*) malloc(sizeof(float) * n);
    y_hat = (float*) malloc(sizeof(float) * n);

    // Allocate device memory
    cudaMalloc((void**) &d_a     , sizeof(float));
    cudaMalloc((void**) &d_x     , sizeof(float) * n);
    cudaMalloc((void**) &d_y     , sizeof(float) * n);

    for (int k = 0; k < NUM_REPS; k++) {

        // Initialize host arrays
        *a = get_uniform_sample();
        for(int i = 0; i < n; i++){
            x[i] = get_uniform_sample();
            y[i] = get_uniform_sample();
        }

        // Transfer data from host to device memory
        cudaMemcpy(d_a , a, sizeof(float)    , cudaMemcpyHostToDevice);
        cudaMemcpy(d_x , x, sizeof(float) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(d_y , y, sizeof(float) * n, cudaMemcpyHostToDevice);

        // CPU implementation
        gettimeofday(&t1_cpu, NULL);
        saxpy_cpu(a, x, y, n);
        gettimeofday(&t2_cpu, NULL);
        elapsed_time_cpu = elapsed_time_ms(t1_cpu, t2_cpu);

        // Executing kernel
        gettimeofday(&t1_gpu, NULL);
        saxpy_gpu<<<32,256>>>(d_a, d_x, d_y, n);
        cudaDeviceSynchronize();
        gettimeofday(&t2_gpu, NULL);
        elapsed_time_gpu = elapsed_time_ms(t1_gpu, t2_gpu);
    
        // Transfer data back to host memory
        cudaMemcpy(y_hat, d_y, sizeof(float) * n, cudaMemcpyDeviceToHost);

        // Compare solutions
        for (int i = 0; i < n; i++) {
            assert(fabs(y[i] - y_hat[i]) <= MAX_ERR);
        }

        // Elapsed time
        printf("Array Size = %d\tElapsed Time CPU [ms] = %f\tElapsed Time GPU [ms] = %f\n", n, elapsed_time_cpu, elapsed_time_gpu);
    }

    // Deallocate device memory
    cudaFree(d_a);
    cudaFree(d_x);
    cudaFree(d_y);

    // Deallocate host memory
    free(a);
    free(x);
    free(y);
    free(y_hat);
}
