__global__ void sum_atomics(float *input, float *output, int32_t array_size) {
    // dest[threadIdx.x] 

    int32_t add_index = threadIdx.x + blockIdx.x * blockDim.x;

    if (array_size >= add_index) {
        atomicAdd(&output[0], input[add_index]);
    }
}