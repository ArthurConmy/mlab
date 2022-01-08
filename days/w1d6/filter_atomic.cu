__global__ void filter_atomic(float *input, int32_t size, float *output, float threshold, int32_t *counter) {
    int32_t add_index = threadIdx.x + blockIdx.x * blockDim.x;
    if (add_index < size){
        float val = input[add_index];

        if ((val > 0 && val < threshold) || (val < 0 && val > -1 * threshold)){
            int16_t write_index = atomicAdd(counter, 1);
            output[write_index] = val;
        }
    }
}