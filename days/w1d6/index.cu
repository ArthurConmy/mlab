__global__ void index(float *dest, float *a, int32_t *b, int32_t a_size, int32_t b_size) {
    
    int32_t b_index = threadIdx.x + blockIdx.x * blockDim.x;

    if (b_size >= b_index) {
        int32_t a_index = b[b_index];

        if (a_index >= a_size) {
            printf("b_index: %d, a_index: %d\n", b_index, a_index);
        } else {
            dest[b_index] = a[b[b_index]];
        }
    }

}