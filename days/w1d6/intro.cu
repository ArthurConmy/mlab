

__global__ void zero(double *dest) {
    double seventeen = 17;
    dest[threadIdx.x] += seventeen;
}


__global__ void one(float *dest) { 
    dest[threadIdx.x] = 1.f; 
}
