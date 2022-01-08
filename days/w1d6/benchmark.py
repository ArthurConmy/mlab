import time
import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from matplotlib import pyplot as plt

from pycuda_utils import Holder, load_module, ceil_divide

def benchmark(fn, n_iterations):
    for _ in range(10):
        fn()

    start_time = time.time()
    
    for i in range(n_iterations):
        fn()
    
    end_time = time.time()

    return (end_time - start_time) / n_iterations


def run_atomic_sum(size):

    def f():
        mod = load_module('sum_atomics.cu')
        sum_atomics = mod.get_function('sum_atomics')

        tensor_size = np.int32(size)
        tensor = torch.randn((tensor_size, ), dtype=torch.float32).cuda()
        output = torch.zeros((1, ), dtype=torch.float32).cuda()

        number_of_blocks = ceil_divide(tensor_size, 512)

        sum_atomics(Holder(tensor), Holder(output), tensor_size, block=(512, 1, 1), grid=(int(number_of_blocks), 1))
        torch.cuda.synchronize()

    return f

def pytorch_sum(size,on_gpu):
    def f():
        tensor_size = np.int32(size)
        
        if on_gpu:
            tensor = torch.randn((tensor_size, ), dtype=torch.float32).cuda()
        else:
            tensor = torch.randn((tensor_size, ), dtype=torch.float32)

        tensor.sum()
    return f


if __name__ == "__main__":
    array_sizes = [10,100,1000,10000,100000,1e6,1e7]

    our_times = []
    cuda_times = []
    cpu_times = []
    for size in array_sizes:
        f = run_atomic_sum(size)
        our_times.append(benchmark(f,10))

        f = pytorch_sum(size,True)
        cuda_times.append(benchmark(f,10))

        f = pytorch_sum(size,False)
        cpu_times.append(benchmark(f,10))

    
    plt.plot(array_sizes,our_times,label="ours")
    plt.plot(array_sizes,cuda_times,label="pytorch gpu")
    plt.plot(array_sizes,cpu_times,label="pytorch cpu")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.title("Time vs array size for initial sum atomic")
    plt.savefig("sum_atomic_times.png")