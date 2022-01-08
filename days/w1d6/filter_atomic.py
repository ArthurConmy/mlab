import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from torch._C import dtype

from pycuda_utils import Holder, load_module, ceil_divide

mod = load_module('filter_atomic.cu')
sum_atomics = mod.get_function('filter_atomic')

tensor_size = np.int32(10)
# tensor_size = 520
tensor = torch.randn((tensor_size, ), dtype=torch.float32).cuda()
output = torch.zeros((tensor_size, ), dtype=torch.float32).cuda()

threshold = np.float32(0.5)
counter = torch.zeros((1,),dtype=torch.int32).cuda()

number_of_blocks = ceil_divide(tensor_size, 512)

sum_atomics(Holder(tensor), tensor_size, Holder(output), threshold, Holder(counter), block=(512, 1, 1), grid=(int(number_of_blocks), 1))
torch.cuda.synchronize()
print(output)
print(tensor)