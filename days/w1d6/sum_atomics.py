import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from pycuda_utils import Holder, load_module, ceil_divide

mod = load_module('sum_atomics.cu')
sum_atomics = mod.get_function('sum_atomics')

tensor_size = np.int32(520)
# tensor_size = 520
tensor = torch.randn((tensor_size, ), dtype=torch.float32).cuda()
output = torch.zeros((1, ), dtype=torch.float32).cuda()

number_of_blocks = ceil_divide(tensor_size, 512)

sum_atomics(Holder(tensor), Holder(output), tensor_size, block=(512, 1, 1), grid=(int(number_of_blocks), 1))
torch.cuda.synchronize()
print(output)
print(tensor.sum())