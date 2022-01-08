import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from torch._C import dtype

from pycuda_utils import Holder, load_module, ceil_divide

mod = load_module('index.cu')
index = mod.get_function('index')

a_size = np.int32(17)
b_size = np.int32(20)

dest = torch.randn((b_size, ), dtype=torch.float32).cuda()
a = torch.randn((a_size, ), dtype=torch.float32).cuda()
b = torch.randint(low=0, high=a_size, size=(b_size, ), dtype=torch.int32).cuda()

number_of_blocks = ceil_divide(b_size, 512)

index(Holder(dest), Holder(a), Holder(b), a_size, b_size, block=(512, 1, 1), grid=(int(number_of_blocks), 1))

torch.cuda.synchronize()
print('a', a)
print('b', b)
print('dest', dest)