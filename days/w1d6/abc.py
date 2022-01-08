import torch
import numpy as np
import pycuda.autoinit as _
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from pycuda_utils import Holder, load_module

mod = load_module('abc.cu')
abc_kernel = mod.get_function('abc')

dest = torch.ones(128, dtype=torch.float32).cuda()
a = torch.ones(128, dtype=torch.float32).cuda() * 10.0
b = torch.ones(128, dtype=torch.float32).cuda() * 3.0
c = torch.ones(128, dtype=torch.float32).cuda() * 2.0

abc_kernel(Holder(dest), Holder(a), Holder(b), Holder(c), block=(128, 1, 1), grid=(1, 1))

torch.cuda.synchronize()
print(dest)