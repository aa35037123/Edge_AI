import torch
import triton
from torch import Tensor
import triton.language as tl
import jaxtyping
from jaxtyping import Float32, Int32
import torch
import triton
from torch import Tensor
import triton.language as tl
import jaxtyping
from jaxtyping import Float32, Int32
# Define the Triton kernel
# @triton.jit
# def test():
#     print(tl.exp2(4))

# print(test())

def extract(x):
  over = torch.arange(8) * 4
  mask = 2**4 - 1
  return (x[..., None] >> over) & mask

weight = torch.rand(32, 8).to(torch.float32)
print(f'weight: {weight}')
print(extract(weight).view(-1, 64))