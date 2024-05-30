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

print(torch.tensor([[1, 3], [2, 4]]) * torch.tensor([[1, 2], [3, 5]]))
# a = torch.tensor([[1, 3, 2], [1, 3, 2], [1, 3, 2]])
# print(a[:1, :])

@triton.jit
def flashatt_kernel(q_ptr, k_ptr, v_ptr, z_ptr, N0, T, B0: tl.constexpr):
    pid_0 = tl.program_id(0)
    log2_e = 1.44269504
    print(f'pid 0: {pid_0}')
    # YOUR CODE STRATS HERE
    # 1. initialize m , d and o



    offsets_q = tl.arange(0, B0) + pid_0 * B0
    mask_q = offsets_q < T
    # print(f'offsets_q: {offsets_q}')
    q =tl.load(q_ptr + offsets_q, mask=mask_q, other=0.0)
    # print(f'q: {q}')
    for i in tl.static_range(0, T, B0):
      offsets_kv = tl.arange(0, B0) + i
      mask_kv = offsets_kv < T

      k = tl.load(k_ptr + offsets_kv, mask=mask_kv, other=0.0)
      v = tl.load(v_ptr + offsets_kv, mask=mask_kv, other=0.0)

      # 2. compute x

      # 3. compute new m,d and o

      # 4. update m,d and o

      # YOUR CODE ENDS HERE

    tl.store(z_ptr + offsets_q, o, mask=mask_q)

    return