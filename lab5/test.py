import torch

def extract(x):
    over = torch.arange(8) * 4  # [0, 4, 8, 12, 16, 20, 24, 28]
    mask = 2**4 - 1  # 15 or 0b1111
    return (x[..., None] >> over) & mask

# Assuming `offset` is a tensor of shape [32]
offset = torch.randint(0, 2**32, (32,1), dtype=torch.int32)

# Apply extract function
offset_extracted = extract(offset)
print("Extracted offset shape:", offset_extracted.shape)  # [32, 8]

# Add a new axis
offset_expanded = offset_extracted[..., None]
print("Offset shape after adding new axis:", offset_expanded.shape)  # [32, 8, 1]

# Expand dimensions
GROUP = 8
offset_expanded = offset_expanded.expand(-1, 1, 8, GROUP)
print("Offset shape after expanding dimensions:", offset_expanded.shape)  # [32, 1, 8, 8]

# Make the tensor contiguous
offset_contiguous = offset_expanded.contiguous()

# Reshape the tensor
offset_final = offset_contiguous.view(-1, 64)
print("Final offset shape:", offset_final.shape)  # [32, 64]
