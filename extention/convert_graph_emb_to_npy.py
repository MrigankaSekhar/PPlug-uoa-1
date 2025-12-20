import torch
import numpy as np
import sys
import os

if len(sys.argv) != 2:
    print("Usage: python convert_graph_emb_to_npy.py <path_to_emb_file>")
    sys.exit(1)

emb_path = sys.argv[1]
npy_path = emb_path.replace(".emb", ".npy")

print(f"Loading {emb_path} ...")
tensor = torch.load(emb_path, map_location="cpu")

if not isinstance(tensor, torch.Tensor):
    raise TypeError(f"Expected a torch.Tensor in {emb_path}, got {type(tensor)}")

print(f"Original dtype: {tensor.dtype}, shape: {tensor.shape}")
array = tensor.cpu().numpy().astype(np.float16)

print(f"Saving FP16 array to {npy_path} ...")
np.save(npy_path, array)

print(f"✅ Done! Memmap-friendly file ready at {npy_path}")