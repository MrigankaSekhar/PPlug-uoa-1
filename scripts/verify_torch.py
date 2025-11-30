import torch, sys

print("torch version:", torch.__version__)
print("torch file:", torch.__file__)
print("has torch.backends:", hasattr(torch, "backends"))
print("cuda available:", torch.cuda.is_available())

# optional: inspect what's on sys.path for accidental shadowing
print("\nfirst 5 sys.path entries:")
for p in sys.path[:5]:
    print(" -", p)