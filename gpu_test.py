import torch

# Check if CUDA (NVIDIA GPU support) is available
print("CUDA available:", torch.cuda.is_available())

# If available, check how many GPUs
print("Number of GPUs:", torch.cuda.device_count())

# If at least one GPU, print its name
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
