import torch
print("version", torch.__version__)
print("cuda_build", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
