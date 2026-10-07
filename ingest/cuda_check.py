import torch
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
import os
tok = os.environ.get("HF_TOKEN", "")
print("hf_token_chars", len(tok))
