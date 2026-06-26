import os
import torch

print(os.environ.get("TORCH_HOME"))
print(torch.hub.get_dir())