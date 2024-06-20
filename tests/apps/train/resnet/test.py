import torch
if torch.backends.cudnn.is_available():
    print("cuDNN is installed.")
else:
    print("cuDNN is not installed.")