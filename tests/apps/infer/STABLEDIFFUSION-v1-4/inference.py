import torch
from diffusers import StableDiffusionPipeline, StableDiffusionOnnxPipeline
import time
import sys
import ctypes
import os

# load remoting bottom library
path = os.getenv('REMOTING_BOTTOM_LIBRARY')
if path is not None:
    cpp_lib = ctypes.CDLL(path)
    start_trace = cpp_lib.startTrace
    end_trace = cpp_lib.endTrace

if(len(sys.argv) < 3):
    print('Usage: python3 inference.py num_iter batch_size [model_path]')
    sys.exit()
    
num_iter = int(sys.argv[1])
batch_size = int(sys.argv[2])
model_path = 'CompVis/stable-diffusion-v1-4'
if len(sys.argv) > 3:
    model_path = sys.argv[3]
    print('model_path:', model_path)
else:
    print('Using remote model:', model_path)
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print("read model")
pipe = StableDiffusionPipeline.from_pretrained(
    model_path,
    revision="main",
    torch_dtype=torch.float32,
).to(device)
print("end read model")

prompt = "a photo of an astronaut riding a horse on mars"

# if batch_size>16:
#     pipe.enable_vae_slicing()

# remove initial overhead
torch.cuda.empty_cache()
for i in range(2):
    print("image")
    images = pipe(prompt=[prompt] * batch_size, num_inference_steps=50).images

if path is not None:
    start_trace()

T1 = time.time()

for i in range(num_iter*10):
    print("iter: ", i)
    images = pipe(prompt=[prompt] * batch_size, num_inference_steps=50).images
    
T2 = time.time()
print('time used: ', (T2-T1)/10)

if path is not None:
    end_trace()

images[0].save("astronaut_rides_horse.png")
