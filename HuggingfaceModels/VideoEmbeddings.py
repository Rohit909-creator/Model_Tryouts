from transformers import AutoModel, VJEPA2VideoProcessor
from decord import VideoReader
from decord import cpu
import numpy as np
import torch
import time

# Load processor and model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

processor = VJEPA2VideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc64-256", cache_dir="./ModelFiles")
model = AutoModel.from_pretrained("facebook/vjepa2-vitl-fpc64-256", cache_dir="./ModelFiles").to(device)

# Load video and sample frames
video_path1 = "car1.mp4"
video_path2 = "bike riding.mp4"
vr1 = VideoReader(video_path1, ctx=cpu(0))  # Decord video reader
vr2 = VideoReader(video_path2, ctx=cpu(0))  # Decord video reader
# start = time.time()
frame_indices = np.arange(0, 8)  # Sampled frame indices (make sure your video has enough frames)
frames1 = vr1.get_batch(frame_indices)  # Returns a (64, H, W, 3) tensor

# Convert to torch format (T, C, H, W)
video_np1 = frames1.asnumpy()
video_np1 = np.transpose(video_np1, (0, 3, 1, 2))  # T x C x H x W

frames2 = vr2.get_batch(frame_indices)  # Returns a (64, H, W, 3) tensor

# Convert to torch format (T, C, H, W)
video_np2 = frames2.asnumpy()
video_np2 = np.transpose(video_np2, (0, 3, 1, 2))  # T x C x H x W
print("Number of frames:", video_np1.shape)
# Process with processor
video_tensor1 = processor(video_np1, return_tensors="pt").to(model.device)
video_tensor2 = processor(video_np2, return_tensors="pt").to(model.device)
print("Video tensor shape: ", end = "")
print(video_tensor1["pixel_values_videos"])
# print(video_tensor1.shape)

strt = time.time()
# Get video embeddings
with torch.no_grad():
    video_embeddings1 = model.get_vision_features(**video_tensor1)
end = time.time()
print(f"Inference time: {end-strt}")

strt = time.time()
# Get video embeddings
with torch.no_grad():
    video_embeddings2 = model.get_vision_features(**video_tensor2)
end = time.time()
print(f"Inference time: {end-strt}")

print("Video EMbeddings shape: ", end="")
print(video_embeddings1.shape)
print(video_embeddings2.shape)

print(torch.cosine_similarity(video_embeddings1.view(1, -1), video_embeddings2.view(1, -1), dim=0))
print(torch.cosine_similarity(video_embeddings1.view(1, -1), video_embeddings2.view(1, -1), dim=-1))
# print(torch.cosine_similarity(video_embeddings1, video_embeddings2, dim=2))