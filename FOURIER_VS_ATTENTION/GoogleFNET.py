# # Load model directly
# from transformers import AutoTokenizer, AutoModelForPreTraining

# tokenizer = AutoTokenizer.from_pretrained("google/fnet-base")
# model = AutoModelForPreTraining.from_pretrained("google/fnet-base")


# inputs = tokenizer("Hello how are you")
# out = model.generate(**inputs)
# print(out)
# Since the above doesnt work
import os
import torch

path = r"C:\Users\Rohit Francis\.cache\huggingface\hub\models--google--fnet-base\snapshots\d89b6fad3cf5384848b783dc480f9685f49d008c"
pytorch_bin_path = r"C:\Users\Rohit Francis\.cache\huggingface\hub\models--google--fnet-base\snapshots\d89b6fad3cf5384848b783dc480f9685f49d008c\pytorch_model.bin"

print(os.listdir(path))

pytorch_bin = torch.load(pytorch_bin_path)

print(pytorch_bin.keys())
# print(pytorch_bin)