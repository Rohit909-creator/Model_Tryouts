

import torch
import clip
from PIL import Image

# 1️⃣ Load CLIP model + preprocessing
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
texts = ["Lawrys", "Prego", 'Coca Cola']
# 2️⃣ Load an image and text
image = preprocess(Image.open(r"C:\Users\Rohit Francis\Documents\GitHub\Amazon_ML_2025\student_resource\images\img5.jpg")).unsqueeze(0).to(device)
text = clip.tokenize(texts).to(device)

# 3️⃣ Get image & text embeddings
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)

# 4️⃣ Normalize embeddings (important for cosine similarity)
image_features /= image_features.norm(dim=-1, keepdim=True)
text_features /= text_features.norm(dim=-1, keepdim=True)

# 5️⃣ Compute similarity (cosine)
similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# 6️⃣ Print results
print("Similarity scores:", similarity)
best_match = similarity.argmax().item()
print(f"Best match text: {best_match} → '{texts[best_match]}'")
