# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="Qwen/Qwen3-0.6B", max_new_tokens=1000)
messages = [
    {"role": "user", "content": "how to make a good shawarma at home?"},
]
print(pipe(messages)[0]['generated_text'][1]['content'])