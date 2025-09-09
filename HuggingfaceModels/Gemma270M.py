# this model sucks
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m", token = "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")
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-270m", token = "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")
# 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
# prompt = """System: You are an AI Agent capable of calling actions and talking with the User.
# Act as a personal Assistant for a mobile phone:

# Currently you can:

# turnonflash()
# turnonbluetooth()
# turnonWifi()
# """

inputs = tokenizer("में एक", return_tensors="pt")

model.eval()

outputs =  model.generate(**inputs, max_new_tokens=200)

print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

with open("response.txt", 'w', encoding='utf-8') as f:
    f.write(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))