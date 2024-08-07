import os
import torch

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("pankajmathur/orca_mini_3b")
model = AutoModelForCausalLM.from_pretrained("pankajmathur/orca_mini_3b").to(device)

inputs = tokenizer("Generate a Python Selenium script to log into a website.", return_tensors="pt").to(device)
outputs = model.generate(inputs['input_ids'], max_length=256)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
