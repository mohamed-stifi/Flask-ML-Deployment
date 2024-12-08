import os
from transformers import pipeline


generator = pipeline('text-generation', model="gpt2")

prompt = "what is machine"

output = generator(prompt, do_sample=False)

print(output[0]['generated_text'])