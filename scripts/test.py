from gpt4all import GPT4All

model = GPT4All("orca-mini-3b-gguf2-q4_0.gguf")

# Generate Selenium code
prompt = "Generate a Python Selenium script to log into a website."
with model.chat_session():
    response = model.generate(prompt, max_tokens=256)
    print(response)