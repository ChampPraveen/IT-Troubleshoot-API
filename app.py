import torch
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel

app = FastAPI()

from huggingface_hub import login
login("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")
# Load model
model_path = "Praveen02072004/Alpaca-gpt2"

tokenizer = GPT2Tokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token

model = GPT2LMHeadModel.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Greeting responses
greetings = {
    "hi": "Hello! 👋 How can I help you today?",
    "hello": "Hi there! 😊 What can I do for you?",
    "hey": "Hey! How can I assist you?",
    "good morning": "Good morning! ☀️ How can I help?",
    "good afternoon": "Good afternoon! What can I help you with?",
    "good evening": "Good evening! 🌙 How may I assist you?"
}

# Request schema
class RequestData(BaseModel):
    query: str

# Chat function (same as yours)
def chat(prompt):

    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    inputs = tokenizer(formatted_prompt, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=200,
        temperature=0.7,
        top_k=40,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2,
        no_repeat_ngram_size=3,
        do_sample=True
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)

    return response.split("### Response:")[-1].strip()


# API endpoint
@app.post("/chat")
def chatbot(data: RequestData):

    user_input = data.query.lower()

    if user_input in greetings:
        return {"response": greetings[user_input]}

    response = chat(data.query)

    return {"response": response}
