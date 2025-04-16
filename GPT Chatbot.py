import pandas as pd
import json
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import ast

# 1. Load JSON file and Convert it to df
with open("F:\chat.json", "r") as file:
    data = json.load(file)
rows = []
for intent in data["intents"]:
    tag = intent["tag"]
    context = intent.get("context_set", "")
    responses = intent["responses"]
    for pattern in intent["patterns"]:
        rows.append({
            "tag": tag,
            "pattern": pattern,
            "responses": responses,
            "context_set": context
        })

df = pd.DataFrame(rows)
df = df.explode("responses").reset_index(drop=True)
df.drop(columns=['context_set'],inplace=True)
print(df.head())


# 2. Split Dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# 3. Tokenization
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token


def tokenize_function(examples):
    # Create prompts with optional category (tag)
    texts = [
        f"Category: {tag}\nUser: {inp}\nBot: {resp}" if tag
        else f"User: {inp}\nBot: {resp}"
        for inp, resp, tag in zip(
            examples["pattern"],  # User input
            examples["responses"],  # Bot responses
            examples.get("tag", [""] * len(examples["pattern"]))  # Optional category (tag)
        )
    ]

    # Tokenizing the texts
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )

    # Set the labels as input_ids for language modeling (auto-regressive)
    tokenized["labels"] = tokenized["input_ids"].clone()

    return tokenized


# Apply tokenization
train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset = test_dataset.map(tokenize_function, batched=True)

# 4. Training Setup
model = GPT2LMHeadModel.from_pretrained("gpt2")

training_args = TrainingArguments(
    output_dir="./gpt2-chatbot",
    evaluation_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    logging_dir="./logs",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

# 5. Training
trainer.train()

# 6. Save Model and Tokenizer
model.save_pretrained("./gpt2-chatbot")
tokenizer.save_pretrained("./gpt2-chatbot")

import torch

def generate_response(user_input, category=None, max_length=100):
    # Prepare the prompt
    if category:
        prompt = f"Category: {category}\nUser: {user_input}\nBot:"
    else:
        prompt = f"User: {user_input}\nBot:"

    # Encode input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,  # random sampling for more diversity
            top_k=50,        # limit to top 50 tokens
            top_p=0.95,      # nucleus sampling
            temperature=0.7  # controls randomness
        )

    # Decode and extract the response part
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    bot_response = output_text.split("Bot:")[-1].strip()
    return bot_response

# 7. Testing
while True:
    user_input = input("You: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    response = generate_response(user_input)
    print("Bot:", response)
