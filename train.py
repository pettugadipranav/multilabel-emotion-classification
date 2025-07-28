import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import f1_score, classification_report
from datasets import load_dataset
import argparse

# Basic setup
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=3)
args = parser.parse_args()

# Load tokenizer and model
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, problem_type="multi_label_classification", num_labels=5)

# Dummy dataset for structure
data = [
    ("I'm scared and shaking.", [0, 1, 0, 1, 0]),
    ("This is exciting!", [0, 0, 1, 0, 1])
]

# Tokenize
inputs = tokenizer([x[0] for x in data], padding=True, truncation=True, return_tensors="pt")
labels = torch.tensor([x[1] for x in data], dtype=torch.float32)

# Training setup
optimizer = AdamW(model.parameters(), lr=2e-5)
model.train()
for epoch in range(args.epochs):
    outputs = model(**inputs, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch+1} Loss: {loss.item()}")
