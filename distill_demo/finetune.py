from datasets import load_dataset, DatasetDict
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from transformers import TrainingArguments
from transformers.trainer_utils import EvaluationStrategy

import random

# === Load CSV ===
dataset = load_dataset("csv", data_files={"full": "classified_texts_balanced.csv"})

# === Rename to consistent columns ===
dataset = dataset["full"].rename_columns({
    "inner_text": "text",
    "electric_car": "label"
})

# === Map labels: Y -> 1, N -> 0 ===
def encode_labels(example):
    example["label"] = 1 if example["label"] == "Y" else 0
    return example

dataset = dataset.map(encode_labels)

# === Shuffle and split ===
dataset = dataset.shuffle(seed=42)
split = dataset.train_test_split(test_size=0.2)  # 80% train, 20% test

print(split)

# === Load tokenizer ===
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

tokenized_datasets = split.map(tokenize_function, batched=True)

# === Load model ===
model = DistilBertForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

# === Training arguments ===
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    acc = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }


# === Trainer ===
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,

)

# === Train ===
trainer.train()

# === Save final model and tokenizer ===
model.save_pretrained("./distill_finetuned")
tokenizer.save_pretrained("./distill_finetuned")

print("✅ Done! Model and tokenizer saved in ./distill_finetuned")
