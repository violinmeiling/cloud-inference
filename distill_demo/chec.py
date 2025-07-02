from transformers import TrainingArguments
args = TrainingArguments(
    output_dir=".",
    evaluation_strategy="epoch"
)
print(args)
