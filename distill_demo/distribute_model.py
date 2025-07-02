import os
import shutil
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

model_folder = "./distill_finetuned"
output_root = "./distill_shards"
num_shards = 2 
# Use config to determine layers per shard dynamically
config = AutoConfig.from_pretrained(model_folder, local_files_only=True)
num_layers = config.n_layers  # e.g. 4 for DistilBERT
layers_per_shard = (num_layers + num_shards - 1) // num_shards  # ceil division

os.makedirs(output_root, exist_ok=True)

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_folder, use_fast=False, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_folder, torch_dtype=torch.float32, local_files_only=True)

layers = model.distilbert.transformer.layer  # <-- changed for DistilBERT

for shard_idx in range(num_shards):
    start = shard_idx * layers_per_shard
    end = min(start + layers_per_shard, num_layers)
    shard_folder = os.path.join(output_root, f"shard_{shard_idx:02d}")
    os.makedirs(shard_folder, exist_ok=True)

    # Save layers for this shard
    shard_layers = torch.nn.ModuleList([layers[i] for i in range(start, end)])
    torch.save(shard_layers.state_dict(), os.path.join(shard_folder, "layers.pt"))

    # Copy tokenizer & config files to all shards for local_files_only compatibility
    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ]:
        src_path = os.path.join(model_folder, fname)
        if os.path.exists(src_path):
            shutil.copy(src_path, shard_folder)

    # Save embeddings only in first shard
    if shard_idx == 0:
        torch.save(model.distilbert.embeddings.state_dict(), os.path.join(shard_folder, "embeddings.pt"))

    # Save classifier only in last shard (no pooler or LayerNorm for DistilBERT)
    if shard_idx == num_shards - 1:
        torch.save(model.classifier.state_dict(), os.path.join(shard_folder, "classifier.pt"))

    print(f"Shard {shard_idx}: layers {start} to {end - 1} saved.")

print("Shard distribution complete.")
print(f"Shards saved in {output_root}.")
