import os
import shutil
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig

model_folder = "./bert"
output_root = "./bert_shards"
num_shards = 2  # For BERT-base (12 layers), adjust as needed
layers_per_shard = 6

os.makedirs(output_root, exist_ok=True)

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_folder, use_fast=False, local_files_only=True)
config = AutoConfig.from_pretrained(model_folder, local_files_only=True)
model = AutoModelForSequenceClassification.from_pretrained(model_folder, torch_dtype=torch.float32, local_files_only=True)

layers = model.bert.encoder.layer
num_layers = len(layers)

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
        torch.save(model.bert.embeddings.state_dict(), os.path.join(shard_folder, "embeddings.pt"))

    # Save pooler, classifier, and norm only in last shard
    if shard_idx == num_shards - 1:
        # Save full pooler state dict (keys like dense.weight remain intact)
        torch.save(model.bert.pooler.state_dict(), os.path.join(shard_folder, "pooler.pt"))
        torch.save(model.classifier.state_dict(), os.path.join(shard_folder, "classifier.pt"))
        # Also save the LayerNorm state from last layer's output
        torch.save(model.bert.encoder.layer[-1].output.LayerNorm.state_dict(), os.path.join(shard_folder, "norm.pt"))

    print(f"Shard {shard_idx}: layers {start} to {end - 1} saved.")

print("Shard distribution complete.")
print(f"Shards saved in {output_root}.")
