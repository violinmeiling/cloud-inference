import os
import shutil
import torch
from safetensors.torch import load_file
from transformers import DistilBertForSequenceClassification, AutoTokenizer, AutoConfig

model_folder = "./bert_latest"
output_root = "./bert_shards"
num_shards = 2  # DistilBERT has 4 layers
layers_per_shard = 2

os.makedirs(output_root, exist_ok=True)

print("Loading tokenizer and config...")
tokenizer = AutoTokenizer.from_pretrained(model_folder, use_fast=False, local_files_only=True)
config = AutoConfig.from_pretrained(model_folder, local_files_only=True)

print("Building model architecture...")
model = DistilBertForSequenceClassification(config)

print("Loading raw state dict...")
state_dict = load_file(os.path.join(model_folder, "model.safetensors"))

# Load the state dict with strict=False to skip mismatches
missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing_keys)
print("Unexpected keys:", unexpected_keys)

# Now your model is partially loaded — fine for splitting layers
layers = model.distilbert.transformer.layer
num_layers = len(layers)

for shard_idx in range(num_shards):
    start = shard_idx * layers_per_shard
    end = min(start + layers_per_shard, num_layers)
    shard_folder = os.path.join(output_root, f"shard_{shard_idx:02d}")
    os.makedirs(shard_folder, exist_ok=True)

    # Save shard layers
    shard_layers = torch.nn.ModuleList([layers[i] for i in range(start, end)])
    torch.save(shard_layers, os.path.join(shard_folder, "layers.pt"))

    # Copy tokenizer files
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

    # Save embeddings only once (first shard)
    if shard_idx == 0:
        torch.save(model.distilbert.embeddings.state_dict(), os.path.join(shard_folder, "embeddings.pt"))
    # Save classifier only once (last shard)
    if shard_idx == num_shards - 1:
        torch.save(model.pre_classifier.state_dict(), os.path.join(shard_folder, "pre_classifier.pt"))
        torch.save(model.classifier.state_dict(), os.path.join(shard_folder, "classifier.pt"))

    print(f"Shard {shard_idx}: layers {start} to {end-1} saved.")

print("Distribution complete.")
print(f"Shards saved in {output_root}.")
