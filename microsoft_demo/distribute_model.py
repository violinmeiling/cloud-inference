import os
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

model_folder = "./microsoft"
output_root = "./phi2_shards"
num_shards = 16
layers_per_shard = 2

os.makedirs(output_root, exist_ok=True)

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_folder, use_fast=False, local_files_only=True)
config = AutoConfig.from_pretrained(model_folder, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_folder, torch_dtype=torch.float32, local_files_only=True)

layers = model.model.layers
num_layers = len(layers)

for shard_idx in range(num_shards):
    start = shard_idx * layers_per_shard
    end = min(start + layers_per_shard, num_layers)
    shard_folder = os.path.join(output_root, f"shard_{shard_idx:02d}")
    os.makedirs(shard_folder, exist_ok=True)

    shard_layers = torch.nn.ModuleList([layers[i] for i in range(start, end)])
    torch.save(shard_layers.state_dict(), os.path.join(shard_folder, "layers.pt"))

    for fname in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ]:
        src_path = os.path.join(model_folder, fname)
        if os.path.exists(src_path):
            shutil.copy(src_path, shard_folder)
    
    if shard_idx == 0:
        torch.save(model.model.embed_tokens.state_dict(), os.path.join(shard_folder, "embed_tokens.pt"))
    if shard_idx == num_shards - 1:
        torch.save(model.lm_head.state_dict(), os.path.join(shard_folder, "lm_head.pt"))
        if hasattr(model.model, "norm"):
            norm_layer = model.model.norm
        elif hasattr(model.model, "final_layernorm"):
            norm_layer = model.model.final_layernorm
        elif hasattr(model.model, "ln_f"):
            norm_layer = model.model.ln_f
        else:
            norm_layer = None

        if norm_layer is not None:
            torch.save(norm_layer.state_dict(), os.path.join(shard_folder, "norm.pt"))
        else:
            print("No final normalization layer found in the model.")

    print(f"Model distributed into {num_shards} shards, each containing {layers_per_shard} layers.")

print("Distribution complete.")
print(f"Shards saved in {output_root}.")
print("You can now load the shards using the provided loading script.")
print("Make sure to adjust the loading script to match the number of shards and layers per shard.")
print("Happy coding!")