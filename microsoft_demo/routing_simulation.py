import os
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

shards_root = "./phi2_shards"
num_shards = 16
layers_per_shard = 2
device = "cpu"

first_shard = os.path.join(shards_root, "shard_00")
tokenizer = AutoTokenizer.from_pretrained(first_shard, use_fast=False, local_files_only=True)
config = AutoConfig.from_pretrained(first_shard, local_files_only=True)

print("Loading full model temporarily...")
full_model = AutoModelForCausalLM.from_pretrained("./microsoft", local_files_only=True, torch_dtype=torch.float32)
LayerClass = type(full_model.model.layers[0])

# Extract rotary embedding module (usually called rotary_emb or similar)
del full_model


prompt = "one plus one is"
inputs = tokenizer(prompt, return_tensors="pt")
input_ids = inputs["input_ids"]
position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long).unsqueeze(0)


embed_tokens_state = torch.load(os.path.join(first_shard, "embed_tokens.pt"), map_location=device)
embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
embed_tokens.load_state_dict(embed_tokens_state)
hidden_states = embed_tokens(input_ids)

for shard_idx in range(num_shards):
    shard_folder = os.path.join(shards_root, f"shard_{shard_idx:02d}")
    layers_state = torch.load(os.path.join(shard_folder, "layers.pt"), map_location=device)
    start_idx = shard_idx * layers_per_shard
    layers = nn.ModuleList([
        LayerClass(config, layer_idx = start_idx + i) 
        for i in range(layers_per_shard)])
    layers.load_state_dict(layers_state)
    layers.eval()
    with torch.no_grad():
        for layer in layers:
            hidden_states = layer(hidden_states, position_ids=position_ids)[0]

last_shard = os.path.join(shards_root, f"shard_{num_shards - 1:02d}")
norm_state = torch.load(os.path.join(last_shard, "norm.pt"), map_location=device)
norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
norm.load_state_dict(norm_state)
hidden_states = norm(hidden_states)

lm_head_state = torch.load(os.path.join(last_shard, "lm_head.pt"), map_location=device)
if "bias" in lm_head_state:
    del lm_head_state["bias"]
lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
lm_head.load_state_dict(lm_head_state)
with torch.no_grad():
    logits = lm_head(hidden_states)

next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
output_ids = torch.cat([input_ids, next_token_id], dim=-1)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f"Output text: {output_text}")