from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import os
import uuid
import base64
import io
import time
from transformers import AutoTokenizer, AutoConfig

app = Flask(__name__)

pending_work = {}  # device_id: {"hidden_states": ..., "position_ids": ..., "request_id": ...}
results = {}       # device_id: hidden_states
outputs = {}       # request_id: output_text
status_logs = {}   # request_id: list of status messages
num_shards = 16
shards_root = "/tmp/microsoft"
max_new_tokens = 32  # You can adjust this

def serialize_tensor(tensor):
    buf = io.BytesIO()
    torch.save(tensor, buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def deserialize_tensor(b64str):
    buf = io.BytesIO(base64.b64decode(b64str))
    return torch.load(buf)

@app.route("/submit_prompt", methods=["POST"])
def submit_prompt():
    prompt = request.json["prompt"]
    request_id = str(uuid.uuid4())
    first_shard = os.path.join(shards_root, "shard_00")
    tokenizer = AutoTokenizer.from_pretrained(first_shard, use_fast=False, local_files_only=True)
    config = AutoConfig.from_pretrained(first_shard, local_files_only=True)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"]
    position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long).unsqueeze(0)
    embed_tokens_state = torch.load(os.path.join(first_shard, "embed_tokens.pt"), map_location="cpu")
    embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
    embed_tokens.load_state_dict(embed_tokens_state)
    hidden_states = embed_tokens(input_ids)
    # Serialize hidden_states to base64
    hidden_states_b64 = serialize_tensor(hidden_states)
    # Start with device 0
    pending_work[0] = {
        "hidden_states": hidden_states_b64,
        "position_ids": position_ids.tolist(),
        "request_id": request_id
    }
    results.clear()
    results["input_ids"] = input_ids.tolist()
    results["position_ids"] = position_ids.tolist()
    results["request_id"] = request_id
    outputs[request_id] = None
    status_logs[request_id] = ["Prompt submitted."]
    # Store tokenizer/config for this request_id for later use
    outputs[f"{request_id}_tokenizer"] = tokenizer
    outputs[f"{request_id}_config"] = config
    # For generation loop
    outputs[f"{request_id}_output_ids"] = input_ids
    outputs[f"{request_id}_step"] = 0
    outputs[f"{request_id}_finished"] = False
    outputs[f"{request_id}_last_token"] = None
    outputs[f"{request_id}_ready"] = False
    outputs[f"{request_id}_prompt_len"] = input_ids.shape[1]  # <--- Store prompt length
    return jsonify({"status": "ok", "request_id": request_id})

@app.route("/get_work", methods=["GET"])
def get_work():
    device_id = int(request.args["device_id"])
    work = pending_work.pop(device_id, None)
    if work:
        return jsonify(work)
    return jsonify({})

@app.route("/submit_result", methods=["POST"])
def submit_result():
    device_id = int(request.json["device_id"])
    print(f"Backend received result from device {device_id}")
    hidden_states = deserialize_tensor(request.json["hidden_states"])
    request_id = request.json.get("request_id", results.get("request_id"))
    results[device_id] = hidden_states
    if request_id not in status_logs:
        status_logs[request_id] = []
    if device_id < num_shards - 1:
        msg = f"Backend received result from device {device_id}, routing to device {device_id + 1}"
        print(msg)
        status_logs[request_id].append(msg)
        hidden_states_b64 = serialize_tensor(hidden_states)
        pending_work[device_id + 1] = {
            "hidden_states": hidden_states_b64,
            "position_ids": results["position_ids"],
            "request_id": request_id
        }
    else:
        msg = "All devices done, computing final output for this token..."
        print(msg)
        status_logs[request_id].append(msg)
        # Save last hidden_states for get_result to use
        outputs[f"{request_id}_last_hidden_states"] = hidden_states
        outputs[f"{request_id}_ready"] = True
    return jsonify({"status": "ok"})

@app.route("/get_result", methods=["GET"])
def get_result():
    request_id = request.args["request_id"]
    # If already finished, return output
    if outputs.get(request_id) is not None:
        return jsonify({"status": "done", "output": outputs[request_id]})
    # Otherwise, run generation loop
    tokenizer = outputs.get(f"{request_id}_tokenizer")
    config = outputs.get(f"{request_id}_config")
    output_ids = outputs.get(f"{request_id}_output_ids")
    step = outputs.get(f"{request_id}_step", 0)
    finished = outputs.get(f"{request_id}_finished", False)
    last_token = outputs.get(f"{request_id}_last_token")
    prompt_len = outputs.get(f"{request_id}_prompt_len", 0)
    if finished or tokenizer is None or output_ids is None:
        return jsonify({"status": "pending"})
    for _ in range(8):  # Generate up to 8 tokens per poll for speed
        if step >= max_new_tokens:
            generated_ids = output_ids[0, prompt_len:]  # Only generated tokens
            outputs[request_id] = tokenizer.decode(generated_ids, skip_special_tokens=True)
            outputs[f"{request_id}_finished"] = True
            msg = f"Generation finished (max tokens)."
            status_logs[request_id].append(msg)
            break
        if last_token is not None and last_token == tokenizer.eos_token_id:
            generated_ids = output_ids[0, prompt_len:]  # Only generated tokens
            outputs[request_id] = tokenizer.decode(generated_ids, skip_special_tokens=True)
            outputs[f"{request_id}_finished"] = True
            msg = f"Generation finished (EOS token)."
            status_logs[request_id].append(msg)
            break
        # Prepare for next token
        input_ids = output_ids
        position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long).unsqueeze(0)
        first_shard = os.path.join(shards_root, "shard_00")
        embed_tokens_state = torch.load(os.path.join(first_shard, "embed_tokens.pt"), map_location="cpu")
        embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        embed_tokens.load_state_dict(embed_tokens_state)
        hidden_states = embed_tokens(input_ids)
        # Route through all devices
        hidden_states_b64 = serialize_tensor(hidden_states)
        pending_work[0] = {
            "hidden_states": hidden_states_b64,
            "position_ids": position_ids.tolist(),
            "request_id": request_id
        }
        results.clear()
        results["input_ids"] = input_ids.tolist()
        results["position_ids"] = position_ids.tolist()
        results["request_id"] = request_id
        # Wait for all devices to finish (polling)
        while True:
            if outputs.get(f"{request_id}_ready"):
                outputs[f"{request_id}_ready"] = False
                break
            time.sleep(0.01)
        # After all devices, get logits and next token
        last_shard = os.path.join(shards_root, f"shard_{num_shards - 1:02d}")
        norm_state = torch.load(os.path.join(last_shard, "norm.pt"), map_location="cpu")
        norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        norm.load_state_dict(norm_state)
        hs = outputs[f"{request_id}_last_hidden_states"]
        hs = norm(hs)
        lm_head_state = torch.load(os.path.join(last_shard, "lm_head.pt"), map_location="cpu")
        if "bias" in lm_head_state:
            del lm_head_state["bias"]
        lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        lm_head.load_state_dict(lm_head_state)
        with torch.no_grad():
            logits = lm_head(hs)
        next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
        output_ids = torch.cat([input_ids, next_token_id], dim=-1)
        outputs[f"{request_id}_output_ids"] = output_ids
        outputs[f"{request_id}_step"] = step + 1
        outputs[f"{request_id}_last_token"] = next_token_id.item()
        msg = f"Generated token: {tokenizer.decode(next_token_id[0])}"
        status_logs[request_id].append(msg)
        step += 1
    return jsonify({"status": "pending"})

@app.route("/get_status", methods=["GET"])
def get_status():
    request_id = request.args["request_id"]
    log = status_logs.get(request_id, [])
    return jsonify({"log": log})

if __name__ == "__main__":
    app.run(port=5000, debug=True, use_reloader=False)