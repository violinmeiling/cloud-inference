from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import os
import uuid
import base64
import io
from transformers import AutoTokenizer, AutoConfig
from transformers.models.distilbert.modeling_distilbert import Embeddings

app = Flask(__name__)

# === STATE ===
pending_work = {}
results = {}
outputs = {}
status_logs = {}

num_shards = 2
shards_root = "/tmp/distill"
first_shard = os.path.join(shards_root, "shard_00")
last_shard = os.path.join(shards_root, f"shard_{num_shards - 1:02d}")

# === STATIC WEIGHTS ===
tokenizer = AutoTokenizer.from_pretrained(first_shard, use_fast=False, local_files_only=True)
config = AutoConfig.from_pretrained(first_shard, local_files_only=True)

# Load DistilBERT embeddings from first shard
embeddings = Embeddings(config)
embeddings.load_state_dict(torch.load(os.path.join(first_shard, "embeddings.pt"), map_location="cpu"))
embeddings.eval()

# Load classifier from last shard
classifier = nn.Linear(config.hidden_size, config.num_labels)
classifier.load_state_dict(torch.load(os.path.join(last_shard, "classifier.pt"), map_location="cpu"))
classifier.eval()

# Optional LayerNorm before classifier (DistilBERT does use LayerNorms inside layers, but
# if you saved a final norm state dict separately, load and use it here; else omit)
norm_path = os.path.join(last_shard, "norm.pt")
if os.path.exists(norm_path):
    norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    norm.load_state_dict(torch.load(norm_path, map_location="cpu"))
    norm.eval()
else:
    norm = None

# === UTILS ===
def serialize_tensor(tnsor):
    buf = io.BytesIO()
    torch.save(tensor, buf)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

def deserialize_tensor(b64str):
    buf = io.BytesIO(base64.b64decode(b64str))
    return torch.load(buf)

# === ROUTES ===
@app.route("/submit_prompt", methods=["POST"])
def submit_prompt():
    prompt = request.json["prompt"]
    request_id = str(uuid.uuid4())

    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True, add_special_tokens=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    hidden_states = embeddings(input_ids)

    pending_work[0] = {
        "hidden_states": serialize_tensor(hidden_states),
        "attention_mask": attention_mask.tolist(),
        "request_id": request_id
    }

    results["attention_mask"] = attention_mask.tolist()
    results["request_id"] = request_id

    outputs[request_id] = None
    outputs[f"{request_id}_last_hidden_states"] = None
    outputs[f"{request_id}_ready"] = False

    status_logs[request_id] = [f"Prompt submitted: {prompt[:50]}"]

    return jsonify({"status": "ok", "request_id": request_id})

@app.route("/get_work", methods=["GET"])
def get_work():
    device_id = int(request.args["device_id"])
    work = pending_work.pop(device_id, None)
    return jsonify(work or {})

@app.route("/submit_result", methods=["POST"])
def submit_result():
    device_id = int(request.json["device_id"])
    hidden_states = deserialize_tensor(request.json["hidden_states"])
    request_id = request.json.get("request_id", results.get("request_id"))

    status_logs.setdefault(request_id, []).append(f"Device {device_id} returned output.")

    if device_id < num_shards - 1:
        next_id = device_id + 1
        pending_work[next_id] = {
            "hidden_states": serialize_tensor(hidden_states),
            "attention_mask": results["attention_mask"],
            "request_id": request_id
        }
        status_logs[request_id].append(f"Routed to device {next_id}.")
    else:
        outputs[f"{request_id}_last_hidden_states"] = hidden_states
        outputs[f"{request_id}_ready"] = True
        status_logs[request_id].append("Shard chain done — ready for classification.")

    return jsonify({"status": "ok"})

@app.route("/get_result", methods=["GET"])
def get_result():
    request_id = request.args["request_id"]

    if outputs.get(request_id) is not None:
        return jsonify({"status": "done", "output": outputs[request_id]})

    if not outputs.get(f"{request_id}_ready", False):
        return jsonify({"status": "pending"})

    hs = outputs[f"{request_id}_last_hidden_states"]

    if not isinstance(hs, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor for hidden states, got {type(hs)}")

    if norm is not None:
        print(f"hs shape before norm: {hs.shape}")
        hs = norm(hs)
        print(f"hs shape after norm: {hs.shape}")

    # Use the first token's hidden state as pooled output (DistilBERT style)
    pooled_output = hs[:, 0, :]  # batch_size x hidden_size

    with torch.no_grad():
        logits = classifier(pooled_output)
        pred_id = logits.argmax(dim=-1).item()

    label = config.id2label.get(str(pred_id), config.id2label.get(pred_id, f"Unknown: {pred_id}"))
    outputs[request_id] = label
    status_logs[request_id].append(f"Prediction: {label}")

    print(f"logits: {logits}")
    print(f"pred_id: {pred_id}")
    print(f"id2label: {config.id2label}")
    print(classifier.weight)

    return jsonify({"status": "done", "output": label})

@app.route("/get_status", methods=["GET"])
def get_status():
    request_id = request.args["request_id"]
    return jsonify({"log": status_logs.get(request_id, [])})

if __name__ == "__main__":
    app.run(port=5000, debug=True, use_reloader=False)
