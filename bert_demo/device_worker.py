import requests
import torch
import torch.nn as nn
import os
import sys
import time
import json
import boto3
import base64
import io
from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertLayer as LayerClass

DEVICE_ID = int(sys.argv[1])
BACKEND_URL = "http://127.0.0.1:5000"
LOCAL_SHARD_DIR = f"/tmp/bert/shard_{DEVICE_ID:02d}/"
BUCKET = "model-shards"
SHARD_PREFIX = f"bert/shard_{DEVICE_ID:02d}/"
LAYERS_PER_SHARD = 6

def download_shard_from_s3(bucket, shard_prefix, local_dir):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=shard_prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if key.endswith('/'):
                continue
            rel_path = os.path.relpath(key, shard_prefix)
            local_path = os.path.join(local_dir, rel_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if os.path.exists(local_path):
                print(f"Skipping {key}, already exists at {local_path}")
                continue
            print(f"Downloading {key} to {local_path}")
            s3.download_file(bucket, key, local_path)

if not os.path.exists(os.path.join(LOCAL_SHARD_DIR, "layers.pt")):
    print(f"Downloading shard {DEVICE_ID} from S3...")
    download_shard_from_s3(BUCKET, SHARD_PREFIX, LOCAL_SHARD_DIR)
else:
    print(f"Shard {DEVICE_ID} already present locally.")

config = AutoConfig.from_pretrained(LOCAL_SHARD_DIR, local_files_only=True)
layers_state = torch.load(os.path.join(LOCAL_SHARD_DIR, "layers.pt"), map_location="cpu")
layers = nn.ModuleList([LayerClass(config) for _ in range(LAYERS_PER_SHARD)])
layers.load_state_dict(layers_state)
layers.eval()

print(f"Device {DEVICE_ID} ready and polling for work.")

while True:
    try:
        resp = requests.get(f"{BACKEND_URL}/get_work", params={"device_id": DEVICE_ID})
        if resp.status_code != 200:
            print(f"Device {DEVICE_ID}: Backend returned status {resp.status_code}")
            time.sleep(1)
            continue
        try:
            work = resp.json()
        except Exception as e:
            print(f"Device {DEVICE_ID}: Failed to decode JSON: {e}")
            time.sleep(1)
            continue

        if work.get("hidden_states") is not None:
            print(f"Device {DEVICE_ID} got work.")

            buf = io.BytesIO(base64.b64decode(work["hidden_states"]))
            hidden_states = torch.load(buf)

            attention_mask = torch.tensor(work["attention_mask"]).float()  # (1, seq_len)
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)      # (1, 1, 1, seq_len)
            attention_mask = (1.0 - attention_mask) * -10000.0

            with torch.no_grad():
                for layer in layers:
                    hidden_states = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        head_mask=None
                    )[0]
                    print(f"Device {DEVICE_ID} output shape: {hidden_states.shape}")


            out_buf = io.BytesIO()
            torch.save(hidden_states, out_buf)
            out_buf.seek(0)
            hidden_states_b64 = base64.b64encode(out_buf.read()).decode("utf-8")

            requests.post(f"{BACKEND_URL}/submit_result", json={
                "device_id": DEVICE_ID,
                "hidden_states": hidden_states_b64,
                "request_id": work.get("request_id")
            })


            out_buf = io.BytesIO()
            torch.save(hidden_states, out_buf)
            out_buf.seek(0)
            hidden_states_b64 = base64.b64encode(out_buf.read()).decode("utf-8")

            requests.post(f"{BACKEND_URL}/submit_result", json={
                "device_id": DEVICE_ID,
                "hidden_states": hidden_states_b64,
                "request_id": work.get("request_id")
            })

    except Exception as e:
        print(f"Device {DEVICE_ID}: Exception during polling: {e}")
    time.sleep(1)
