import os
import json
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-14B-Instruct")
args = parser.parse_args()

path = f"/root/.cache/huggingface/hub/models--{args.model_path.replace('/', '--')}/snapshots"

# Find the actual snapshot subdirectory within the path
subdirs = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
if not subdirs:
    raise FileNotFoundError(f"No snapshot directory found in {path}")

# Assuming there's only one snapshot directory, or we want the first one
folder = subdirs[0]

config_path = os.path.join(path, folder, "config.json")
config = json.load(open(config_path, "r"))

config["rope_scaling"] = {
    "factor": 4.0,
    "original_max_position_embeddings": 32768,
    "type": "yarn"
}

with open(config_path, "w") as f:
    json.dump(config, f, indent=4)