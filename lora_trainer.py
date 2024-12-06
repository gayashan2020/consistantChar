from flask import Flask, request, jsonify
import os
import re
import toml
from time import time

app = Flask(__name__)

globals().update({
    "dependencies_installed": False,
    "model_file": None,
    "COLAB": True,
    "XFORMERS": True,
    "optimizer": "AdamW8bit",
    "optimizer_args": None,
    "project_name": "",
    "old_model_url": None,
})

root_dir = "/content" if COLAB else "~/Loras"
deps_dir = os.path.join(root_dir, "deps")
repo_dir = os.path.join(root_dir, "kohya-trainer")
main_dir = os.path.join(root_dir, "drive/MyDrive/Loras") if COLAB else root_dir

@app.route("/configure", methods=["POST"])
def configure():
    try:
        data = request.json

        globals().update({
            "project_name": data.get("project_name", ""),
            "folder_structure": data.get("folder_structure", "Organize by project"),
            "training_model": data.get("training_model", "Anime"),
            "optional_custom_training_model_url": data.get("optional_custom_training_model_url", ""),
            "resolution": data.get("resolution", 512),
            "flip_aug": data.get("flip_aug", False),
            "caption_extension": data.get("caption_extension", ".txt"),
            "shuffle_tags": data.get("shuffle_tags", True),
            "activation_tags": data.get("activation_tags", "1"),
            "num_repeats": data.get("num_repeats", 10),
            "preferred_unit": data.get("preferred_unit", "Epochs"),
            "how_many": data.get("how_many", 10),
            "train_batch_size": data.get("train_batch_size", 2),
            "unet_lr": data.get("unet_lr", 5e-4),
            "text_encoder_lr": data.get("text_encoder_lr", 1e-4),
            "lr_scheduler": data.get("lr_scheduler", "cosine_with_restarts"),
            "lr_scheduler_number": data.get("lr_scheduler_number", 3),
            "min_snr_gamma": data.get("min_snr_gamma", True),
            "network_dim": data.get("network_dim", 16),
            "network_alpha": data.get("network_alpha", 8),
            "conv_dim": data.get("conv_dim", 8),
            "conv_alpha": data.get("conv_alpha", 4),
            "lora_type": data.get("lora_type", "LoRA"),
        })

        return jsonify({"message": "Configuration updated successfully."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/train", methods=["POST"])
def train():
    try:
        validate_dataset()

        if not dependencies_installed:
            install_dependencies()

        download_model()
        create_config()

        os.chdir(repo_dir)
        os.system(
            f"accelerate launch --config_file={repo_dir}/accelerate_config/config.yaml --num_cpu_threads_per_process=1 train_network_wrapper.py --dataset_config={repo_dir}/dataset_config.toml --config_file={repo_dir}/training_config.toml"
        )

        return jsonify({"message": "Training started."}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def install_dependencies():
    os.chdir(root_dir)
    if not os.path.exists(repo_dir):
        os.system(f"git clone https://github.com/uYouUs/sd-scripts {repo_dir}")
    os.chdir(repo_dir)

    os.system("pip install -r requirements.txt")
    globals()["dependencies_installed"] = True

def validate_dataset():
    if not project_name.strip():
        raise ValueError("Project name is invalid.")

def download_model():
    global model_file

    if optional_custom_training_model_url:
        model_file = f"/content/{os.path.basename(optional_custom_training_model_url)}"
        os.system(f"wget {optional_custom_training_model_url} -O {model_file}")
    else:
        model_file = f"{main_dir}/default_model.safetensors"

    if not os.path.exists(model_file):
        raise FileNotFoundError("Model file could not be downloaded or found.")

def create_config():
    config_dict = {
        "additional_network_arguments": {
            "unet_lr": unet_lr,
            "text_encoder_lr": text_encoder_lr,
            "network_dim": network_dim,
            "network_alpha": network_alpha,
        },
        "training_arguments": {
            "train_batch_size": train_batch_size,
            "resolution": resolution,
        },
    }

    config_file_path = f"{repo_dir}/training_config.toml"
    with open(config_file_path, "w") as f:
        toml.dump(config_dict, f)

    return True

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
