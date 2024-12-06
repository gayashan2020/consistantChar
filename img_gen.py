from flask import Flask, request, jsonify
import base64
import requests
import os
import time
from datetime import datetime

app = Flask(__name__)

WEBUI_SERVER_URL = 'http://127.0.0.1:7860'

OUT_DIR = 'api_generated_images'
TXT2IMG_DIR = os.path.join(OUT_DIR, 'txt2img')
IMG2IMG_DIR = os.path.join(OUT_DIR, 'img2img')
os.makedirs(TXT2IMG_DIR, exist_ok=True)
os.makedirs(IMG2IMG_DIR, exist_ok=True)

def timestamp():
    return datetime.fromtimestamp(time.time()).strftime("%Y%m%d-%H%M%S")

def save_image(base64_image, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{prefix}-{timestamp()}.png")
    with open(save_path, "wb") as f:
        f.write(base64.b64decode(base64_image))
    return save_path

def call_webui_api(endpoint, payload):
    response = requests.post(f'{WEBUI_SERVER_URL}/{endpoint}', json=payload)
    response.raise_for_status()
    return response.json()

@app.route('/txt2img', methods=['POST'])
def txt2img():
    try:
        payload = request.json
        response = call_webui_api('sdapi/v1/txt2img', payload)
        image_paths = []
        for index, image in enumerate(response.get('images', [])):
            save_path = save_image(image, TXT2IMG_DIR, f'txt2img-{index}')
            image_paths.append(save_path)
        return jsonify({"message": "Images generated successfully", "images": image_paths}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/img2img', methods=['POST'])
def img2img():
    try:
        payload = request.json
        response = call_webui_api('sdapi/v1/img2img', payload)
        image_paths = []
        for index, image in enumerate(response.get('images', [])):
            save_path = save_image(image, IMG2IMG_DIR, f'img2img-{index}')
            image_paths.append(save_path)
        return jsonify({"message": "Images generated successfully", "images": image_paths}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/txt2img-with-lora-controlnet', methods=['POST'])
def txt2img_with_lora_and_controlnet():
    try:
        payload = request.json

        lora_prompt = payload.pop('lora_prompt', '')
        controlnet_args = payload.pop('controlnet_args', [])

        if lora_prompt:
            payload['prompt'] = f"{payload.get('prompt', '')} {lora_prompt}"

        if controlnet_args:
            payload['alwayson_scripts'] = {
                'controlnet': {
                    'args': controlnet_args
                }
            }

        response = call_webui_api('sdapi/v1/txt2img', payload)
        image_paths = []
        for index, image in enumerate(response.get('images', [])):
            save_path = save_image(image, TXT2IMG_DIR, f'txt2img-lora-controlnet-{index}')
            image_paths.append(save_path)
        return jsonify({"message": "Images generated successfully with LoRA and ControlNet", "images": image_paths}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)