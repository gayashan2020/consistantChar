from flask import Flask, request, jsonify
import requests
import base64
import os

app = Flask(__name__)

YOLO_SERVER_URL = 'http://127.0.0.1:5000'

SEGMENT_OUT_DIR = 'segmented_images'
os.makedirs(SEGMENT_OUT_DIR, exist_ok=True)

def save_segmented_image(image_base64, output_name):
    file_path = os.path.join(SEGMENT_OUT_DIR, output_name)
    with open(file_path, "wb") as file:
        file.write(base64.b64decode(image_base64))
    return file_path

@app.route('/segment', methods=['POST'])
def segment_image():
    try:
        data = request.json
        image_path = data.get('image_path')
        save_txt = data.get('save_txt', 'F')

        if not os.path.exists(image_path):
            return jsonify({"error": "Image file not found."}), 404

        with open(image_path, 'rb') as file:
            encoded_image = base64.b64encode(file.read()).decode('utf-8')

        response = requests.get(
            f"{YOLO_SERVER_URL}/predict",
            params={
                "source": encoded_image,
                "save_txt": save_txt
            },
            verify=False
        )

        if response.status_code != 200:
            return jsonify({"error": f"Failed to process image: {response.text}"}), response.status_code

        results = response.json().get('results', [])
        return jsonify({"message": "Segmentation successful", "results": results}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
