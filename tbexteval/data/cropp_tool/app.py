from flask import Flask, send_from_directory, jsonify, request, render_template
import os
from werkzeug.utils import secure_filename
import base64
import uuid

app = Flask(__name__)

IMAGES_DIR = './images'
CROPPED_IMAGES_DIR = './cropped_images'

os.makedirs(CROPPED_IMAGES_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/images')
def list_images():
    image_files = [f for f in os.listdir(IMAGES_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    return jsonify(image_files)

@app.route('/image/<filename>')
def get_image(filename):
    return send_from_directory(IMAGES_DIR, filename)

@app.route('/save_cropped', methods=['POST'])
def save_cropped():
    data = request.json
    image_data = data['image']
    original_filename = data['filename']

    header, encoded = image_data.split(",", 1)
    file_extension = header.split('/')[1].split(';')[0]  # e.g., 'png'
    imgdata = base64.b64decode(encoded)

    safe_filename = secure_filename(original_filename)
    name, ext = os.path.splitext(safe_filename)
    unique_filename = f"{name}_{uuid.uuid4().hex}.{file_extension}"
    save_path = os.path.join(CROPPED_IMAGES_DIR, unique_filename)

    with open(save_path, 'wb') as f:
        f.write(imgdata)

    return jsonify({'status': 'success', 'filename': unique_filename})

if __name__ == '__main__':
    app.run(debug=False)
