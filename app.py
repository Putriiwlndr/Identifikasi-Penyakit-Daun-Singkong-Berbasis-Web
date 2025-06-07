import base64
import os
import sqlite3
from collections import OrderedDict
from io import BytesIO

import timm
import torch
import torch.nn as nn
from flask import Flask, redirect, render_template, request, url_for
from PIL import Image
from torchvision import transforms

# --- Konfigurasi Flask ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('tf_efficientnet_b6_ns', pretrained=False, num_classes=5)

# Load checkpoint
model_path = 'model/tf_efficientnet_b6.ns_jft_in1k_fold_0_best.pth'
checkpoint = torch.load(model_path, map_location=device)

# Periksa apakah checkpoint berisi state_dict langsung atau dict lengkap
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# Bersihkan nama key jika mengandung 'model.'
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace('model.', '', 1) if k.startswith('model.') else k
    new_state_dict[name] = v

# Load ke model
model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

print("✅ Model berhasil dimuat dan siap digunakan.")

# --- Kelas Penyakit ---
class_names = [
    'Cassava Bacterial Blight (CBB)',
    'Cassava Brown Streak Disease (CBSD)',
    'Cassava Green Mottle (CGM)',
    'Cassava Mosaic Disease (CMD)',
    'Healthy'
]

# --- Transformasi Gambar ---
transform = transforms.Compose([
    transforms.Resize((528, 528)),
    transforms.ToTensor(),
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

# --- Fungsi Prediksi ---
def predict_image(img_path):
    image = Image.open(img_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return class_names[predicted.item()]

# --- Inisialisasi DB SQLite ---
def init_db():
    with sqlite3.connect('cassava.db') as conn:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                prediction TEXT
            )
        ''')
        conn.commit()

# --- Halaman Utama ---
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('image')
        if file and file.filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            prediction = predict_image(filepath)

            # Simpan ke database
            with sqlite3.connect('cassava.db') as conn:
                c = conn.cursor()
                c.execute("INSERT INTO predictions (filename, prediction) VALUES (?, ?)",
                          (file.filename, prediction))
                conn.commit()

            return render_template('result.html', prediction=prediction, image_url=filepath)

    return render_template('index.html')

# --- Halaman Riwayat ---
@app.route('/history')
def history():
    with sqlite3.connect('cassava.db') as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM predictions ORDER BY id DESC")
        data = c.fetchall()
    return render_template('history.html', data=data)

@app.route('/upload_camera', methods=['POST'])
def upload_camera():
    data_url = request.form['camera_image']
    header, encoded = data_url.split(",", 1)
    image_data = base64.b64decode(encoded)
    image = Image.open(BytesIO(image_data)).convert("RGB")

    # Simpan file dari kamera
    filename = 'camera_capture.png'
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    # Prediksi dan simpan ke DB
    prediction = predict_image(filepath)
    with sqlite3.connect('cassava.db') as conn:
        c = conn.cursor()
        c.execute("INSERT INTO predictions (filename, prediction) VALUES (?, ?)",
                  (filename, prediction))
        conn.commit()

    return render_template('result.html', prediction=prediction, image_url=filepath)

@app.route('/result/<filename>')
def show_result(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    if not os.path.exists(filepath):
        return "File tidak ditemukan", 404

    prediction = predict_image(filepath)
    image_url = url_for('static', filename='uploads/' + filename)

    return render_template('result.html', prediction=prediction, image_url=image_url)



# --- Jalankan Aplikasi ---
if __name__ == '__main__':
    init_db()
    app.run(debug=True)
