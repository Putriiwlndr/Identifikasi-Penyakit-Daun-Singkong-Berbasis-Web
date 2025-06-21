import base64
import os
import re
import sqlite3
import uuid
from collections import OrderedDict
from io import BytesIO

import gdown
import timm
import torch
import torch.nn as nn
import wikipedia
from deep_translator import GoogleTranslator
from flask import Flask, flash, redirect, render_template, request, url_for
from PIL import Image
from torchvision import transforms

# --- Mapping Nama Wikipedia ---
wiki_mapping = {
    'Cassava Bacterial Blight (CBB)': 'Cassava bacterial blight',
    'Cassava Brown Streak Disease (CBSD)': 'Cassava brown streak virus',
    'Cassava Green Mottle (CGM)': 'Cassava green mottle virus',
    'Cassava Mosaic Disease (CMD)': 'Cassava mosaic virus',
    'Healthy': 'Cassava'
}

# --- Konfigurasi Flask ---
app = Flask(__name__)
app.secret_key = 'cassava_leaf_disease_123456'
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model('tf_efficientnet_b6_ns', pretrained=False, num_classes=5)

# Download dari Google Drive jika belum ada
model_local_path = 'model.pth'
file_id = '1GwQD_XUd2ku4HUi9yFeypWej_DdFXER5'
gdown_url = f'https://drive.google.com/uc?id={file_id}'
if not os.path.exists(model_local_path):
    gdown.download(gdown_url, model_local_path, quiet=False)

checkpoint = torch.load(model_local_path, map_location=device)
state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
new_state_dict = OrderedDict((k.replace('model.', '', 1) if k.startswith('model.') else k, v)
                             for k, v in state_dict.items())
model.load_state_dict(new_state_dict)
model.to(device)
model.eval()

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

# --- Ambil Ringkasan dan Pencegahan Wikipedia ---
wikipedia.set_lang("en")
def get_wikipedia_summary(prediction):
    wiki_title = wiki_mapping.get(prediction, prediction)
    try:
        page = wikipedia.page(wiki_title, auto_suggest=False)
        summary = wikipedia.summary(wiki_title, sentences=3, auto_suggest=False)

        content = page.content.lower()
        prevention_sentences = re.findall(
            r'([^.]?(mencegah|pencegahan|pengendalian|menghindari|control)[^.]\.)', content)
        prevention = prevention_sentences[0][0].capitalize() if prevention_sentences else "Informasi pencegahan tidak ditemukan."

        translated_summary = GoogleTranslator(source='en', target='id').translate(summary)
        translated_prevention = GoogleTranslator(source='en', target='id').translate(prevention)

    except Exception as e:
        print("❌ Wikipedia Error:", e)
        translated_summary = "Informasi tidak tersedia di Wikipedia."
        translated_prevention = "Informasi pencegahan tidak tersedia."

    return translated_summary, translated_prevention

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
        if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            wiki_summary, wiki_prevention = get_wikipedia_summary(prediction)

            with sqlite3.connect('cassava.db') as conn:
                c = conn.cursor()
                c.execute("INSERT INTO predictions (filename, prediction) VALUES (?, ?)",
                          (filename, prediction))
                conn.commit()

            return render_template('result.html', prediction=prediction, image_url=filepath,
                                   wiki_summary=wiki_summary, wiki_prevention=wiki_prevention)
        else:
            flash('Mohon unggah gambar dalam format PNG, JPG, atau JPEG.', 'danger')
    return render_template('index.html')

# --- Halaman Riwayat ---
@app.route('/history')
def history():
    with sqlite3.connect('cassava.db') as conn:
        c = conn.cursor()
        c.execute("SELECT id, filename, prediction FROM predictions ORDER BY id DESC")
        data = c.fetchall()
    return render_template('history.html', data=data)

# --- Upload dari Kamera ---
@app.route('/upload_camera', methods=['POST'])
def upload_camera():
    data_url = request.form['camera_image']
    header, encoded = data_url.split(",", 1)
    image_data = base64.b64decode(encoded)
    image = Image.open(BytesIO(image_data)).convert("RGB")

    filename = f"{uuid.uuid4().hex}_camera.png"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)

    prediction = predict_image(filepath)
    wiki_summary, wiki_prevention = get_wikipedia_summary(prediction)

    with sqlite3.connect('cassava.db') as conn:
        c = conn.cursor()
        c.execute("INSERT INTO predictions (filename, prediction) VALUES (?, ?)",
                  (filename, prediction))
        conn.commit()

    return render_template('result.html', prediction=prediction, image_url=filepath,
                           wiki_summary=wiki_summary, wiki_prevention=wiki_prevention)

# --- Tampilkan Hasil ---
@app.route('/result/<filename>')
def show_result(filename):
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(filepath):
        return "File tidak ditemukan", 404
    prediction = predict_image(filepath)
    wiki_summary, wiki_prevention = get_wikipedia_summary(prediction)
    image_url = url_for('static', filename='uploads/' + filename)
    return render_template('result.html', prediction=prediction, image_url=image_url,
                           wiki_summary=wiki_summary, wiki_prevention=wiki_prevention)

# --- Hapus Riwayat ---
@app.route('/delete_history', methods=['POST'])
def delete_history():
    record_id = request.form['id']
    with sqlite3.connect('cassava.db') as conn:
        c = conn.cursor()
        c.execute("SELECT filename FROM predictions WHERE id = ?", (record_id,))
        result = c.fetchone()
        if result:
            filename = result[0]
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            c.execute("DELETE FROM predictions WHERE id = ?", (record_id,))
            conn.commit()
            c.execute("SELECT COUNT(*) FROM predictions WHERE filename = ?", (filename,))
            count = c.fetchone()[0]
            if count == 0 and os.path.exists(filepath):
                os.remove(filepath)
            flash('Data berhasil dihapus.', 'success')
        else:
            flash('Data tidak ditemukan.', 'danger')
    return redirect(url_for('history'))

# --- Jalankan Aplikasi ---
if __name__ == '__main__':
    init_db()
    app.run(debug=True)