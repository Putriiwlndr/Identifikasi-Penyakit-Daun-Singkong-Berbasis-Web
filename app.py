
# ===== 0. Import ============================================================
import base64
import os
import re
import sqlite3
from collections import OrderedDict
from io import BytesIO

import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
import wikipedia
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from deep_translator import GoogleTranslator
from flask import (Flask, flash, jsonify, redirect, render_template, request,
                   url_for)
from PIL import Image

# ===== 1. Hyper‑parameter & Konfigurasi =====================================
CFG = {
    "model_arch": "tf_efficientnet_b6_ns",
    "img_size": 512,
    "mean": (0.485, 0.456, 0.406),
    "std":  (0.229, 0.224, 0.225),
    "num_classes": 5
}
WEIGHT_PATH = 'model/tf_efficientnet_b6.ns_jft_in1k_fold_0_best.pth'

CLASS_NAMES = [
    'Cassava Bacterial Blight (CBB)',
    'Cassava Brown Streak Disease (CBSD)',
    'Cassava Green Mottle (CGM)',
    'Cassava Mosaic Disease (CMD)',
    'Healthy'
]

WIKI_MAPPING = {
    'Cassava Bacterial Blight (CBB)': 'Bacterial blight of cassava',
    'Cassava Brown Streak Disease (CBSD)': 'Cassava brown streak virus',
    'Cassava Green Mottle (CGM)': 'Cassava green mottle virus',
    'Cassava Mosaic Disease (CMD)': 'Cassava mosaic virus',
    'Healthy': 'Cassava'
}
wikipedia.set_lang("en")   # siapkan Wikipedia berbahasa Inggris

# ===== 2. Model =============================================================
class CassavaClassifier(nn.Module):
    """EfficientNet wrapper – layer FC otomatis menyesuaikan jumlah kelas."""
    def __init__(self, model_arch: str, n_class: int, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            model_arch, pretrained=pretrained, num_classes=n_class
        )

    def forward(self, x):
        return self.backbone(x)

# ===== 3. Transform & Utilities ============================================
def get_transforms():
    return Compose([
        Resize(CFG["img_size"], CFG["img_size"]),
        Normalize(CFG["mean"], CFG["std"]),
        ToTensorV2()
    ])

def read_rgb(path: str):
    """Baca gambar pakai OpenCV kemudian konversi BGR→RGB."""
    img = cv2.imread(path)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def predict_np(img_np: np.ndarray) -> int:
    """Inferensi pada ndarray RGB, mengembalikan indeks kelas."""
    tensor = TRANSFORMS(image=img_np)["image"].unsqueeze(0).to(DEVICE).float()
    with torch.no_grad():
        logits = MODEL(tensor)
        return torch.argmax(logits, 1).item()

def predict_image(path: str) -> str:
    img = read_rgb(path)
    if img is None:
        raise ValueError("Gambar tidak dapat dibaca.")
    idx = predict_np(img)
    return CLASS_NAMES[idx]

def get_wiki_info(prediction: str):
    """Ambil ringkasan, gejala, dan pencegahan dari Wikipedia (diterjemahkan)."""
    title = WIKI_MAPPING.get(prediction, prediction)
    try:
        page = wikipedia.page(title, auto_suggest=False)
        summary_en = wikipedia.summary(title, sentences=3, auto_suggest=False)

        content = page.content.lower()

        # ====== Ekstraksi GEJALA ======
        m_symptom = re.search(r'([^.]*?(symptom|sign)[^.]*\.)', content)
        symptom_en = m_symptom.group(1).capitalize() if m_symptom else "Symptom information not found."

        # ====== Ekstraksi PENCEGAHAN/PENGENDALIAN ======
        m_prevention = re.search(r'([^.]*?(control|prevent|management)[^.]*\.)', content)
        prevention_en = m_prevention.group(1).capitalize() if m_prevention else "Prevention information not found."

        # ====== Terjemahkan ke Bahasa Indonesia ======
        summary_id = GoogleTranslator("en", "id").translate(summary_en)
        symptom_id = GoogleTranslator("en", "id").translate(symptom_en)
        prevention_id = GoogleTranslator("en", "id").translate(prevention_en)

    except Exception as e:
        print("❌ Wikipedia error:", e)
        summary_id = "Informasi tidak tersedia di Wikipedia."
        symptom_id = "Informasi gejala tidak tersedia."
        prevention_id = "Informasi pencegahan tidak tersedia."

    return summary_id, symptom_id, prevention_id


# ===== 4. Inisialisasi Global =====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1) Bangun model persis seperti saat training
MODEL = timm.create_model(
    CFG["model_arch"],
    pretrained=False,
    num_classes=CFG["num_classes"]
).to(DEVICE)

# 2) Muat weight
ckpt  = torch.load(WEIGHT_PATH, map_location=DEVICE)
state = ckpt.get("state_dict", ckpt)
state = {k.replace("model.", "").replace("module.", ""): v for k, v in state.items()}
MODEL.load_state_dict(state, strict=False)   # strict=False biarkan head beda kelas
MODEL.eval()

TRANSFORMS = get_transforms()


# ===== 5. Flask Setup =======================================================
app = Flask(__name__)
app.secret_key = "cassava_leaf_disease_123456"
app.config["UPLOAD_FOLDER"] = "static/uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# ===== 6. SQLite ============================================================
def init_db():
    with sqlite3.connect("cassava.db") as conn:
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT,
                prediction TEXT
            )
        """)
        conn.commit()

# ===== 7. Routes ============================================================
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        f = request.files.get("image")
        if not f or f.filename == "":
            flash("Harap pilih gambar.", "danger")
            return redirect(url_for("index"))

        save_path = os.path.join(app.config["UPLOAD_FOLDER"], f.filename)
        f.save(save_path)

        try:
            pred = predict_image(save_path)
        except Exception as e:
            flash(str(e), "danger")
            return redirect(url_for("index"))

        ringkasan, gejala, pencegahan = get_wiki_info(pred)

        with sqlite3.connect("cassava.db") as conn:
            conn.execute("INSERT INTO predictions (filename, prediction) VALUES (?, ?)",
                         (f.filename, pred))
            conn.commit()

        return render_template("result.html", prediction=pred,
                       image_url=save_path,
                       wiki_summary=ringkasan,
                       wiki_symptom=gejala,
                       wiki_prevention=pencegahan)

    return render_template("index.html")

@app.route("/upload_camera", methods=["POST"])
def upload_camera():
    data_url = request.form["camera_image"]
    _, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")

    filename = "camera_capture.png"
    save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    pil_img.save(save_path)

    pred = predict_image(save_path)
    ringkasan, gejala, pencegahan = get_wiki_info(pred)

    with sqlite3.connect("cassava.db") as conn:
        conn.execute("INSERT INTO predictions (filename, prediction) VALUES (?, ?)",
                     (filename, pred))
        conn.commit()

    return render_template("result.html", prediction=pred,
                       image_url=save_path,
                       wiki_summary=ringkasan,
                       wiki_symptom=gejala,
                       wiki_prevention=pencegahan)

@app.route("/history")
def history():
    with sqlite3.connect("cassava.db") as conn:
        data = conn.execute("SELECT id, filename, prediction FROM predictions ORDER BY id DESC").fetchall()
    return render_template("history.html", data=data)

@app.route("/result/<filename>")
def show_result(filename):
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    if not os.path.exists(path):
        return "File tidak ditemukan", 404
    pred = predict_image(path)
    ringkasan, pencegahan = get_wiki_info(pred)
    img_url = url_for("static", filename="uploads/" + filename)
    return render_template("result.html", prediction=pred,
                           image_url=img_url,
                           wiki_summary=ringkasan,
                           wiki_prevention=pencegahan)

@app.route("/delete_history", methods=["POST"])
def delete_history():
    rec_id = request.form["id"]
    with sqlite3.connect("cassava.db") as conn:
        cur = conn.cursor()
        cur.execute("SELECT filename FROM predictions WHERE id=?", (rec_id,))
        row = cur.fetchone()
        if row:
            filename = row[0]
            path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            cur.execute("DELETE FROM predictions WHERE id=?", (rec_id,))
            conn.commit()

            # Hapus file fisik bila tak dipakai entry lain
            cnt = cur.execute("SELECT COUNT(*) FROM predictions WHERE filename=?",
                              (filename,)).fetchone()[0]
            if cnt == 0 and os.path.exists(path):
                os.remove(path)
            flash("Data berhasil dihapus.", "success")
        else:
            flash("Data tidak ditemukan.", "danger")
    return redirect(url_for("history"))

# ===== 8. API JSON opsional (tanpa antarmuka) ===============================
@app.route("/api/predict", methods=["POST"])
def api_predict():
    if "file" not in request.files:
        return jsonify(error="Parameter 'file' tidak ditemukan"), 400
    f = request.files["file"]
    if f.filename == "":
        return jsonify(error="Nama file kosong"), 400

    tmp_dir = "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, f.filename)
    f.save(tmp_path)

    img = read_rgb(tmp_path)
    os.remove(tmp_path)
    if img is None:
        return jsonify(error="Gagal membaca gambar"), 400

    label_idx = predict_np(img)
    return jsonify(predicted_label=label_idx, predicted_name=CLASS_NAMES[label_idx]), 200

# ===== 9. Main ==============================================================
if __name__ == "__main__":
    init_db()
    app.run(debug=True, host="0.0.0.0", port=5000)
