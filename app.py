import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw
from tensorflow.keras.models import load_model
import os
from datetime import datetime
import unicodedata
import re

# === CONFIG ===
MODEL_PATH = "model.h5"
DATA_DIR = "duancuoiki"
IMG_SIZE = (128, 128)
CONF_THRESHOLD = 0.6

# === LOAD MODEL ===
model = load_model(MODEL_PATH)
class_names = sorted(os.listdir(DATA_DIR))

# === BẢNG GIÁ ===
PRICE_TABLE = {
    "COM": 10000,
    "CA KHO": 30000,
    "TRUNG CHIEN": 25000,
    "RAU XAO": 10000,
    "CANH RAU CAI": 7000,
    "THIT KHO": 25000,
    "THIT KHO TRUNG": 30000,
    "CANH CHUA CO CA": 25000,
    "CANH CHUA KHONG CA": 20000,
    "SUON NUONG": 30000,
    "DAU HU SOT CA": 20000
}


def normalize_name(name):
    """Chuẩn hóa tên lớp để so với bảng giá"""
    name = name.strip().upper()
    name = unicodedata.normalize('NFD', name)
    name = re.sub(r'[\u0300-\u036f]', '', name)
    name = name.replace("_", " ").replace("-", " ")
    name = re.sub(r'\s+', ' ', name)
    return name


def detect_food_regions(image_bgr):
    """Cắt 5 vùng khay cơm: Canh, Cơm, Rau, Thịt, Trứng"""
    h, w = image_bgr.shape[:2]
    return [
        ("Canh", image_bgr[int(h * 0.03):int(h * 0.43), int(w * 0.02):int(w * 0.48)]),
        ("Cơm", image_bgr[int(h * 0.03):int(h * 0.43), int(w * 0.55):int(w * 0.98)]),
        ("Rau", image_bgr[int(h * 0.55):int(h * 0.97), int(w * 0.70):int(w * 0.97)]),
        ("Thịt", image_bgr[int(h * 0.55):int(h * 0.97), int(w * 0.02):int(w * 0.30)]),
        ("Trứng", image_bgr[int(h * 0.55):int(h * 0.97), int(w * 0.40):int(w * 0.60)])
    ]


def predict_food(pil_img):
    """Dự đoán món ăn"""
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    parts = detect_food_regions(img_bgr)

    total_price = 0
    dish_details = []
    result_images = []

    for name, roi in parts:
        resized = cv2.resize(roi, IMG_SIZE)
        arr = np.expand_dims(resized / 255.0, axis=0)
        preds = model.predict(arr, verbose=0)[0]

        idx = np.argmax(preds)
        class_name = class_names[idx]
        conf = float(preds[idx])

        norm_name = normalize_name(class_name)
        matched_price = 0
        for key, val in PRICE_TABLE.items():
            if key in norm_name:
                matched_price = val
                break

        if conf >= CONF_THRESHOLD:
            total_price += matched_price

        dish_details.append({
            "slot": name,
            "food": class_name,
            "conf": conf,
            "price": matched_price
        })

        # Vẽ overlay text trên từng ROI
        roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(roi_pil)
        draw.rectangle([(0, 0), (roi_pil.width, 60)], fill=(0, 0, 0, 160))
        draw.text((10, 10), f"{class_name} ({conf*100:.1f}%)", fill="lime")
        result_images.append(roi_pil)

    return dish_details, total_price, result_images


# === GIAO DIỆN STREAMLIT ===
st.set_page_config(page_title="🍱 Nhận diện khay cơm", page_icon="🍱")
st.title("🍱 Nhận diện khay cơm & Tính tiền")

uploaded_file = st.file_uploader("📸 Upload ảnh khay cơm", type=["jpg", "png", "jpeg"])
use_webcam = st.checkbox("🎥 Dùng webcam", value=False)

captured_image = None

if use_webcam:
    st.info("🔹 Mở webcam và chụp khay cơm, sau đó nhấn 'Phân tích'.")
    camera_image = st.camera_input("📷 Webcam")
    analyze = st.button("🔍 Phân tích ảnh webcam")
    if analyze and camera_image:
        captured_image = Image.open(camera_image)
elif uploaded_file:
    captured_image = Image.open(uploaded_file)
    analyze = st.button("🔍 Phân tích ảnh tải lên")
else:
    analyze = st.button("🔍 Phân tích")

# === XỬ LÝ PHÂN TÍCH ===
if analyze and captured_image is not None:
    st.image(captured_image, caption="Ảnh khay cơm")
    dish_details, total_price, result_images = predict_food(captured_image)

    st.subheader("🍽️ Các món phát hiện được:")
    for d in dish_details:
        st.write(f"- {d['food']} ({d['conf'] * 100:.1f}%) — {d['price']:,}đ")

    st.markdown(f"### 💰 **Tổng tiền: {total_price:,} VNĐ**")

    st.subheader("📸 Ảnh từng vùng món ăn:")
    st.image(result_images, width=200)

    # Lưu ảnh nếu cần
    filename = f"tray_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
    captured_image.save(filename)
    st.info(f"💾 Ảnh đã lưu: {filename}")

elif analyze:
    st.warning("⚠️ Hãy tải ảnh hoặc bật webcam trước khi phân tích.")
