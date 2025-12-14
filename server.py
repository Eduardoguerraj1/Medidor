from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract
import io
from PIL import Image

app = Flask(__name__)
model = YOLO("best.pt")
model.to("cpu")

OCR_CONFIG = '--psm 6 -c tessedit_char_whitelist=0123456789.,'

@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["image"]
    img = Image.open(io.BytesIO(file.read()))
    frame = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    results = model.predict(frame, imgsz=320, device="cpu")
    valores = []

    for box in results[0].boxes.xyxy:
        x1, y1, x2, y2 = map(int, box)
        crop = frame[max(0,y1-30):y1, x1:x2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray, config=OCR_CONFIG).strip()
        valores.append(text)

    return jsonify({"valores": valores})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

