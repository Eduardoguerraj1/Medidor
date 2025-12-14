from flask import Flask, request, jsonify
from PIL import Image
import io
import os

app = Flask(__name__)

model = None
OCR_CONFIG = '--psm 6 -c tessedit_char_whitelist=0123456789.,'

@app.route("/")
def health():
    return "OK"

@app.route("/predict", methods=["POST"])
def predict():
    global model

    # imports pesados só quando precisa
    import cv2
    import numpy as np
    import pytesseract
    from ultralytics import YOLO

    if model is None:
        model = YOLO("best.pt")
        model.to("cpu")

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
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
