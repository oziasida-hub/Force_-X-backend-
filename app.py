from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from ultralytics import YOLO
import numpy as np
import cv2
from datetime import datetime

app = Flask(__name__)
CORS(app, origins=["https://force-x-frontend.onrender.com"]
model = YOLO("yolo11n.pt")


@app.route("/")
def home():
    return jsonify({
        "name": "Snow AI",
        "status": "online"
    })

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_bytes = request.files["image"].read()

    frame = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return jsonify({"error": "Invalid image"}), 400

    results = model(frame)

    detections = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])

            detections.append({
                "object": model.names[class_id],
                "confidence": round(confidence, 4)
            })

    return jsonify({
        "snow_ai": True,
        "timestamp": datetime.now().isoformat(),
        "detections": detections
    })

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
