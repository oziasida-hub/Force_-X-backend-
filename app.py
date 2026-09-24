from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from datetime import datetime
from base64 import b64decode
import numpy as np
import cv2
import urllib.parse

app = Flask(__name__)

CORS(
    app,
    origins=[
        "https://force-x.onrender.com",
        "https://force-x-frontend.onrender.com"
    ]
)

model = YOLO("yolo11n.pt")

object_info = {
    "person": "A human being.",
    "dog": "A domesticated mammal commonly kept as a companion animal.",
    "cat": "A domesticated mammal commonly kept as a companion animal.",
    "bird": "A warm-blooded animal with feathers, wings and a beak.",
    "horse": "A large domesticated mammal commonly used for riding and work.",
    "cow": "A domesticated mammal commonly raised for milk and meat.",
    "sheep": "A domesticated mammal commonly raised for wool and meat.",
    "elephant": "A very large land mammal known for its trunk and tusks.",
    "bear": "A large mammal belonging to the bear family.",
    "zebra": "A wild African mammal known for its black-and-white stripes.",
    "giraffe": "A tall African mammal known for its long neck and legs.",
    "car": "A motor vehicle mainly designed to transport people.",
    "bicycle": "A human-powered vehicle with two wheels.",
    "motorcycle": "A two-wheeled motor vehicle.",
    "bus": "A large road vehicle designed to transport passengers.",
    "truck": "A motor vehicle designed mainly for transporting goods.",
    "laptop": "A portable personal computer.",
    "cell phone": "A portable electronic device used for communication and digital tasks.",
    "bottle": "A container commonly used to hold liquids.",
    "chair": "A piece of furniture designed for one person to sit on.",
    "backpack": "A bag designed to be carried on a person's back."
}

@app.route("/")
def home():
    return jsonify({
        "name": "Snow AI",
        "status": "online"
    })

@app.route("/detect", methods=["POST"])
def detect():

    try:

        data = request.get_json()

        if not data:
            return jsonify({
                "success": False,
                "error": "No JSON data received"
            }), 400

        if "image" not in data:
            return jsonify({
                "success": False,
                "error": "No image provided"
            }), 400

        image_data = data["image"]

        timestamp = data.get(
            "timestamp",
            datetime.now().isoformat()
        )

        duration = data.get(
            "duration",
            "0"
        )

        image_bytes = b64decode(
            image_data.split(",", 1)[1]
        )

        frame = cv2.imdecode(
            np.frombuffer(
                image_bytes,
                np.uint8
            ),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return jsonify({
                "success": False,
                "error": "Invalid image"
            }), 400

        results = model(frame)

        objects = []

        for result in results:

            for box in result.boxes:

                class_id = int(box.cls[0])

                confidence = float(
                    box.conf[0]
                )

                object_name = model.names[
                    class_id
                ].lower()

                description = object_info.get(
                    object_name,
                    "Snow AI detected this object, but a built-in description is not available yet."
                )

                google_url = (
                    "https://www.google.com/search?q="
                    + urllib.parse.quote(
                        object_name + " information"
                    )
                )

                objects.append({

                    "name":
                        object_name.title(),

                    "confidence":
                        f"{confidence:.1%}",

                    "description":
                        description,

                    "google_url":
                        google_url
                })

        return jsonify({

            "success": True,

            "snow_ai": True,

            "timestamp":
                timestamp,

            "duration":
                duration,

            "objects":
                objects
        })

    except Exception as error:

        return jsonify({

            "success": False,

            "error":
                str(error)

        }), 500


if __name__ == "__main__":

    import os

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        )
)
