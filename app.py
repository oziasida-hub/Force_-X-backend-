from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from datetime import datetime
from base64 import b64decode
import numpy as np
import cv2
import urllib.parse
import urllib.request
import json
import os

app = Flask(__name__)

CORS(
    app,
    origins=[
        "https://force-x.onrender.com",
        "https://force-x-frontend.onrender.com"
    ]
)

model = YOLO("yolo11n.pt")

VISION_API_KEY = os.environ.get("VISION_API_KEY")

VISION_URL = (
    "https://vision.googleapis.com/v1/images:annotate"
)

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


def verify_image_online(image_bytes):

    if not VISION_API_KEY:
        return {
            "available": False,
            "message": "Online verification is not configured."
        }

    try:
        encoded_image = (
            __import__("base64")
            .b64encode(image_bytes)
            .decode("utf-8")
        )

        payload = {
            "requests": [
                {
                    "image": {
                        "content": encoded_image
                    },
                    "features": [
                        {
                            "type": "WEB_DETECTION",
                            "maxResults": 10
                        }
                    ]
                }
            ]
        }

        url = (
            VISION_URL
            + "?key="
            + urllib.parse.quote(VISION_API_KEY)
        )

        request_data = json.dumps(
            payload
        ).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=request_data,
            headers={
                "Content-Type": "application/json"
            },
            method="POST"
        )

        with urllib.request.urlopen(
            req,
            timeout=20
        ) as response:

            result = json.loads(
                response.read().decode("utf-8")
            )

        response_data = result.get(
            "responses",
            [{}]
        )[0]

        web = response_data.get(
            "webDetection",
            {}
        )

        best_guess = []

        for item in web.get(
            "bestGuessLabels",
            []
        ):

            label = item.get("label")

            if label:
                best_guess.append(label)

        entities = []

        for entity in web.get(
            "webEntities",
            []
        ):

            description = entity.get(
                "description"
            )

            score = entity.get(
                "score"
            )

            if description:

                entities.append({
                    "name": description,
                    "score": score
                })

        matching_pages = []

        for page in web.get(
            "pagesWithMatchingImages",
            []
        ):

            page_url = page.get("url")
            page_title = page.get("pageTitle")

            if page_url:

                matching_pages.append({
                    "title":
                        page_title or "Matching web page",
                    "url":
                        page_url
                })

        full_matches = []

        for image in web.get(
            "fullMatchingImages",
            []
        ):

            image_url = image.get("url")

            if image_url:
                full_matches.append(image_url)

        partial_matches = []

        for image in web.get(
            "partialMatchingImages",
            []
        ):

            image_url = image.get("url")

            if image_url:
                partial_matches.append(image_url)

        visually_similar = []

        for image in web.get(
            "visuallySimilarImages",
            []
        ):

            image_url = image.get("url")

            if image_url:
                visually_similar.append(image_url)

        return {
            "available": True,
            "best_guess": best_guess,
            "web_entities": entities[:10],
            "matching_pages": matching_pages[:10],
            "full_matches": full_matches[:5],
            "partial_matches": partial_matches[:5],
            "visually_similar": visually_similar[:5]
        }

    except Exception as error:

        print(
            "Online verification error:",
            error
        )

        return {
            "available": False,
            "message": "Online verification failed."
        }


def determine_precise_identity(
    object_name,
    web_result
):

    best_guess = web_result.get(
        "best_guess",
        []
    )

    entities = web_result.get(
        "web_entities",
        []
    )

    online_names = []

    for item in best_guess:

        online_names.append(
            item.lower()
        )

    for entity in entities:

        name = entity.get(
            "name",
            ""
        )

        if name:

            online_names.append(
                name.lower()
            )

    known_species = [
        {
            "common": "Bonobo",
            "scientific": "Pan paniscus",
            "type": "Great ape",
            "keywords": [
                "bonobo",
                "pan paniscus"
            ]
        },
        {
            "common": "Okapi",
            "scientific": "Okapia johnstoni",
            "type": "Giraffid",
            "keywords": [
                "okapi",
                "okapia johnstoni"
            ]
        },
        {
            "common": "Likweli",
            "scientific": "Colobus congoensis",
            "type": "Colobus monkey",
            "keywords": [
                "likweli",
                "colobus congoensis"
            ]
        }
    ]

    for species in known_species:

        for online_name in online_names:

            for keyword in species["keywords"]:

                if keyword in online_name:

                    return {
                        "identified": True,
                        "type": species["type"],
                        "species": species["common"],
                        "scientific_name":
                            species["scientific"],
                        "source":
                            "Online web verification"
                    }

    if best_guess:

        return {
            "identified": True,
            "type": object_name.title(),
            "species": best_guess[0],
            "scientific_name": None,
            "source":
                "Online web verification"
        }

    return {
        "identified": False,
        "type": object_name.title(),
        "species": "Unknown",
        "scientific_name": None,
        "source":
            "Snow AI object detection only"
    }


@app.route("/")
def home():

    return jsonify({
        "name": "Snow AI",
        "status": "online",
        "gps": "supported",
        "online_verification":
            bool(VISION_API_KEY)
    })


@app.route(
    "/detect",
    methods=["POST"]
)
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

        latitude = data.get(
            "latitude"
        )

        longitude = data.get(
            "longitude"
        )

        accuracy = data.get(
            "accuracy",
            data.get("gps_accuracy")
        )

        try:

            if latitude is not None:
                latitude = float(latitude)

            if longitude is not None:
                longitude = float(longitude)

            if accuracy is not None:
                accuracy = float(accuracy)

        except (
            ValueError,
            TypeError
        ):

            return jsonify({
                "success": False,
                "error": "Invalid GPS coordinates"
            }), 400

        if latitude is not None:

            if latitude < -90 or latitude > 90:

                return jsonify({
                    "success": False,
                    "error":
                        "Latitude must be between -90 and 90"
                }), 400

        if longitude is not None:

            if longitude < -180 or longitude > 180:

                return jsonify({
                    "success": False,
                    "error":
                        "Longitude must be between -180 and 180"
                }), 400

        if "," in image_data:

            image_data = image_data.split(
                ",",
                1
            )[1]

        try:

            image_bytes = b64decode(
                image_data
            )

        except Exception:

            return jsonify({
                "success": False,
                "error": "Invalid Base64 image"
            }), 400

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

        online_result = verify_image_online(
            image_bytes
        )

        for result in results:

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                object_name = model.names[
                    class_id
                ].lower()

                description = object_info.get(
                    object_name,
                    "Snow AI detected this object, "
                    "but a built-in description "
                    "is not available yet."
                )

                precise = determine_precise_identity(
                    object_name,
                    online_result
                )

                google_url = (
                    "https://www.google.com/search?q="
                    + urllib.parse.quote(
                        (
                            precise.get("species")
                            or object_name
                        )
                        + " information"
                    )
                )

                objects.append({

                    "name":
                        object_name.title(),

                    "confidence":
                        f"{confidence:.1%}",

                    "type":
                        precise.get("type"),

                    "species":
                        precise.get("species"),

                    "scientific_name":
                        precise.get(
                            "scientific_name"
                        ),

                    "identification_source":
                        precise.get(
                            "source"
                        ),

                    "description":
                        description,

                    "google_url":
                        google_url
                })

        return jsonify({

            "success":
                True,

            "snow_ai":
                True,

            "online_verification":
                online_result,

            "timestamp":
                timestamp,

            "duration":
                duration,

            "location": {

                "latitude":
                    latitude,

                "longitude":
                    longitude,

                "accuracy":
                    accuracy
            },

            "objects":
                objects
        })

    except Exception as error:

        print(
            "Detection error:",
            error
        )

        return jsonify({

            "success":
                False,

            "error":
                str(error)

        }), 500


if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(
            os.environ.get(
                "PORT",
                5000
            )
        )
)
