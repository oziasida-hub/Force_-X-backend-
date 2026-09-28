from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
from datetime import datetime
from base64 import b64decode
from google import genai
from google.genai import types

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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

gemini_client = None

if GEMINI_API_KEY:
    gemini_client = genai.Client(
        api_key=GEMINI_API_KEY
    )


KNOWLEDGE_FILE = "knowledge_base.json"


def load_knowledge():

    if not os.path.exists(KNOWLEDGE_FILE):
        return {}

    try:
        with open(
            KNOWLEDGE_FILE,
            "r",
            encoding="utf-8"
        ) as file:
            return json.load(file)

    except Exception as error:

        print(
            "Knowledge database load error:",
            error
        )

        return {}


def save_knowledge(database):

    try:

        with open(
            KNOWLEDGE_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                database,
                file,
                indent=4,
                ensure_ascii=False
            )

    except Exception as error:

        print(
            "Knowledge database save error:",
            error
        )


knowledge_base = load_knowledge()


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


def analyze_image_with_gemini(
    image_bytes,
    object_name,
    yolo_confidence
):

    if not gemini_client:

        return {
            "available": False,
            "identified": False,
            "message": "Gemini is not configured."
        }

    prompt = f"""
You are the visual verification system for Snow AI.

YOLO detected this object as:
{object_name}

YOLO confidence:
{yolo_confidence:.1%}

Analyze the supplied image carefully.

Determine:
1. What object or animal is actually visible.
2. The most specific identification that can reasonably be made from the image.
3. For animals, give the likely species if the visual evidence supports it.
4. For domestic animals, give a breed only if the image provides enough evidence.
5. Give the scientific name only when reasonably confident.
6. Do not invent an identification.
7. If the image is insufficient for precise identification, say so.

Return ONLY valid JSON in this exact structure:

{{
  "identified": true,
  "name": "specific name",
  "type": "general type",
  "species": "species or null",
  "scientific_name": "scientific name or null",
  "breed": "breed or null",
  "confidence": 0.0,
  "reason": "short explanation"
}}

The confidence value must be between 0 and 1.
"""

    try:

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type="image/jpeg"
                ),
                prompt
            ],
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=500
            )
        )

        text = response.text.strip()

        if text.startswith("```"):
            text = text.replace(
                "```json",
                ""
            ).replace(
                "```",
                ""
            ).strip()

        result = json.loads(text)

        if not isinstance(result, dict):
            raise ValueError(
                "Gemini returned invalid data."
            )

        return {
            "available": True,
            **result
        }

    except Exception as error:

        print(
            "Gemini analysis error:",
            error
        )

        return {
            "available": False,
            "identified": False,
            "message": "Gemini image analysis failed."
        }


def search_online(query):

    try:

        search_query = urllib.parse.quote(
            query
        )

        search_url = (
            "https://en.wikipedia.org/w/api.php"
            "?action=query"
            "&list=search"
            "&srsearch="
            + search_query
            + "&format=json"
            "&utf8=1"
            "&srlimit=5"
        )

        req = urllib.request.Request(
            search_url,
            headers={
                "User-Agent": "SnowAI/1.0"
            }
        )

        with urllib.request.urlopen(
            req,
            timeout=10
        ) as response:

            search_data = json.loads(
                response.read().decode(
                    "utf-8"
                )
            )

        search_results = (
            search_data
            .get("query", {})
            .get("search", [])
        )

        results = []

        for item in search_results:

            title = item.get(
                "title"
            )

            if title:

                results.append({
                    "title": title,
                    "url":
                        "https://en.wikipedia.org/wiki/"
                        + urllib.parse.quote(
                            title.replace(
                                " ",
                                "_"
                            )
                        )
                })

        return {
            "available": True,
            "found": bool(results),
            "results": results
        }

    except Exception as error:

        print(
            "Online search error:",
            error
        )

        return {
            "available": False,
            "found": False,
            "results": []
        }


def get_online_summary(title):

    try:

        encoded_title = urllib.parse.quote(
            title.replace(
                " ",
                "_"
            )
        )

        url = (
            "https://en.wikipedia.org/api/rest_v1/page/summary/"
            + encoded_title
        )

        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "SnowAI/1.0"
            }
        )

        with urllib.request.urlopen(
            req,
            timeout=10
        ) as response:

            data = json.loads(
                response.read().decode(
                    "utf-8"
                )
            )

        extract = data.get(
            "extract"
        )

        if not extract:
            return None

        return {
            "title":
                data.get(
                    "title",
                    title
                ),
            "description":
                data.get(
                    "description"
                ),
            "summary":
                extract,
            "url":
                data.get(
                    "content_urls",
                    {}
                )
                .get(
                    "desktop",
                    {}
                )
                .get(
                    "page"
                )
        }

    except Exception as error:

        print(
            "Online summary error:",
            error
        )

        return None


def research_object(
    object_name,
    precise_name=None
):

    query = (
        precise_name
        if precise_name
        else object_name
    )

    key = query.lower().strip()

    if key in knowledge_base:

        return {
            "available": True,
            "learned_before": True,
            "source":
                "Snow AI knowledge base",
            "knowledge":
                knowledge_base[key]
        }

    search = search_online(
        query
    )

    if not search.get(
        "found"
    ):

        return {
            "available":
                search.get(
                    "available",
                    False
                ),
            "learned_before":
                False,
            "source":
                "Online research",
            "knowledge":
                None
        }

    summary = get_online_summary(
        search["results"][0]["title"]
    )

    if not summary:

        return {
            "available": True,
            "learned_before": False,
            "source":
                "Online research",
            "knowledge":
                None
        }

    knowledge = {
        "object":
            object_name.title(),
        "online_title":
            summary.get(
                "title"
            ),
        "description":
            summary.get(
                "description"
            ),
        "summary":
            summary.get(
                "summary"
            ),
        "source":
            "Wikipedia",
        "source_url":
            summary.get(
                "url"
            ),
        "learned_at":
            datetime.now().isoformat()
    }

    knowledge_base[key] = knowledge

    save_knowledge(
        knowledge_base
    )

    return {
        "available": True,
        "learned_before": False,
        "source":
            "Online research",
        "knowledge":
            knowledge
    }


def determine_precise_identity(
    object_name,
    gemini_result
):

    if not gemini_result.get(
        "available"
    ):

        return {
            "identified": False,
            "type":
                object_name.title(),
            "species": "Unknown",
            "scientific_name": None,
            "breed": None,
            "source":
                "Snow AI object detection"
        }

    confidence = float(
        gemini_result.get(
            "confidence",
            0
        )
    )

    if not gemini_result.get(
        "identified"
    ) or confidence < 0.70:

        return {
            "identified": False,
            "type":
                gemini_result.get(
                    "type",
                    object_name.title()
                ),
            "species": "Unknown",
            "scientific_name": None,
            "breed": None,
            "source":
                "Insufficient visual evidence"
        }

    return {
        "identified": True,
        "type":
            gemini_result.get(
                "type",
                object_name.title()
            ),
        "species":
            gemini_result.get(
                "species"
            ),
        "scientific_name":
            gemini_result.get(
                "scientific_name"
            ),
        "breed":
            gemini_result.get(
                "breed"
            ),
        "source":
            "YOLO + Gemini visual verification"
    }


@app.route("/")
def home():

    return jsonify({
        "name":
            "Snow AI",
        "status":
            "online",
        "gps":
            "supported",
        "online_research":
            True,
        "knowledge_base":
            True,
        "gemini":
            bool(gemini_client)
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
                "error":
                    "No JSON data received"
            }), 400

        if "image" not in data:

            return jsonify({
                "success": False,
                "error":
                    "No image provided"
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
            data.get(
                "gps_accuracy"
            )
        )

        try:

            if latitude is not None:
                latitude = float(
                    latitude
                )

            if longitude is not None:
                longitude = float(
                    longitude
                )

            if accuracy is not None:
                accuracy = float(
                    accuracy
                )

        except (
            ValueError,
            TypeError
        ):

            return jsonify({
                "success": False,
                "error":
                    "Invalid GPS coordinates"
            }), 400

        if (
            latitude is not None
            and not -90 <= latitude <= 90
        ):

            return jsonify({
                "success": False,
                "error":
                    "Latitude must be between -90 and 90"
            }), 400

        if (
            longitude is not None
            and not -180 <= longitude <= 180
        ):

            return jsonify({
                "success": False,
                "error":
                    "Longitude must be between -180 and 180"
            }), 400

        if "," in image_data:

            image_data = (
                image_data.split(
                    ",",
                    1
                )[1]
            )

        try:

            image_bytes = b64decode(
                image_data
            )

        except Exception:

            return jsonify({
                "success": False,
                "error":
                    "Invalid Base64 image"
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
                "error":
                    "Invalid image"
            }), 400

        results = model(
            frame
        )

        objects = []

        for result in results:

            for box in result.boxes:

                class_id = int(
                    box.cls[0]
                )

                confidence = float(
                    box.conf[0]
                )

                object_name = (
                    model.names[
                        class_id
                    ]
                    .lower()
                )

                description = object_info.get(
                    object_name,
                    "Snow AI detected this object, but a built-in description is not available yet."
                )

                gemini_result = (
                    analyze_image_with_gemini(
                        image_bytes,
                        object_name,
                        confidence
                    )
                )

                precise = (
                    determine_precise_identity(
                        object_name,
                        gemini_result
                    )
                )

                precise_name = (
                    precise.get(
                        "species"
                    )
                    or precise.get(
                        "breed"
                    )
                    or precise.get(
                        "type"
                    )
                    or object_name
                )

                research = research_object(
                    object_name,
                    precise_name
                )

                google_url = (
                    "https://www.google.com/search?q="
                    + urllib.parse.quote(
                        precise_name
                        + " information"
                    )
                )

                online_knowledge = (
                    research.get(
                        "knowledge"
                    )
                )

                objects.append({

                    "name":
                        object_name.title(),

                    "confidence":
                        f"{confidence:.1%}",

                    "type":
                        precise.get(
                            "type"
                        ),

                    "species":
                        precise.get(
                            "species"
                        ),

                    "scientific_name":
                        precise.get(
                            "scientific_name"
                        ),

                    "breed":
                        precise.get(
                            "breed"
                        ),

                    "identification_source":
                        precise.get(
                            "source"
                        ),

                    "gemini":
                        gemini_result,

                    "description":
                        description,

                    "online_research":
                        research.get(
                            "available",
                            False
                        ),

                    "learned_before":
                        research.get(
                            "learned_before",
                            False
                        ),

                    "online_knowledge":
                        online_knowledge,

                    "google_url":
                        google_url
                })

        return jsonify({

            "success":
                True,

            "snow_ai":
                True,

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
