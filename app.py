from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
import json
from datetime import datetime
import logging
import os

# Google Cloud imports
from google.cloud import vision
from google.oauth2 import service_account

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ----------------------------
# GOOGLE VISION SETUP
# ----------------------------
try:
    creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    if creds_json:
        creds_dict = json.loads(creds_json)
        creds = service_account.Credentials.from_service_account_info(creds_dict)
        vision_client = vision.ImageAnnotatorClient(credentials=creds)
        logger.info("Google Vision API initialized from environment variable.")
    else:
        vision_client = vision.ImageAnnotatorClient()
        logger.warning("Using default Google Vision credentials (local mode).")
except Exception as e:
    logger.error(f"Error initializing Google Vision: {str(e)}")
    vision_client = None


# ----------------------------
# HEALTH CHECK
# ----------------------------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "AI Camera Backend"
    })


# ----------------------------
# FRAME ANALYSIS ENDPOINT
# ----------------------------
@app.route('/api/analyze-frame', methods=['POST'])
def analyze_frame():
    """Main endpoint for analyzing camera frames (base64 encoded images)"""
    try:
        if not request.json or 'frame' not in request.json:
            return jsonify({"error": "No frame data provided"}), 400

        frame_data = request.json['frame']

        # Convert base64 to OpenCV image
        image = base64_to_cv2(frame_data)
        if image is None:
            return jsonify({"error": "Invalid image data"}), 400

        logger.info("Frame received for analysis")

        # Analyze with Google Cloud Vision
        if not vision_client:
            return jsonify({"error": "Vision API not initialized"}), 500

        results = analyze_with_vision_api(image)

        response = {
            "timestamp": datetime.now().isoformat(),
            "analysis_id": f"ana_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            "objects_detected": results,
            "summary": {
                "total_objects": len(results),
                "people_count": len([r for r in results if r['class'] == 'person']),
                "alert_level": "high" if any(r['confidence'] > 0.9 for r in results) else "low"
            }
        }

        logger.info(f"Analysis complete: {response['summary']}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({"error": "Analysis failed", "details": str(e)}), 500


# ----------------------------
# CAMERA STATUS ENDPOINT
# ----------------------------
@app.route('/api/camera-status', methods=['POST'])
def camera_status():
    """Receives camera heartbeat/status"""
    data = request.json
    logger.info(f"Camera status update: {data}")
    return jsonify({
        "status": "received",
        "timestamp": datetime.now().isoformat(),
        "server_time": datetime.now().strftime('%H:%M:%S')
    })


# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def base64_to_cv2(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Image decoding error: {str(e)}")
        return None


def analyze_with_vision_api(image):
    """Use Google Cloud Vision API to detect faces and objects"""
    try:
        # Encode OpenCV image to bytes
        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            raise ValueError("Failed to encode image")

        image_bytes = encoded_image.tobytes()
        vision_image = vision.Image(content=image_bytes)

        results = []

        # Detect faces
        faces = vision_client.face_detection(image=vision_image).face_annotations
        for i, face in enumerate(faces):
            results.append({
                "class": "person",
                "confidence": face.detection_confidence,
                "emotion": get_emotion(face)
            })

        # Detect objects
        objects = vision_client.object_localization(image=vision_image).localized_object_annotations
        for obj in objects:
            results.append({
                "class": obj.name.lower(),
                "confidence": obj.score
            })

        return results

    except Exception as e:
        logger.error(f"Vision API analysis error: {str(e)}")
        return []


def get_emotion(face):
    """Simple emotion inference from Vision API face annotations"""
    emotions = {
        "joy": face.joy_likelihood,
        "sorrow": face.sorrow_likelihood,
        "anger": face.anger_likelihood,
        "surprise": face.surprise_likelihood
    }
    # Find most likely emotion
    emotion = max(emotions, key=emotions.get)
    return emotion


# ----------------------------
# ENTRY POINT
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
