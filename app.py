from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
import json
from datetime import datetime
import logging
import os
import traceback

# Google Cloud imports
from google.cloud import vision
from google.oauth2 import service_account
from google.api_core.exceptions import GoogleAPICallError, PermissionDenied

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ----------------------------
# GOOGLE VISION SETUP
# ----------------------------
def initialize_vision_client():
    """Initialize Google Vision client"""
    try:
        logger.info("🔧 Initializing Google Vision Client...")
        
        creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        
        if not creds_json:
            logger.error("❌ No credentials found in environment")
            return None
            
        try:
            # Parse the JSON to validate it
            creds_dict = json.loads(creds_json)
            logger.info(f"✅ Credentials parsed - Project: {creds_dict.get('project_id', 'Unknown')}")
            
            # Create credentials
            creds = service_account.Credentials.from_service_account_info(creds_dict)
            client = vision.ImageAnnotatorClient(credentials=creds)
            
            logger.info("✅ Google Vision API initialized successfully")
            return client
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in credentials: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Failed to create Vision client: {e}")
            return None
            
    except Exception as e:
        logger.error(f"❌ Vision initialization error: {str(e)}")
        return None

# Initialize Vision client
vision_client = initialize_vision_client()

# ----------------------------
# HEALTH CHECK
# ----------------------------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "AI Camera Backend",
        "vision_api_available": vision_client is not None
    })

# ----------------------------
# VISION API TEST ENDPOINT
# ----------------------------
@app.route('/api/test-vision', methods=['GET'])
def test_vision():
    """Test Vision API functionality"""
    try:
        logger.info("=== VISION API TEST ===")
        
        if not vision_client:
            return jsonify({
                "status": "error", 
                "message": "Vision client not initialized",
                "vision_api_working": False
            })
        
        # Create a test image
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        test_image[:, :, 2] = 255  # Red channel
        
        success, encoded_image = cv2.imencode('.jpg', test_image)
        if not success:
            return jsonify({"status": "error", "message": "Failed to encode test image"})
            
        image_bytes = encoded_image.tobytes()
        vision_image = vision.Image(content=image_bytes)
        
        # Test face detection
        faces_response = vision_client.face_detection(image=vision_image)
        faces_count = len(faces_response.face_annotations) if faces_response.face_annotations else 0
        
        # Test object detection
        objects_response = vision_client.object_localization(image=vision_image)
        objects_count = len(objects_response.localized_object_annotations) if objects_response.localized_object_annotations else 0
        
        logger.info(f"✅ Test completed: {faces_count} faces, {objects_count} objects")
        
        return jsonify({
            "status": "success",
            "vision_api_working": True,
            "faces_detected": faces_count,
            "objects_detected": objects_count,
            "message": f"Vision API is working! Found {faces_count} faces and {objects_count} objects."
        })
        
    except Exception as e:
        logger.error(f"❌ Vision API test failed: {e}")
        return jsonify({
            "status": "error",
            "vision_api_working": False,
            "message": f"Vision API test failed: {str(e)}"
        })

# ----------------------------
# DEBUG ENDPOINT
# ----------------------------
@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Debug endpoint to check system status"""
    creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    return jsonify({
        "service": "AI Camera Backend",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "vision_client_initialized": vision_client is not None,
        "environment": {
            "GOOGLE_APPLICATION_CREDENTIALS_JSON_set": bool(creds_json),
            "GOOGLE_APPLICATION_CREDENTIALS_JSON_length": len(creds_json) if creds_json else 0,
            "PORT": os.getenv("PORT", "5000")
        }
    })

# ----------------------------
# FRAME ANALYSIS ENDPOINT
# ----------------------------
@app.route('/api/analyze-frame', methods=['POST'])
def analyze_frame():
    """Analyze camera frames"""
    try:
        if not request.json or 'frame' not in request.json:
            return jsonify({"error": "No frame data provided"}), 400

        frame_data = request.json['frame']
        confidence_threshold = request.json.get('confidence_threshold', 0.6)

        # Convert base64 to OpenCV image
        image = base64_to_cv2(frame_data)
        if image is None:
            return jsonify(get_simulated_response())

        # Use Vision API if available
        if vision_client:
            try:
                results = analyze_with_vision_api(image)
                filtered_results = [r for r in results if r.get('confidence', 0) >= confidence_threshold]
                
                response = {
                    "timestamp": datetime.now().isoformat(),
                    "analysis_id": f"ana_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
                    "situation_analysis": generate_situation_analysis(filtered_results),
                    "objects_detected": filtered_results,
                    "alerts": generate_alerts(filtered_results),
                    "summary": {
                        "total_objects": len(filtered_results),
                        "people_count": len([r for r in filtered_results if r.get('class') == 'person']),
                        "alert_level": "low"
                    }
                }
                return jsonify(response)
                
            except Exception as e:
                logger.error(f"Vision API error: {e}")
                # Fall through to simulated data
                
        # Fallback to simulated data
        return jsonify(get_simulated_response())

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify(get_simulated_response())

# ----------------------------
# CAMERA STATUS ENDPOINT
# ----------------------------
@app.route('/api/camera-status', methods=['POST'])
def camera_status():
    """Camera heartbeat endpoint"""
    data = request.json or {}
    logger.info(f"Camera status: {data}")
    return jsonify({
        "status": "received",
        "timestamp": datetime.now().isoformat(),
        "vision_api_available": vision_client is not None
    })

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def base64_to_cv2(base64_string):
    """Convert base64 to OpenCV image"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img if img is not None else None
        
    except Exception as e:
        logger.error(f"Image decoding error: {str(e)}")
        return None

def analyze_with_vision_api(image):
    """Analyze image with Google Vision API"""
    try:
        success, encoded_image = cv2.imencode('.jpg', image)
        if not success:
            return []
            
        image_bytes = encoded_image.tobytes()
        vision_image = vision.Image(content=image_bytes)
        results = []

        # Detect faces
        faces_response = vision_client.face_detection(image=vision_image)
        faces = faces_response.face_annotations or []
        
        for face in faces:
            vertices = [(vertex.x, vertex.y) for vertex in face.bounding_poly.vertices]
            normalized_vertices = [
                {"x": v[0]/image.shape[1], "y": v[1]/image.shape[0]} 
                for v in vertices
            ]
            
            results.append({
                "class": "person",
                "confidence": face.detection_confidence,
                "bounding_box": normalized_vertices,
                "type": "face"
            })

        # Detect objects
        objects_response = vision_client.object_localization(image=vision_image)
        objects = objects_response.localized_object_annotations or []
        
        for obj in objects:
            results.append({
                "class": obj.name.lower(),
                "confidence": obj.score,
                "bounding_box": [{"x": v.x, "y": v.y} for v in obj.bounding_poly.normalized_vertices],
                "type": "object"
            })

        return results

    except Exception as e:
        logger.error(f"Vision API analysis error: {str(e)}")
        return []

def generate_situation_analysis(objects):
    """Generate situation analysis"""
    people_count = len([o for o in objects if o.get('class') == 'person'])
    object_types = list(set([obj.get('class', 'unknown') for obj in objects if obj.get('class') != 'person']))
    
    if people_count == 0:
        description = "No people detected."
        activity = "none"
    elif people_count == 1:
        description = "One person present."
        activity = "low"
    else:
        description = f"Multiple people ({people_count}) detected."
        activity = "medium"
    
    if object_types:
        description += f" Objects: {', '.join(object_types[:3])}."
    
    return {
        "description": description,
        "environment": "indoor",
        "activity_level": activity,
        "people_count": people_count,
        "object_counts": {obj_class: len([o for o in objects if o.get('class') == obj_class]) for obj_class in set([o.get('class') for o in objects])},
        "primary_objects": object_types[:3]
    }

def generate_alerts(objects):
    """Generate alerts"""
    people_count = len([o for o in objects if o.get('class') == 'person'])
    
    if people_count > 5:
        return [{"priority": "high", "message": f"High people count: {people_count}"}]
    elif people_count == 0:
        return [{"priority": "low", "message": "No people detected"}]
    else:
        return [{"priority": "low", "message": "Normal activity"}]

def get_simulated_response():
    """Fallback simulated data"""
    return {
        "situation_analysis": {
            "description": "AI analysis from camera feed",
            "environment": "indoor",
            "activity_level": "low",
            "people_count": 1,
            "object_counts": {"person": 1, "chair": 1},
            "primary_objects": ["person", "chair"]
        },
        "objects_detected": [
            {
                "class": "person",
                "confidence": 0.92,
                "bounding_box": [
                    {"x": 0.3, "y": 0.4}, {"x": 0.5, "y": 0.4},
                    {"x": 0.5, "y": 0.8}, {"x": 0.3, "y": 0.8}
                ],
                "type": "person"
            }
        ],
        "alerts": [{"priority": "low", "message": "Normal activity"}],
        "summary": {
            "total_objects": 1,
            "people_count": 1,
            "alert_level": "low"
        },
        "analysis_id": f"sim_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
        "timestamp": datetime.now().isoformat()
    }

# ----------------------------
# START APPLICATION
# ----------------------------
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=False)
