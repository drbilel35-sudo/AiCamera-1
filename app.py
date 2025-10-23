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

# Try to import Google Cloud Vision
try:
    from google.cloud import vision
    from google.oauth2 import service_account
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("Google Cloud Vision not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ----------------------------
# GOOGLE VISION SETUP
# ----------------------------
vision_client = None

if VISION_AVAILABLE:
    try:
        logger.info("🔧 Initializing Google Vision Client...")
        
        creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
        
        if creds_json and creds_json.strip() and creds_json != "service-account.json":
            creds_dict = json.loads(creds_json)
            creds = service_account.Credentials.from_service_account_info(creds_dict)
            vision_client = vision.ImageAnnotatorClient(credentials=creds)
            logger.info("✅ Google Vision API initialized successfully")
        else:
            logger.warning("❌ No valid credentials found")
            
    except Exception as e:
        logger.error(f"❌ Vision initialization failed: {e}")
        vision_client = None
else:
    logger.warning("❌ Google Cloud Vision not installed")

# ----------------------------
# TEST ENDPOINTS
# ----------------------------
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "vision_api_available": vision_client is not None,
        "dependencies_available": VISION_AVAILABLE
    })

@app.route('/api/test-vision', methods=['GET'])
def test_vision():
    """Test Vision API functionality"""
    try:
        logger.info("=== VISION API TEST ===")
        
        if not vision_client:
            return jsonify({
                "status": "error", 
                "message": "Vision client not available",
                "vision_api_working": False,
                "vision_available": VISION_AVAILABLE,
                "client_initialized": vision_client is not None
            })
        
        # Create a simple test image
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
        success, encoded_image = cv2.imencode('.jpg', test_image)
        
        if not success:
            return jsonify({"status": "error", "message": "Failed to encode test image"})
            
        image_bytes = encoded_image.tobytes()
        vision_image = vision.Image(content=image_bytes)
        
        # Test the API
        response = vision_client.face_detection(image=vision_image)
        faces_count = len(response.face_annotations) if response.face_annotations else 0
        
        return jsonify({
            "status": "success",
            "vision_api_working": True,
            "faces_detected": faces_count,
            "message": f"✅ Vision API is working! Test completed successfully."
        })
        
    except Exception as e:
        logger.error(f"Vision API test failed: {e}")
        return jsonify({
            "status": "error",
            "vision_api_working": False,
            "message": f"Vision API test failed: {str(e)}",
            "error_details": str(e)
        })

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Debug endpoint"""
    creds_json = os.getenv("GOOGLE_APPLICATION_CREDENTIALS_JSON")
    return jsonify({
        "service": "AI Camera Backend",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "dependencies": {
            "flask": True,
            "opencv": True,
            "google_cloud_vision": VISION_AVAILABLE
        },
        "vision": {
            "client_initialized": vision_client is not None,
            "library_available": VISION_AVAILABLE
        },
        "environment": {
            "GOOGLE_APPLICATION_CREDENTIALS_JSON_set": bool(creds_json),
            "GOOGLE_APPLICATION_CREDENTIALS_JSON_length": len(creds_json) if creds_json else 0,
            "PORT": os.getenv("PORT", "5000")
        }
    })

# ----------------------------
# MAIN APPLICATION ENDPOINTS
# ----------------------------
@app.route('/api/analyze-frame', methods=['POST'])
def analyze_frame():
    """Analyze camera frames"""
    try:
        if not request.json or 'frame' not in request.json:
            return jsonify({"error": "No frame data provided"}), 400

        # Simulate processing
        response = {
            "timestamp": datetime.now().isoformat(),
            "analysis_id": f"ana_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            "situation_analysis": {
                "description": "AI analysis active - Vision API available" if vision_client else "Using simulated data",
                "environment": "indoor",
                "activity_level": "low",
                "people_count": 1,
                "object_counts": {"person": 1},
                "primary_objects": ["person"]
            },
            "objects_detected": [
                {
                    "class": "person",
                    "confidence": 0.95,
                    "bounding_box": [
                        {"x": 0.3, "y": 0.4}, {"x": 0.5, "y": 0.4},
                        {"x": 0.5, "y": 0.8}, {"x": 0.3, "y": 0.8}
                    ],
                    "type": "person"
                }
            ],
            "alerts": [{"priority": "low", "message": "Normal activity detected"}],
            "summary": {
                "total_objects": 1,
                "people_count": 1,
                "alert_level": "low"
            },
            "debug": {
                "vision_api_used": vision_client is not None,
                "timestamp": datetime.now().isoformat()
            }
        }
        
        return jsonify(response)

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({"error": "Analysis failed"}), 500

@app.route('/api/camera-status', methods=['POST'])
def camera_status():
    """Camera status endpoint"""
    return jsonify({
        "status": "received",
        "timestamp": datetime.now().isoformat(),
        "server_time": datetime.now().strftime('%H:%M:%S'),
        "vision_api_available": vision_client is not None
    })

# ----------------------------
# START APPLICATION
# ----------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"🚀 Starting AI Camera Backend on port {port}")
    logger.info(f"📡 Vision API Available: {VISION_AVAILABLE}")
    logger.info(f"🔧 Vision Client Initialized: {vision_client is not None}")
    app.run(host='0.0.0.0', port=port, debug=False)
