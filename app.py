from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import cv2
import numpy as np
import json
from datetime import datetime
import logging

# Import our services
from services.vision_service import VisionService
from services.situation_analyzer import SituationAnalyzer
from services.alert_service import AlertService

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize services
vision_service = VisionService()
situation_analyzer = SituationAnalyzer()
alert_service = AlertService()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "AI Camera Backend"
    })

@app.route('/api/analyze-frame', methods=['POST'])
def analyze_frame():
    """
    Main endpoint for analyzing camera frames
    Expects JSON with base64 encoded image
    """
    try:
        # Validate request
        if not request.json or 'frame' not in request.json:
            return jsonify({"error": "No frame data provided"}), 400

        # Get frame data
        frame_data = request.json['frame']
        
        # Optional: Get analysis preferences
        confidence_threshold = request.json.get('confidence_threshold', 0.5)
        max_objects = request.json.get('max_objects', 10)
        
        logger.info("Received frame for analysis")

        # Convert base64 to image
        image = base64_to_cv2(frame_data)
        if image is None:
            return jsonify({"error": "Invalid image data"}), 400

        # Step 1: Object detection with Google Cloud Vision
        vision_results = vision_service.analyze_image(image)
        
        # Step 2: Analyze the overall situation
        situation_analysis = situation_analyzer.analyze(vision_results)
        
        # Step 3: Check for alerts
        alerts = alert_service.check_alerts(vision_results, situation_analysis)

        # Prepare response
        response = {
            "timestamp": datetime.now().isoformat(),
            "analysis_id": f"ana_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}",
            "objects_detected": vision_results,
            "situation_analysis": situation_analysis,
            "alerts": alerts,
            "summary": {
                "total_objects": len(vision_results),
                "people_count": len([obj for obj in vision_results if obj['class'] == 'person']),
                "alert_level": "high" if any(alert.get('priority', 0) > 7 for alert in alerts) else "medium" if alerts else "low"
            }
        }

        logger.info(f"Analysis completed: {response['summary']}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Analysis error: {str(e)}")
        return jsonify({"error": "Analysis failed", "details": str(e)}), 500

@app.route('/api/batch-analyze', methods=['POST'])
def batch_analyze():
    """
    Analyze multiple frames at once
    """
    try:
        frames = request.json.get('frames', [])
        results = []
        
        for i, frame_data in enumerate(frames):
            try:
                image = base64_to_cv2(frame_data)
                vision_results = vision_service.analyze_image(image)
                situation_analysis = situation_analyzer.analyze(vision_results)
                alerts = alert_service.check_alerts(vision_results, situation_analysis)
                
                results.append({
                    "frame_index": i,
                    "objects_detected": vision_results,
                    "situation_analysis": situation_analysis,
                    "alerts": alerts
                })
            except Exception as e:
                results.append({
                    "frame_index": i,
                    "error": str(e)
                })
        
        return jsonify({"batch_results": results})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/camera-status', methods=['POST'])
def camera_status():
    """
    Endpoint for camera heartbeat and status updates
    """
    data = request.json
    logger.info(f"Camera status: {data}")
    
    return jsonify({
        "status": "received",
        "timestamp": datetime.now().isoformat(),
        "server_time": datetime.now().strftime('%H:%M:%S')
    })

def base64_to_cv2(base64_string):
    """Convert base64 string to OpenCV image"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        # Decode base64
        img_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        logger.error(f"Image decoding error: {str(e)}")
        return None

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
