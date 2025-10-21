import logging
from google.cloud import vision
import cv2
import base64
from config import Config

logger = logging.getLogger(__name__)

class VisionService:
    def __init__(self):
        try:
            self.client = vision.ImageAnnotatorClient()
            logger.info("Google Cloud Vision client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Vision client: {str(e)}")
            raise

    def analyze_image(self, image):
        """Analyze image using Google Cloud Vision API"""
        try:
            # Convert OpenCV image to bytes
            success, encoded_image = cv2.imencode('.jpg', image)
            if not success:
                raise ValueError("Could not encode image")
            
            image_bytes = encoded_image.tobytes()
            
            # Prepare vision image
            vision_image = vision.Image(content=image_bytes)
            
            # Perform object detection
            response = self.client.object_localization(image=vision_image)
            
            if response.error.message:
                raise Exception(f"Vision API error: {response.error.message}")
            
            return self._process_vision_response(response)
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {str(e)}")
            raise

    def _process_vision_response(self, response):
        """Process Google Cloud Vision API response"""
        objects = []
        
        for obj in response.localized_object_annotations:
            if obj.score >= Config.CONFIDENCE_THRESHOLD:
                # Get bounding box coordinates
                vertices = [(vertex.x, vertex.y) for vertex in obj.bounding_poly.normalized_vertices]
                
                object_data = {
                    'class': obj.name.lower(),
                    'confidence': round(obj.score, 3),
                    'bounding_box': vertices,
                    'midpoint': self._calculate_midpoint(vertices)
                }
                objects.append(object_data)
                
                # Limit number of objects
                if len(objects) >= Config.MAX_OBJECTS:
                    break
        
        # Sort by confidence
        objects.sort(key=lambda x: x['confidence'], reverse=True)
        
        return objects

    def _calculate_midpoint(self, vertices):
        """Calculate midpoint of bounding box"""
        if not vertices:
            return (0.5, 0.5)
        
        x_coords = [v[0] for v in vertices]
        y_coords = [v[1] for v in vertices]
        
        mid_x = sum(x_coords) / len(x_coords)
        mid_y = sum(y_coords) / len(y_coords)
        
        return (mid_x, mid_y)

    def safe_search_detection(self, image):
        """Detect inappropriate content"""
        try:
            success, encoded_image = cv2.imencode('.jpg', image)
            image_bytes = encoded_image.tobytes()
            vision_image = vision.Image(content=image_bytes)
            
            response = self.client.safe_search_detection(image=vision_image)
            safe = response.safe_search_annotation
            
            return {
                'adult': safe.adult.name,
                'violence': safe.violence.name,
                'racy': safe.racy.name,
                'medical': safe.medical.name,
                'spoof': safe.spoof.name
            }
        except Exception as e:
            logger.error(f"Safe search failed: {str(e)}")
            return {}
