import cv2
import base64
import numpy as np

def resize_image(image, max_size=(1920, 1080)):
    """Resize image to maximum dimensions while maintaining aspect ratio"""
    height, width = image.shape[:2]
    
    if width > max_size[0] or height > max_size[1]:
        scale = min(max_size[0] / width, max_size[1] / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height))
    
    return image

def enhance_image(image):
    """Apply basic image enhancement"""
    # Convert to LAB color space to enhance contrast in L channel
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0).apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    return enhanced

def cv2_to_base64(image):
    """Convert OpenCV image to base64 string"""
    success, encoded_image = cv2.imencode('.jpg', image)
    if success:
        return base64.b64encode(encoded_image).decode('utf-8')
    return None
