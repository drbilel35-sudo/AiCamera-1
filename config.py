import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Google Cloud Vision
    GOOGLE_APPLICATION_CREDENTIALS = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    GCP_PROJECT_ID = os.getenv('GCP_PROJECT_ID', 'your-project-id')
    
    # Analysis settings
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', 0.6))
    MAX_OBJECTS = int(os.getenv('MAX_OBJECTS', 15))
    
    # Alert settings
    ENABLE_ALERTS = os.getenv('ENABLE_ALERTS', 'true').lower() == 'true'
    ALERT_COOLDOWN = int(os.getenv('ALERT_COOLDOWN', 30))  # seconds
    
    # Performance settings
    MAX_IMAGE_SIZE = (1920, 1080)  # Resize large images for faster processing
    REQUEST_TIMEOUT = 30

    # Security
    API_KEYS = os.getenv('API_KEYS', '').split(',')
