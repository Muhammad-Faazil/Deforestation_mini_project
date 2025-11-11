import os

class Config:
    # Model settings
    MODEL_PATH = os.getenv('MODEL_PATH', 'deforestation_model.keras')
    IMAGE_SIZE = (128, 128)
    
    # Prediction settings
    CONFIDENCE_THRESHOLD = float(os.getenv('CONFIDENCE_THRESHOLD', '0.7'))
    MAX_FILE_SIZE = int(os.getenv('MAX_FILE_SIZE', '5242880'))  # 5MB
    
    # App settings
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# You can import this in your other files like:
# from config import Config