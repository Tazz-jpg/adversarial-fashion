"""
Configuration settings for the Adversarial Fashion project.
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')

# Ensure directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Model configuration
YOLOV5_MODEL = 'yolov5s'  # Options: yolov5n, yolov5s, yolov5m, yolov5l, yolov5x
CONFIDENCE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.45

# Patch generation parameters
PATCH_SIZE = 224
NUM_ITERATIONS = 300  # Reduced for faster testing
LEARNING_RATE = 0.01
EOT_SAMPLES = 4

# Transformation ranges for EoT
ROTATION_RANGE = 15
SCALE_RANGE = (0.8, 1.2)
BRIGHTNESS_RANGE = (0.7, 1.3)
CONTRAST_RANGE = (0.8, 1.2)

# Flask settings
SECRET_KEY = 'adversarial-fashion-secret-key'
DEBUG = True
HOST = '0.0.0.0'
PORT = 5001
MAX_CONTENT_LENGTH = 16 * 1024 * 1024