import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
PRODUCT_IMAGE_FOLDER = os.path.join(UPLOAD_FOLDER, "products")
TEMP_UPLOAD_FOLDER = os.path.join(UPLOAD_FOLDER, "temp")

DB_PATH = os.path.join(BASE_DIR, "data", "products.db")
EMBEDDING_PATH = os.path.join(BASE_DIR, "data", "embeddings.pkl")

# Put your LOCAL SigLIP model folder here
# Example structure:
# models/
#   siglip/
#     config.json
#     model.safetensors
#     preprocessor_config.json
#     tokenizer.json (if needed)
MODEL_DIR = os.path.join(BASE_DIR, "models", "siglip")

os.makedirs(PRODUCT_IMAGE_FOLDER, exist_ok=True)
os.makedirs(TEMP_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "data"), exist_ok=True)