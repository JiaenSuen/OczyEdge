from transformers import SiglipModel, SiglipProcessor

MODEL_ID = "google/siglip-base-patch16-224"
SAVE_DIR = "./models/siglip"

print("Downloading processor...")
processor = SiglipProcessor.from_pretrained(MODEL_ID)

print("Downloading model...")
model = SiglipModel.from_pretrained(MODEL_ID)

print("Saving locally...")
processor.save_pretrained(SAVE_DIR)
model.save_pretrained(SAVE_DIR)

print("Done.")