import os
import numpy as np
import torch
from PIL import Image
from transformers import SiglipModel, SiglipProcessor

from config import MODEL_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SigLIPModelWrapper:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model_dir = model_dir
        self.model = None
        self.processor = None
        self._warmed_up = False

    def load(self):
        if self.model is not None and self.processor is not None:
            return

        if not os.path.isdir(self.model_dir):
            raise FileNotFoundError(
                f"Local SigLIP model directory not found: {self.model_dir}"
            )

        self.processor = SiglipProcessor.from_pretrained(
            self.model_dir,
            local_files_only=True,
        )
        self.model = SiglipModel.from_pretrained(
            self.model_dir,
            local_files_only=True,
        ).to(device)

        self.model.eval()

    @torch.inference_mode()
    def warmup(self):
        self.load()
        if self._warmed_up:
            return

        dummy = Image.new("RGB", (224, 224), color="white")
        inputs = self.processor(images=dummy, return_tensors="pt").to(device)
        _ = self.model.get_image_features(**inputs)
        self._warmed_up = True

    @torch.inference_mode()
    def get_embedding(self, image_path: str):
        self.load()

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(device)

        features = self.model.get_image_features(**inputs)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)

        return features.cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def get_embedding_from_image(self, image):
        self.load()

        if image.mode != "RGB":
            image = image.convert("RGB")

        inputs = self.processor(images=image, return_tensors="pt").to(device)
        features = self.model.get_image_features(**inputs)
        features = torch.nn.functional.normalize(features, p=2, dim=-1)

        return features.cpu().numpy().astype(np.float32)