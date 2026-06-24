import os
from typing import Any, Dict, List, Optional

from PIL import Image


DEFAULT_ZERO_SHOT_CLASSES = [
    "product",
    "retail product",
    "packaged product",
    "package",
    "box",
    "bottle",
    "drink bottle",
    "can",
    "carton",
    "container",
    "snack package",
    "food package",
    "merchandise",
]


def _normalize_classes(labels: List[str]) -> List[str]:
    normalized = []
    seen = set()

    for item in labels:
        label = str(item or "").strip()

        if not label:
            continue

        key = label.lower()

        if key in seen:
            continue

        normalized.append(label)
        seen.add(key)

    return normalized


class ZeroShotDetector:
    """
    YOLO-World zero-shot local detector.

    This replaces ZeroShot server mode.

    Purpose:
    - Detect product-like objects locally.
    - No extra API server.
    - No training data required.
    - Return normalized YOLO-style boxes:
      [x_center, y_center, width, height]

    Output:
    [
        {
            "det_id": 1,
            "bbox": [x_center, y_center, width, height],
            "detector_label": "bottle",
            "detector_score": 0.82
        }
    ]
    """

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        api_url: Optional[str] = None,
        model_name: Optional[str] = None,
        conf: float = 0.18,
        iou: float = 0.60,
        imgsz: int = 640,
        max_detections: int = 30,
        device: Optional[str] = None,
    ):
        # api_url / model_name are accepted for compatibility with older app.py versions.
        self.model_name_or_path = (
            model_name_or_path
            or model_name
            or os.environ.get("YOLO_WORLD_MODEL")
            or "models/yolov8s-worldv2.pt"
        )

        self.conf = float(os.environ.get("YOLO_WORLD_CONF", conf))
        self.iou = float(os.environ.get("YOLO_WORLD_IOU", iou))
        self.imgsz = int(os.environ.get("YOLO_WORLD_IMGSZ", imgsz))
        self.max_detections = int(os.environ.get("YOLO_WORLD_MAX_DETECTIONS", max_detections))
        self.device = device or os.environ.get("YOLO_WORLD_DEVICE", None)

        self.api_url = "local-yolo-world"
        self.model_name = self.model_name_or_path

        self.model = None

        prompt_text = os.environ.get("YOLO_WORLD_CLASSES")

        if prompt_text:
            self.classes = _normalize_classes(prompt_text.split(","))
        else:
            self.classes = list(DEFAULT_ZERO_SHOT_CLASSES)

    def set_classes(self, labels: List[str]):
        """
        Update YOLO-World text prompts used for local zero-shot detection.

        The Flask app calls this whenever detection targets are changed in SQLite.
        If the detector model is already loaded, the new target list is applied immediately.

        Note:
        Ultralytics YOLO-World may tokenize class prompts on CPU even when the
        model has already been moved to CUDA. Re-applying classes directly on a
        CUDA-loaded model can therefore raise a CPU/CUDA device mismatch error.
        To keep dynamic label editing stable, prompt text features are rebuilt on
        CPU first, then the detector is moved back to its previous or configured
        device.
        """
        cleaned = _normalize_classes(labels)

        if not cleaned:
            cleaned = list(DEFAULT_ZERO_SHOT_CLASSES)

        self.classes = cleaned
        self._apply_classes_to_loaded_model()

        return self.classes

    def _get_loaded_model_device(self) -> Optional[str]:
        """Return the current torch device of the loaded Ultralytics model."""
        if self.model is None:
            return self.device

        try:
            torch_model = getattr(self.model, "model", None)

            if torch_model is None:
                return self.device

            for parameter in torch_model.parameters():
                return str(parameter.device)
        except Exception:
            pass

        return self.device

    def _move_loaded_model(self, device: str) -> bool:
        """Best-effort model device move."""
        if self.model is None or not device:
            return False

        try:
            self.model.to(device)
            return True
        except Exception as e:
            print(f"YOLO-World device setting skipped: {e}")
            return False

    def _apply_classes_to_loaded_model(self):
        """Apply current classes to YOLO-World without CPU/CUDA mismatch."""
        if self.model is None:
            return

        original_device = self._get_loaded_model_device()
        original_device_text = str(original_device or "").lower()
        restore_device = self.device or original_device

        # Rebuild YOLO-World text features on CPU. This avoids Ultralytics/CLIP
        # token tensors staying on CPU while the text embedding weights are on CUDA.
        if original_device_text and original_device_text != "cpu":
            self._move_loaded_model("cpu")

        try:
            self.model.set_classes(self.classes)
        except RuntimeError as e:
            message = str(e)

            if "Expected all tensors to be on the same device" not in message:
                raise

            # One more conservative retry on CPU for environments where the
            # first move did not reach every submodule/cache.
            self._move_loaded_model("cpu")
            self.model.set_classes(self.classes)

        if restore_device and str(restore_device).lower() != "cpu":
            self._move_loaded_model(str(restore_device))

    def load(self):
        if self.model is not None:
            return

        try:
            from ultralytics import YOLOWorld
        except ImportError as e:
            raise RuntimeError(
                "Ultralytics is not installed. "
                "Please run: pip install -U ultralytics"
            ) from e

        print(f"Loading YOLO-World detector: {self.model_name_or_path}")

        self.model = YOLOWorld(self.model_name_or_path)

        self._apply_classes_to_loaded_model()

        print(f"YOLO-World classes: {self.classes}")

    def warmup(self):
        """
        Warm up YOLO-World local inference.
        """
        self.load()

        dummy = Image.new("RGB", (640, 640), color="white")

        try:
            _ = self.detect_products_from_image(dummy)
        except Exception as e:
            # Blank image may produce no detections. That is acceptable.
            print(f"YOLO-World warmup warning: {e}")

    def detect_products(self, image_path: str) -> List[Dict[str, Any]]:
        """
        Detect product-like boxes from image path.
        """
        self.load()

        image = Image.open(image_path).convert("RGB")
        return self.detect_products_from_image(image, source=image_path)

    def detect_products_from_image(
        self,
        image: Image.Image,
        source: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect product-like boxes from a PIL image.

        If source path is provided, Ultralytics uses that path.
        Otherwise, it uses the PIL image.
        """
        self.load()

        image = image.convert("RGB")
        image_width, image_height = image.size

        predict_source = source if source else image

        predict_kwargs = {
            "source": predict_source,
            "conf": self.conf,
            "iou": self.iou,
            "imgsz": self.imgsz,
            "max_det": self.max_detections,
            "verbose": False,
        }

        if self.device:
            predict_kwargs["device"] = self.device

        results = self.model.predict(**predict_kwargs)

        if not results:
            return []

        result = results[0]

        if result.boxes is None or len(result.boxes) == 0:
            return []

        detections = []

        boxes_xyxy = result.boxes.xyxy.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        class_ids = result.boxes.cls.cpu().tolist()

        for box, score, class_id in zip(boxes_xyxy, scores, class_ids):
            yolo_box = self._xyxy_to_normalized_yolo(
                box,
                image_width,
                image_height,
            )

            if yolo_box is None:
                continue

            cls_index = int(class_id)
            label = self._get_label(cls_index)

            detections.append({
                "det_id": len(detections) + 1,
                "bbox": yolo_box,
                "detector_label": label,
                "detector_score": round(float(score), 4),
            })

        detections = self._deduplicate_boxes(detections)

        for idx, det in enumerate(detections, start=1):
            det["det_id"] = idx

        return detections

    def _get_label(self, cls_index: int) -> str:
        try:
            if hasattr(self.model, "names"):
                names = self.model.names

                if isinstance(names, dict) and cls_index in names:
                    return str(names[cls_index])

                if isinstance(names, list) and 0 <= cls_index < len(names):
                    return str(names[cls_index])
        except Exception:
            pass

        if 0 <= cls_index < len(self.classes):
            return self.classes[cls_index]

        return "product"

    def _xyxy_to_normalized_yolo(
        self,
        box: List[float],
        image_width: int,
        image_height: int,
    ) -> Optional[List[float]]:
        x1, y1, x2, y2 = box

        x1 = max(0, min(image_width - 1, float(x1)))
        x2 = max(0, min(image_width - 1, float(x2)))
        y1 = max(0, min(image_height - 1, float(y1)))
        y2 = max(0, min(image_height - 1, float(y2)))

        if x2 <= x1 or y2 <= y1:
            return None

        box_w = x2 - x1
        box_h = y2 - y1

        if box_w < 8 or box_h < 8:
            return None

        x_center = (x1 + x2) / 2 / image_width
        y_center = (y1 + y2) / 2 / image_height
        width = box_w / image_width
        height = box_h / image_height

        return [
            round(x_center, 6),
            round(y_center, 6),
            round(width, 6),
            round(height, 6),
        ]

    def _deduplicate_boxes(
        self,
        detections: List[Dict[str, Any]],
        iou_threshold: float = 0.70,
    ) -> List[Dict[str, Any]]:
        """
        Lightweight NMS after YOLO-World prediction.

        YOLO already performs NMS, but open-vocabulary prompts may overlap.
        This removes near-duplicate product boxes.
        """
        if not detections:
            return []

        sorted_detections = sorted(
            detections,
            key=lambda item: item.get("detector_score", 0.0),
            reverse=True,
        )

        kept = []

        for det in sorted_detections:
            duplicate = False

            for existing in kept:
                if self._iou_yolo(det["bbox"], existing["bbox"]) >= iou_threshold:
                    duplicate = True
                    break

            if not duplicate:
                kept.append(det)

        return kept

    def _iou_yolo(self, a: List[float], b: List[float]) -> float:
        ax, ay, aw, ah = a
        bx, by, bw, bh = b

        ax1 = ax - aw / 2
        ay1 = ay - ah / 2
        ax2 = ax + aw / 2
        ay2 = ay + ah / 2

        bx1 = bx - bw / 2
        by1 = by - bh / 2
        bx2 = bx + bw / 2
        by2 = by + bh / 2

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h

        area_a = aw * ah
        area_b = bw * bh
        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union