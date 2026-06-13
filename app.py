import atexit
import base64
import binascii
import io
import os
import shutil
import uuid
from collections import Counter

from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from PIL import Image, ImageDraw, ImageFont

from config import (
    PRODUCT_IMAGE_FOLDER,
    TEMP_UPLOAD_FOLDER,
)
from database import (
    init_db,
    insert_product,
    get_all_products,
    delete_product,
    get_product,
    update_product,
    get_zero_shot_labels,
    insert_zero_shot_label,
    delete_zero_shot_label,
)
from SigLIP import SigLIPModelWrapper
from retrieval import add_embedding, delete_embedding, search_top_k, load_embeddings
from ZeroShotDetector import ZeroShotDetector


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10MB

model = SigLIPModelWrapper()

detector = ZeroShotDetector(
    model_name_or_path=os.environ.get(
        "YOLO_WORLD_MODEL",
        "models/yolov8s-worldv2.pt",
    ),
)

init_db()

CHECKOUT_STATE = {}
CURRENT_CHECKOUT_MODE = None


# --------------------------------------------------
# Utility
# --------------------------------------------------

def _font(size=18):
    try:
        return ImageFont.truetype("arial.ttf", size=size)
    except Exception:
        return ImageFont.load_default()


def _save_uploaded_image(file_storage, target_dir):
    if not file_storage or not file_storage.filename:
        raise ValueError("No image file provided.")

    os.makedirs(target_dir, exist_ok=True)

    filename = secure_filename(file_storage.filename)
    ext = os.path.splitext(filename)[1].lower()

    if ext == "":
        ext = ".jpg"

    new_name = f"{uuid.uuid4().hex}{ext}"
    path = os.path.join(target_dir, new_name)

    file_storage.save(path)
    return path


def _save_captured_image(data_url, target_dir):
    """Save a browser camera frame submitted as a base64 data URL."""
    data_url = (data_url or "").strip()

    if not data_url:
        raise ValueError("No captured camera image provided.")

    if "," in data_url:
        header, encoded = data_url.split(",", 1)
        if "base64" not in header.lower():
            raise ValueError("Captured image must be base64 encoded.")
    else:
        encoded = data_url

    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as e:
        raise ValueError("Captured image data is invalid.") from e

    if not raw:
        raise ValueError("Captured image data is empty.")

    os.makedirs(target_dir, exist_ok=True)

    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as e:
        raise ValueError("Captured image is not a readable image.") from e

    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(target_dir, filename)
    image.save(path, format="JPEG", quality=92)

    return path


def _zero_shot_labels_payload():
    return [
        {"id": row[0], "name": row[1]}
        for row in get_zero_shot_labels()
    ]


def _sync_detector_labels():
    labels = [row[1] for row in get_zero_shot_labels()]
    return detector.set_classes(labels)


def _cleanup_file(path):
    try:
        if path and os.path.exists(path) and os.path.isfile(path):
            os.remove(path)
    except Exception as e:
        print(f"File deletion failed: {e}")


def _cleanup_checkout_files(temp_filename):
    if not temp_filename:
        return

    state = CHECKOUT_STATE.pop(temp_filename, None)

    original_path = os.path.join(TEMP_UPLOAD_FOLDER, temp_filename)
    annotated_path = os.path.join(
        TEMP_UPLOAD_FOLDER,
        "annotated",
        f"{os.path.splitext(temp_filename)[0]}.png",
    )

    _cleanup_file(original_path)
    _cleanup_file(annotated_path)

    if state:
        if state.get("annotated_path"):
            _cleanup_file(state["annotated_path"])

        for crop_path in state.get("crop_paths", []):
            _cleanup_file(crop_path)


def _clear_all_checkout_temp():
    global CHECKOUT_STATE

    CHECKOUT_STATE.clear()

    if os.path.exists(TEMP_UPLOAD_FOLDER):
        try:
            shutil.rmtree(TEMP_UPLOAD_FOLDER)
        except Exception as e:
            print(f"Temp folder cleanup failed: {e}")

    os.makedirs(TEMP_UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(TEMP_UPLOAD_FOLDER, "annotated"), exist_ok=True)
    os.makedirs(os.path.join(TEMP_UPLOAD_FOLDER, "crops"), exist_ok=True)


def _normalize_bbox(bbox, width, height):
    """
    Input:
        [x_center, y_center, box_width, box_height], normalized 0~1

    Output:
        x1, y1, x2, y2 in pixel coordinates
    """
    x_center, y_center, bw, bh = bbox

    x1 = int((x_center - bw / 2) * width)
    y1 = int((y_center - bh / 2) * height)
    x2 = int((x_center + bw / 2) * width)
    y2 = int((y_center + bh / 2) * height)

    return x1, y1, x2, y2


def _crop_bbox_image(image, bbox):
    width, height = image.size

    x1, y1, x2, y2 = _normalize_bbox(bbox, width, height)

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(width, x2)
    y2 = min(height, y2)

    if x2 <= x1 or y2 <= y1:
        return None

    crop = image.crop((x1, y1, x2, y2))

    if crop.size[0] < 8 or crop.size[1] < 8:
        return None

    return crop.convert("RGB")


def _save_crop_image(crop, temp_filename, det_id):
    crop_dir = os.path.join(TEMP_UPLOAD_FOLDER, "crops")
    os.makedirs(crop_dir, exist_ok=True)

    base_name = os.path.splitext(temp_filename)[0]
    crop_filename = f"{base_name}_box_{det_id}.png"
    crop_path = os.path.join(crop_dir, crop_filename)

    crop.save(crop_path, format="PNG")
    return crop_path


# --------------------------------------------------
# Retrieval
# --------------------------------------------------

def _get_image_matches_from_image(image, top_k=5):
    """
    SigLIP retrieval from a PIL Image.
    """
    emb = model.get_embedding_from_image(image)

    products = get_all_products()
    valid_ids = [p[0] for p in products]

    matches = search_top_k(emb, valid_ids, k=top_k)
    id_to_product = {p[0]: p for p in products}

    candidates = []
    for item in matches:
        product = id_to_product.get(item["product_id"])

        if not product:
            continue

        candidates.append({
            "product_id": product[0],
            "name": product[1],
            "price": float(product[2]),
            "score": round(float(item["score"]), 4),
        })

    return candidates


def _get_image_matches_from_path(image_path, top_k=5):
    image = Image.open(image_path).convert("RGB")
    return _get_image_matches_from_image(image, top_k=top_k)


def _build_summary_from_single_result(selected):
    if not selected:
        return [], 0.0

    price = float(selected["price"])

    return [
        {
            "name": selected["name"],
            "price": price,
            "quantity": 1,
            "subtotal": round(price, 2),
        }
    ], round(price, 2)


def _build_summary_from_detections(detections):
    counter = Counter()
    price_map = {}

    for det in detections:
        selected = det.get("selected")

        if selected:
            name = selected["name"]
            counter[name] += 1
            price_map[name] = float(selected["price"])

    items = []
    total = 0.0

    for name, quantity in counter.items():
        price = price_map.get(name, 0.0)
        subtotal = price * quantity

        items.append({
            "name": name,
            "price": price,
            "quantity": quantity,
            "subtotal": round(subtotal, 2),
        })

        total += subtotal

    return items, round(total, 2)


# --------------------------------------------------
# Recognition State
# --------------------------------------------------

def _build_single_state(image_path, selected_pid=None):
    top5 = _get_image_matches_from_path(image_path, top_k=5)

    selected = None

    if selected_pid is not None:
        for candidate in top5:
            if candidate["product_id"] == selected_pid:
                selected = candidate
                break

    if selected is None and top5:
        selected = top5[0]

    items, total = _build_summary_from_single_result(selected)

    return {
        "mode": "single",
        "top5": top5,
        "selected": selected,
        "detections": [],
        "items": items,
        "total": total,
    }


def _build_detection_state(image_path, temp_filename):
    """
    Multiple-products flow:

    1. ZeroShotDetector detects product-like boxes locally with YOLO-World.
    2. Each bbox is cropped from the original image.
    3. Each crop is sent to SigLIP retrieval.
    4. Each box gets its own Top-5 candidates.
    5. Checkout summary is calculated from selected candidates.
    """
    original = Image.open(image_path).convert("RGB")

    try:
        located_boxes = detector.detect_products(image_path)
    except Exception as e:
        raise RuntimeError(
            "ZeroShotDetector detection failed. "
            "Please make sure Ultralytics is installed and the YOLO-World model path is correct."
        ) from e

    detections = []
    crop_paths = []

    for box in located_boxes:
        det_id = box["det_id"]
        bbox = box["bbox"]

        crop = _crop_bbox_image(original, bbox)

        if crop is None:
            top5 = []
            crop_path = None
        else:
            crop_path = _save_crop_image(crop, temp_filename, det_id)
            crop_paths.append(crop_path)
            top5 = _get_image_matches_from_image(crop, top_k=5)

        selected = top5[0] if top5 else None

        detections.append({
            "det_id": det_id,
            "bbox": bbox,
            "top5": top5,
            "selected": selected,
            "selected_pid": selected["product_id"] if selected else None,
            "crop_path": crop_path,
        })

    items, total = _build_summary_from_detections(detections)

    return {
        "mode": "detection",
        "top5": [],
        "selected": None,
        "detections": detections,
        "items": items,
        "total": total,
        "crop_paths": crop_paths,
    }


# --------------------------------------------------
# Annotation
# --------------------------------------------------

def _annotate_single(image_path, output_path, selected_name):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    w, h = img.size
    font = _font(18)
    small_font = _font(14)

    banner_h = max(44, int(h * 0.12))

    draw.rectangle([0, 0, w, banner_h], fill=(235, 244, 255))
    draw.text((18, 12), "Best match", fill=(75, 105, 155), font=small_font)
    draw.text(
        (18, 24),
        selected_name or "Not selected yet",
        fill=(25, 35, 50),
        font=font,
    )

    img.save(output_path, format="PNG")
    return output_path


def _annotate_detection(image_path, output_path, detections):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    width, height = img.size

    box_color = (102, 168, 255)
    label_bg = (235, 244, 255)
    text_color = (25, 35, 50)

    font = _font(16)
    small_font = _font(13)

    for det in detections:
        x1, y1, x2, y2 = _normalize_bbox(det["bbox"], width, height)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width - 1, x2)
        y2 = min(height - 1, y2)

        draw.rectangle(
            [x1, y1, x2, y2],
            outline=box_color,
            width=max(2, width // 250),
        )

        selected = det.get("selected")
        label_name = selected["name"] if selected else "Unassigned"
        label = f"#{det['det_id']}  {label_name}"

        label_bbox = draw.textbbox((0, 0), label, font=font)
        text_w = label_bbox[2] - label_bbox[0]
        text_h = label_bbox[3] - label_bbox[1]

        pad_x = 8
        pad_y = 5

        label_x1 = x1
        label_y1 = max(0, y1 - text_h - pad_y * 2 - 2)
        label_x2 = min(width - 1, label_x1 + text_w + pad_x * 2)
        label_y2 = label_y1 + text_h + pad_y * 2

        draw.rectangle(
            [label_x1, label_y1, label_x2, label_y2],
            fill=label_bg,
        )
        draw.text(
            (label_x1 + pad_x, label_y1 + pad_y),
            label,
            fill=text_color,
            font=font,
        )

    note = "Annotated preview"
    note_bbox = draw.textbbox((0, 0), note, font=small_font)
    note_w = note_bbox[2] - note_bbox[0]
    note_h = note_bbox[3] - note_bbox[1]

    draw.rectangle(
        [12, height - note_h - 18, 12 + note_w + 16, height - 8],
        fill=(255, 255, 255),
    )
    draw.text(
        (20, height - note_h - 14),
        note,
        fill=(90, 100, 120),
        font=small_font,
    )

    img.save(output_path, format="PNG")
    return output_path


# --------------------------------------------------
# Render
# --------------------------------------------------

def _state_to_summary(result):
    if not result:
        return [], 0.0

    return result.get("items", []), result.get("total", 0.0)


def _render_checkout(
    mode="single",
    temp_filename="",
    result=None,
    error=None,
    checkout_done=False,
):
    uploaded_url = None
    annotated_url = None

    if temp_filename:
        temp_path = os.path.join(TEMP_UPLOAD_FOLDER, temp_filename)

        if os.path.exists(temp_path):
            uploaded_url = url_for(
                "static",
                filename=f"uploads/temp/{temp_filename}",
            )

        state = CHECKOUT_STATE.get(temp_filename)

        if state and state.get("annotated_filename"):
            annotated_url = url_for(
                "static",
                filename=f"uploads/temp/annotated/{state['annotated_filename']}",
            )

    if result is None and temp_filename and CHECKOUT_STATE.get(temp_filename):
        result = CHECKOUT_STATE[temp_filename]["result"]

    items, total = _state_to_summary(result)

    return render_template(
        "checkout.html",
        mode=mode,
        result=result,
        error=error,
        checkout_done=checkout_done,
        temp_filename=temp_filename,
        uploaded_url=uploaded_url,
        annotated_url=annotated_url,
        items=items,
        total=total,
    )


def _process_checkout_image(uploaded_path, temp_filename, mode):
    """Run the selected checkout pipeline and create the annotated preview."""
    annotated_path = os.path.join(
        TEMP_UPLOAD_FOLDER,
        "annotated",
        f"{os.path.splitext(temp_filename)[0]}.png",
    )

    if mode in ["detection", "camera"]:
        result = _build_detection_state(
            image_path=uploaded_path,
            temp_filename=temp_filename,
        )
        result["mode"] = mode

        _annotate_detection(
            image_path=uploaded_path,
            output_path=annotated_path,
            detections=result["detections"],
        )
    else:
        result = _build_single_state(uploaded_path)
        result["mode"] = mode
        selected_name = result["selected"]["name"] if result.get("selected") else None

        _annotate_single(
            image_path=uploaded_path,
            output_path=annotated_path,
            selected_name=selected_name,
        )

    CHECKOUT_STATE[temp_filename] = {
        "mode": mode,
        "image_path": uploaded_path,
        "annotated_filename": os.path.basename(annotated_path),
        "annotated_path": annotated_path,
        "crop_paths": result.get("crop_paths", []),
        "result": result,
    }

    return result


def _render_register(error=None):
    return render_template(
        "register.html",
        products=get_all_products(),
        zero_shot_labels=get_zero_shot_labels(),
        error=error,
    )


try:
    _sync_detector_labels()
except Exception as e:
    print(f"Initial detection target sync skipped: {e}")


# --------------------------------------------------
# Routes
# --------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        price_raw = request.form.get("price", "").strip()
        file = request.files.get("image")

        if not name or not price_raw or not file or not file.filename:
            return _render_register(
                error="Please fill in all fields and select an image."
            )

        try:
            price = float(price_raw)
        except ValueError:
            return _render_register(
                error="Price must be a valid number."
            )

        image_path = None
        pid = None

        try:
            image_path = _save_uploaded_image(file, PRODUCT_IMAGE_FOLDER)
            pid = insert_product(name, price, image_path)

            emb = model.get_embedding(image_path)
            add_embedding(pid, emb)

            return redirect(url_for("register"))

        except Exception as e:
            if pid is not None:
                delete_embedding(pid)
                delete_product(pid)

            _cleanup_file(image_path)

            return _render_register(
                error=f"Failed to register product: {e}"
            )

    return _render_register()


@app.route("/delete/<int:pid>")
def delete(pid):
    product = get_product(pid)

    if product:
        image_path = product[3]
        delete_product(pid)
        delete_embedding(pid)
        _cleanup_file(image_path)

    return redirect(url_for("register"))


@app.route("/edit/<int:pid>", methods=["POST"])
def edit(pid):
    name = request.form.get("name", "").strip()
    price_raw = request.form.get("price", "").strip()

    if not get_product(pid):
        return _render_register(error="Product not found.")

    if not name or not price_raw:
        return _render_register(error="Please fill in product name and price.")

    try:
        price = float(price_raw)
    except ValueError:
        return _render_register(error="Price must be a valid number.")

    try:
        update_product(pid, name, price)
    except Exception as e:
        return _render_register(error=f"Failed to update product: {e}")

    return redirect(url_for("register"))


@app.route("/checkout", methods=["GET", "POST"])
def checkout():
    global CURRENT_CHECKOUT_MODE

    mode = request.values.get("mode", "single")
    temp_filename = request.values.get("temp_filename", "").strip()

    error = None
    result = None
    checkout_done = False

    if mode not in ["single", "detection", "camera"]:
        mode = "single"

    # Clear temp files when switching subpage.
    if CURRENT_CHECKOUT_MODE is None:
        CURRENT_CHECKOUT_MODE = mode
    elif CURRENT_CHECKOUT_MODE != mode:
        _clear_all_checkout_temp()
        CURRENT_CHECKOUT_MODE = mode
        temp_filename = ""

    if request.method == "POST":
        action = request.form.get("action", "analyze")

        if action == "analyze":
            old_temp = request.form.get("temp_filename", "").strip()
            file = request.files.get("image")
            captured_image = request.form.get("captured_image", "").strip()

            if mode == "camera":
                has_input = bool(captured_image)
                missing_message = "Please capture an image first."
            else:
                has_input = bool(file and file.filename)
                missing_message = "Please select an image first."

            if not has_input:
                return _render_checkout(
                    mode=mode,
                    temp_filename=old_temp,
                    error=missing_message,
                )

            uploaded_path = None

            try:
                if old_temp:
                    _cleanup_checkout_files(old_temp)

                if mode == "camera":
                    uploaded_path = _save_captured_image(captured_image, TEMP_UPLOAD_FOLDER)
                else:
                    uploaded_path = _save_uploaded_image(file, TEMP_UPLOAD_FOLDER)

                temp_filename = os.path.basename(uploaded_path)
                result = _process_checkout_image(uploaded_path, temp_filename, mode)

            except Exception as e:
                _cleanup_file(uploaded_path)
                error = f"Recognition failed: {e}"

        elif action == "apply_correction":
            temp_filename = request.form.get("temp_filename", "").strip()
            det_id_raw = request.form.get("det_id", "").strip()
            selected_pid_raw = request.form.get("selected_pid", "").strip()

            if not temp_filename or temp_filename not in CHECKOUT_STATE:
                error = "Please analyze an image first."
            else:
                state = CHECKOUT_STATE[temp_filename]
                image_path = state["image_path"]
                result = state["result"]

                selected_pid = int(selected_pid_raw) if selected_pid_raw.isdigit() else None
                det_id = int(det_id_raw) if det_id_raw.isdigit() else None

                if state["mode"] == "single":
                    selected = None

                    for candidate in result.get("top5", []):
                        if candidate["product_id"] == selected_pid:
                            selected = candidate
                            break

                    if selected is None and result.get("top5"):
                        selected = result["top5"][0]

                    result["selected"] = selected
                    items, total = _build_summary_from_single_result(selected)

                    result["items"] = items
                    result["total"] = total

                    _annotate_single(
                        image_path=image_path,
                        output_path=state["annotated_path"],
                        selected_name=selected["name"] if selected else None,
                    )

                else:
                    for det in result.get("detections", []):
                        if det["det_id"] == det_id:
                            new_selected = None

                            for candidate in det.get("top5", []):
                                if candidate["product_id"] == selected_pid:
                                    new_selected = candidate
                                    break

                            if new_selected is None and det.get("top5"):
                                new_selected = det["top5"][0]

                            det["selected"] = new_selected
                            det["selected_pid"] = (
                                new_selected["product_id"] if new_selected else None
                            )
                            break

                    items, total = _build_summary_from_detections(result.get("detections", []))
                    result["items"] = items
                    result["total"] = total

                    _annotate_detection(
                        image_path=image_path,
                        output_path=state["annotated_path"],
                        detections=result["detections"],
                    )

                state["result"] = result
                CHECKOUT_STATE[temp_filename] = state

                return _render_checkout(
                    mode=state["mode"],
                    temp_filename=temp_filename,
                    result=result,
                )

        elif action == "checkout":
            temp_filename = request.form.get("temp_filename", "").strip()

            if temp_filename:
                _cleanup_checkout_files(temp_filename)
                checkout_done = True

                return _render_checkout(
                    mode=mode,
                    temp_filename="",
                    result=None,
                    checkout_done=True,
                )

            error = "No temporary image found to check out."

    return _render_checkout(
        mode=mode,
        temp_filename=temp_filename,
        result=result,
        error=error,
        checkout_done=checkout_done,
    )


# --------------------------------------------------
# API Routes
# --------------------------------------------------

@app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({
        "status": "ok",
        "model_loaded": model.model is not None,
        "embedding_count": len(load_embeddings()),
        "active_checkout_sessions": len(CHECKOUT_STATE),
        "detector_backend": "YOLO-World",
        "detector_model": detector.model_name,
        "detector_classes": detector.classes,
        "detector_conf": detector.conf,
        "detector_iou": detector.iou,
    })


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    mode = request.form.get("mode", "single")

    if mode not in ["single", "detection", "camera"]:
        mode = "single"

    file = request.files.get("image")
    captured_image = request.form.get("captured_image", "").strip()

    if mode == "camera":
        has_input = bool(captured_image) or bool(file and file.filename)
        missing_error = "No captured image uploaded."
    else:
        has_input = bool(file and file.filename)
        missing_error = "No image uploaded."

    if not has_input:
        return jsonify({
            "ok": False,
            "error": missing_error,
        }), 400

    uploaded_path = None

    try:
        if mode == "camera" and captured_image:
            uploaded_path = _save_captured_image(captured_image, TEMP_UPLOAD_FOLDER)
        else:
            uploaded_path = _save_uploaded_image(file, TEMP_UPLOAD_FOLDER)

        temp_filename = os.path.basename(uploaded_path)
        result = _process_checkout_image(uploaded_path, temp_filename, mode)

        return jsonify({
            "ok": True,
            "mode": mode,
            "temp_filename": temp_filename,
            "result": result,
        })

    except Exception as e:
        _cleanup_file(uploaded_path)
        return jsonify({
            "ok": False,
            "error": str(e),
        }), 500


@app.route("/api/zero-shot-labels", methods=["GET"])
def api_get_zero_shot_labels():
    return jsonify({
        "ok": True,
        "labels": _zero_shot_labels_payload(),
        "detector_classes": detector.classes,
    })


@app.route("/api/zero-shot-labels", methods=["POST"])
def api_add_zero_shot_label():
    label = request.form.get("label", "").strip()

    try:
        _, created = insert_zero_shot_label(label)
        classes = _sync_detector_labels()

        return jsonify({
            "ok": True,
            "created": created,
            "labels": _zero_shot_labels_payload(),
            "detector_classes": classes,
        })

    except ValueError as e:
        return jsonify({
            "ok": False,
            "error": str(e),
        }), 400
    except Exception as e:
        return jsonify({
            "ok": False,
            "error": f"Failed to add detection target: {e}",
        }), 500


@app.route("/api/zero-shot-labels/<int:label_id>", methods=["DELETE"])
def api_delete_zero_shot_label(label_id):
    try:
        deleted = delete_zero_shot_label(label_id)

        if not deleted:
            return jsonify({
                "ok": False,
                "error": "Detection target not found.",
            }), 404

        classes = _sync_detector_labels()

        return jsonify({
            "ok": True,
            "labels": _zero_shot_labels_payload(),
            "detector_classes": classes,
        })

    except Exception as e:
        return jsonify({
            "ok": False,
            "error": f"Failed to delete detection target: {e}",
        }), 500


@app.route("/api/checkout", methods=["POST"])
def api_checkout():
    temp_filename = request.form.get("temp_filename", "").strip()

    if not temp_filename:
        return jsonify({
            "ok": False,
            "error": "No temporary image found.",
        }), 400

    if temp_filename not in CHECKOUT_STATE:
        return jsonify({
            "ok": False,
            "error": "Temporary image not found.",
        }), 404

    _cleanup_checkout_files(temp_filename)

    return jsonify({
        "ok": True,
        "message": "Checkout completed and temporary image removed.",
    })


@app.route("/api/cleanup-temp", methods=["POST"])
def api_cleanup_temp():
    temp_filename = request.form.get("temp_filename", "").strip()

    if temp_filename:
        _cleanup_checkout_files(temp_filename)
    else:
        _clear_all_checkout_temp()

    return jsonify({
        "ok": True,
        "message": "Temporary checkout files cleaned.",
    })


# --------------------------------------------------
# Startup / Shutdown
# --------------------------------------------------

def startup():
    print("Clearing temp checkout files...")
    _clear_all_checkout_temp()

    print("Applying detection targets from SQLite...")
    try:
        _sync_detector_labels()
    except Exception as e:
        print(f"Detection target sync skipped: {e}")

    print("Loading SigLIP model...")
    model.load()

    print("Warming up SigLIP model...")
    model.warmup()

    print("Warming up ZeroShotDetector / YOLO-World detector...")
    try:
        detector.warmup()
        print("ZeroShotDetector ready.")
    except Exception as e:
        print(f"ZeroShotDetector warmup skipped: {e}")
        print("Multiple Products mode will fail until Ultralytics and the YOLO-World model are available.")

    print("System ready.")


def shutdown_cleanup():
    try:
        print("Clearing temp checkout files before shutdown...")
        _clear_all_checkout_temp()
    except Exception as e:
        print(f"Shutdown cleanup failed: {e}")


atexit.register(shutdown_cleanup)


if __name__ == "__main__":
    startup()
    app.run(debug=True, use_reloader=False)