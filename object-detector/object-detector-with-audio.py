import torch
import numpy as np
import scipy.io.wavfile
from transformers import (
    AutoImageProcessor,
    AutoModelForObjectDetection,
    VitsModel,
    AutoTokenizer,
)
from PIL import ImageDraw
import gradio as gr

# ---------------------------
# 1. Load models once at startup
# ---------------------------
# Loading outside the functions means we download and initialize
# the models only once, not on every prediction.

# --- Object detection model (DETR) ---
DETECTION_MODEL_NAME = "facebook/detr-resnet-50"
processor = AutoImageProcessor.from_pretrained(DETECTION_MODEL_NAME)
detection_model = AutoModelForObjectDetection.from_pretrained(DETECTION_MODEL_NAME)
detection_model.eval()

# --- Text-to-speech model (MMS-TTS, English) ---
# This model converts a plain text description into spoken audio.
TTS_MODEL_NAME = "facebook/mms-tts-eng"
tts_tokenizer = AutoTokenizer.from_pretrained(TTS_MODEL_NAME)
tts_model = VitsModel.from_pretrained(TTS_MODEL_NAME)
tts_model.eval()

# Automatically use GPU if available, otherwise fall back to CPU.
device = "cuda" if torch.cuda.is_available() else "cpu"
detection_model.to(device)
tts_model.to(device)

# A small color palette to cycle through so each box gets a different color.
COLORS = [
    "#e6194b",
    "#3cb44b",
    "#ffe119",
    "#4363d8",
    "#f58231",
    "#911eb4",
    "#46f0f0",
    "#f032e6",
    "#bcf60c",
    "#fabebe",
]


# ---------------------------
# 2. Object detection logic
# ---------------------------
def detect_objects(image, confidence_threshold: float = 0.7):
    """
    Run object detection and draw bounding boxes on the image.

    Returns:
        annotated: the image with boxes drawn on it.
        detections: a list of (label, confidence) tuples, one per detected object.
    """
    inputs = processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = detection_model(**inputs)

    # image.size is (width, height), but target_sizes expects (height, width).
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(
        outputs, target_sizes=target_sizes, threshold=confidence_threshold
    )[0]

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)

    detections = []
    for i, (score, label_id, box) in enumerate(
        zip(results["scores"], results["labels"], results["boxes"])
    ):
        box = [round(coord, 1) for coord in box.tolist()]
        label = detection_model.config.id2label[label_id.item()]
        confidence = round(score.item(), 3)
        color = COLORS[i % len(COLORS)]

        draw.rectangle(box, outline=color, width=3)
        draw.text((box[0], max(box[1] - 12, 0)), f"{label} ({confidence})", fill=color)

        detections.append((label, confidence))

    return annotated, detections


# ---------------------------
# 3. Build a spoken description from detections
# ---------------------------
def build_description(detections) -> str:
    """Turn a list of (label, confidence) tuples into a natural sentence."""
    if not detections:
        return "No objects were detected in the image."

    # Count how many of each object was found (e.g. 2 people, 1 dog).
    counts = {}
    for label, _ in detections:
        counts[label] = counts.get(label, 0) + 1

    parts = [
        f"{count} {label}{'s' if count > 1 else ''}" for label, count in counts.items()
    ]

    if len(parts) == 1:
        objects_text = parts[0]
    else:
        objects_text = ", ".join(parts[:-1]) + f" and {parts[-1]}"

    return f"I detected {objects_text} in the image."


# ---------------------------
# 4. Convert the description text into speech
# ---------------------------
def text_to_speech(text: str):
    """
    Convert text into a spoken audio waveform.

    Returns:
        A tuple of (sample_rate, numpy_audio_array) that Gradio's
        Audio component can play directly.
    """
    inputs = tts_tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        output = tts_model(**inputs).waveform

    # Convert the torch tensor to a numpy array Gradio can play.
    audio = output.squeeze().cpu().numpy()
    sample_rate = tts_model.config.sampling_rate
    return sample_rate, audio


# ---------------------------
# 5. Combined pipeline: image -> boxes + description + audio
# ---------------------------
def process_image(image, confidence_threshold):
    if image is None:
        return None, "Please upload an image.", None

    annotated, detections = detect_objects(image, confidence_threshold)
    description = build_description(detections)
    sample_rate, audio = text_to_speech(description)

    return annotated, description, (sample_rate, audio)


# ---------------------------
# 6. Build the web UI with Gradio
# ---------------------------
with gr.Blocks(title="Object Detector with Audio") as demo:
    gr.Markdown("## Object Detector with Audio Description")
    gr.Markdown(
        "Detects objects using DETR, then reads out a spoken summary using MMS-TTS."
    )

    with gr.Row():
        image_input = gr.Image(type="pil", label="Upload Image")
        image_output = gr.Image(type="pil", label="Detected Objects")

    confidence_slider = gr.Slider(
        minimum=0.1, maximum=0.95, value=0.7, step=0.05, label="Confidence Threshold"
    )
    detect_btn = gr.Button("Detect Objects")

    description_output = gr.Textbox(label="Description")
    audio_output = gr.Audio(label="Spoken Description", type="numpy")

    detect_btn.click(
        fn=process_image,
        inputs=[image_input, confidence_slider],
        outputs=[image_output, description_output, audio_output],
    )

if __name__ == "__main__":
    demo.launch()
