import gradio as gr
import cv2
import numpy as np
import yaml
import time
from pathlib import Path
from ultralytics import YOLO
from PIL import Image, ImageDraw


# Load config
def load_config() -> dict:
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


config = load_config()

# Load model
model_path = Path(__file__).parent / config['model']['path']
model = YOLO(str(model_path))
conf_threshold = config['model']['conf_threshold']

COLORS = {
    'bullish_fvg': (0, 255, 0),
    'bearish_fvg': (255, 0, 0),
}


#  Inference
def detect_fvg(image: np.ndarray, confidence: float):
    if image is None:
        return None, "No image provided."

    # Convert RGB to BGR for OpenCV
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Run inference
    start = time.perf_counter()
    results = model(frame, conf=confidence, verbose=False)[0]
    inference_time_ms = (time.perf_counter() - start) * 1000

    # Draw boxes
    annotated = image.copy()
    pil_img = Image.fromarray(annotated)
    draw = ImageDraw.Draw(pil_img)

    bullish_count = 0
    bearish_count = 0
    detection_lines = []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        conf = round(float(box.conf[0]), 4)
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        color = COLORS.get(cls_name, (255, 255, 255))

        if cls_name == 'bullish_fvg':
            bullish_count += 1
        else:
            bearish_count += 1

        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, y1 - 12), f"{cls_name} {conf:.2f}", fill=color)
        detection_lines.append(f"• {cls_name} — confidence: {conf:.2%} | box: ({x1}, {y1}, {x2}, {y2})")

    # Build summary
    summary = f"""
## Detection Results

**Total FVGs detected:** {len(results.boxes)}
**Bullish FVGs:** {bullish_count}
**Bearish FVGs:** {bearish_count}
**Inference time:** {inference_time_ms:.1f}ms

### Detections
{"".join([f"{line}      " for line in detection_lines]) if detection_lines else "No FVGs detected."}
    """.strip()

    return np.array(pil_img), summary


#  UI
with gr.Blocks(title="FVG Detector", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # NQ Fair Value Gap (FVG) Detector
    Upload a TradingView screenshot of NQ futures and detect **Bullish** and **Bearish** Fair Value Gaps.

    > Trained on NQ1! charts with white background, no indicators, timeframes: 1m, 5m
    """)

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Chart Screenshot",
                type="numpy"
            )
            confidence_slider = gr.Slider(
                minimum=0.1,
                maximum=0.9,
                value=conf_threshold,
                step=0.05,
                label="Confidence Threshold"
            )
            detect_btn = gr.Button("Detect FVGs", variant="primary")

        with gr.Column():
            image_output = gr.Image(label="Detections")
            summary_output = gr.Markdown(label="Results")

    detect_btn.click(
        fn=detect_fvg,
        inputs=[image_input, confidence_slider],
        outputs=[image_output, summary_output]
    )

    gr.Markdown("""
    ---
    **Classes:** 🟢 Bullish FVG | 🔴 Bearish FVG

    **Links:** [GitHub](https://github.com/ibi111/market-structure-cv) · 
    [Model on HuggingFace](https://huggingface.co/ibtsamsadiq/nq-fvg-detector) · 
    [Dataset on Kaggle](https://www.kaggle.com/datasets/ibtsamsadiq/nq-fvg-dataset)
    """)

if __name__ == '__main__':
    demo.launch()