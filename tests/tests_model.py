from ultralytics import YOLO
import numpy as np
import yaml


def load_config():
    with open('config.yaml') as f:
        return yaml.safe_load(f)


def test_model_loads():
    config = load_config()
    model = YOLO(config['model']['path'])
    assert model is not None


def test_model_classes():
    config = load_config()
    model = YOLO(config['model']['path'])
    assert 'bullish_fvg' in model.names.values()
    assert 'bearish_fvg' in model.names.values()


def test_model_inference():
    config = load_config()
    model = YOLO(config['model']['path'])

    # Dummy white image
    dummy = np.ones((640, 640, 3), dtype=np.uint8) * 255
    results = model(dummy, conf=0.25, verbose=False)

    assert results is not None
    assert len(results) == 1


def test_inference_output_format():
    config = load_config()
    model = YOLO(config['model']['path'])

    dummy = np.ones((640, 640, 3), dtype=np.uint8) * 255
    results = model(dummy, conf=0.25, verbose=False)[0]

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        assert cls_id in [0, 1]
        assert 0.0 <= conf <= 1.0
        assert x1 < x2
        assert y1 < y2