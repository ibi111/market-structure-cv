import cv2
import numpy as np
import mss
import tkinter as tk
import threading
import keyboard
import yaml
from ultralytics import YOLO
from pathlib import Path


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class FVGOverlay:
    def __init__(self, config: dict):
        self.config = config
        self.running = True

        # Load model
        model_path = Path(__file__).parent.parent / config['model']['path']
        self.model = YOLO(str(model_path))
        self.conf_threshold = config['model']['conf_threshold']

        # Region and colors
        self.region = config['region']
        self.colors = config['colors']
        self.trigger_key = config['inference']['trigger_key']

        self._setup_window()
        self._setup_hotkey()

    def _setup_window(self):
        self.root = tk.Tk()
        self.root.title("FVG Detector")
        self.root.attributes('-topmost', True)
        self.root.attributes('-transparentcolor', 'black')
        self.root.attributes('-alpha', 1.0)
        self.root.configure(bg='black')
        self.root.overrideredirect(True)

        self.root.geometry(
            f"{self.region['width']}x{self.region['height']}"
            f"+{self.region['left']}+{self.region['top']}"
        )

        self.canvas = tk.Canvas(
            self.root,
            width=self.region['width'],
            height=self.region['height'],
            bg='black',
            highlightthickness=0
        )
        self.canvas.pack()

        self.status_label = tk.Label(
            self.root,
            text=f"Press '{self.trigger_key.upper()}' to detect | ESC to quit",
            bg='black', fg='white',
            font=('Arial', 10)
        )
        self.status_label.place(x=10, y=10)
        self.root.bind('<Escape>', lambda e: self.on_close())

    def _setup_hotkey(self):
        keyboard.add_hotkey(self.trigger_key, self.on_trigger)

    def on_trigger(self):
        """Called when user presses trigger hotkey"""
        threading.Thread(target=self.capture_and_infer, daemon=True).start()

    def capture_and_infer(self):
        # Update status
        self.canvas.after(0, lambda: self.status_label.config(
            text='Detecting...', fg='yellow'
        ))

        with mss.MSS() as sct:
            # Capture screen region
            screenshot = sct.grab(self.region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        # Run inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]

        # Collect box data
        box_data = []
        detection_count = {'bullish_fvg': 0, 'bearish_fvg': 0}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detection_count[cls_name] = detection_count.get(cls_name, 0) + 1
            box_data.append((x1, y1, x2, y2, cls_name, conf))

        # Update canvas
        self.canvas.after(0, self.update_canvas, box_data, detection_count)

        # Restore status
        self.canvas.after(0, lambda: self.status_label.config(
            text=f"Press '{self.trigger_key.upper()}' to detect | ESC to quit",
            fg='white'
        ))

    def update_canvas(self, boxes: list, detection_count: dict):
        self.canvas.delete("all")

        for (x1, y1, x2, y2, cls_name, conf) in boxes:
            color = self.colors.get(cls_name, '#FFFFFF')

            # Box outline only — background stays transparent
            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color, width=2, fill=''
            )
            # Label with background for readability
            self.canvas.create_text(
                x1, y1 - 8,
                text=f"{cls_name} {conf:.2f}",
                fill=color, anchor='w',
                font=('Arial', 9, 'bold')
            )

        # Detection summary top right
        bull = detection_count.get('bullish_fvg', 0)
        bear = detection_count.get('bearish_fvg', 0)
        self.canvas.create_text(
            self.region['width'] - 10, 10,
            text=f"Bullish: {bull}  |  Bearish: {bear}",
            anchor='ne', fill='white',
            font=('Arial', 11, 'bold')
        )

    def on_close(self):
        self.running = False
        keyboard.unhook_all()
        self.root.destroy()

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    config = load_config()
    app = FVGOverlay(config)
    app.run()