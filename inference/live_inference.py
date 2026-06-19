import cv2
import numpy as np
import mss
import tkinter as tk
import threading
import keyboard
from ultralytics import YOLO
from pathlib import Path


class FVGOverlay:
    def __init__(self, config: dict):
        self.config = config
        self.running = True
        self._auto_timer = None

        # Load model
        model_path = Path(__file__).parent.parent / config['model']['path']
        self.model = YOLO(str(model_path))
        self.conf_threshold = config['model']['conf_threshold']

        # Region, colors, display
        self.region = config['region']
        self.colors = config['colors']
        self.trigger_key = config['inference']['trigger_key']
        self.quit_key = config['inference']['quit_key']
        self.auto_refresh_interval = config['inference']['auto_refresh_interval']
        self.show_labels = config['display']['show_labels']

        self._setup_window()
        self._setup_hotkeys()

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
            text=f"{self.trigger_key.upper()}: detect | {self.quit_key.upper()}: quit | Auto: {self.auto_refresh_interval}s",
            bg='black', fg='white',
            font=('Arial', 10)
        )
        self.status_label.place(x=10, y=10)

    def _setup_hotkeys(self):
        keyboard.add_hotkey(self.trigger_key, self.on_trigger)
        keyboard.add_hotkey(self.quit_key, self.on_close)

    def _schedule_auto_refresh(self):
        """Schedule next auto refresh"""
        self._auto_timer = threading.Timer(
            self.auto_refresh_interval, self._auto_refresh
        )
        self._auto_timer.start()

    def _auto_refresh(self):
        """Called automatically every N seconds"""
        if self.running:
            threading.Thread(target=self.capture_and_infer, daemon=True).start()
            self._schedule_auto_refresh()

    def on_trigger(self):
        """Called on manual hotkey press — also resets auto timer"""
        if self._auto_timer is not None:
            self._auto_timer.cancel()
        threading.Thread(target=self.capture_and_infer, daemon=True).start()
        self._schedule_auto_refresh()

    def capture_and_infer(self):
        self.canvas.after(0, lambda: self.status_label.config(
            text='Detecting...', fg='yellow'
        ))

        with mss.MSS() as sct:
            screenshot = sct.grab(self.region)
            frame = np.array(screenshot)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        results = self.model(frame, conf=self.conf_threshold, verbose=False)[0]

        box_data = []
        detection_count = {'bullish_fvg': 0, 'bearish_fvg': 0}

        for box in results.boxes:
            cls_id = int(box.cls[0])
            cls_name = self.model.names[cls_id]
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            detection_count[cls_name] = detection_count.get(cls_name, 0) + 1
            box_data.append((x1, y1, x2, y2, cls_name, conf))

        self.canvas.after(0, self.update_canvas, box_data, detection_count)

        self.canvas.after(0, lambda: self.status_label.config(
            text=f"{self.trigger_key.upper()}: detect | {self.quit_key.upper()}: quit | Auto: {self.auto_refresh_interval}s",
            fg='white'
        ))

    def update_canvas(self, boxes: list, detection_count: dict):
        self.canvas.delete("all")

        for (x1, y1, x2, y2, cls_name, conf) in boxes:
            color = self.colors.get(cls_name, '#FFFFFF')

            self.canvas.create_rectangle(
                x1, y1, x2, y2,
                outline=color, width=2, fill=''
            )

            if self.show_labels:
                self.canvas.create_text(
                    x1, y1 - 8,
                    text=f"{cls_name} {conf:.2f}",
                    fill=color, anchor='w',
                    font=('Arial', 9, 'bold')
                )

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
        if self._auto_timer is not None:
            self._auto_timer.cancel()
        keyboard.unhook_all()
        self.root.destroy()

    def run(self):
        # Start auto refresh loop
        self._schedule_auto_refresh()
        self.root.mainloop()

