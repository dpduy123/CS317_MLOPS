import cv2
import numpy as np
import torch
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer


class ANPR:
    def __init__(self, model_path: str = "yolo11n.pt"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.reader = LicensePlateRecognizer('cct-s-v2-global-model')

    def detect_plates(self, im0: np.ndarray):
        results = self.model.predict(im0, verbose=False, device=self.device)
        if results and results[0].boxes is not None:
            return results[0].boxes.xyxy.cpu().numpy()
        return []

    def extract_text(self, im0: np.ndarray, bbox: np.ndarray):
        x1, y1, x2, y2 = map(int, bbox)
        roi = im0[y1:y2, x1:x2]

        if roi.size == 0:
            return ""

        results = self.reader.run(roi)
        if not results:
            return ""

        pred = results[0]

        if hasattr(pred, 'text'):
            return pred.text
        elif hasattr(pred, 'plate'):
            return pred.plate
        elif hasattr(pred, 'chars'):
            return "".join(pred.chars) if isinstance(pred.chars, list) else pred.chars

        return str(pred)

    def infer_image_array(self, im0: np.ndarray):
        boxes = self.detect_plates(im0)

        records = []
        for bbox in boxes:
            text = self.extract_text(im0, bbox)
            if text:
                records.append({
                    "text": text,
                    "bbox": bbox.tolist()
                })

        return records