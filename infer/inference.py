import cv2
import numpy as np
import torch
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer


class ANPR:
    def __init__(self, model_path: str = "yolo11n.pt", debug: bool = False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path)
        self.reader = LicensePlateRecognizer("cct-s-v2-global-model")
        self.debug = debug

    # -------------------------
    # DETECTION
    # -------------------------
    def detect_plates(self, im0: np.ndarray):
        results = self.model.predict(im0, verbose=False, device=self.device)

        if not results or results[0].boxes is None:
            return []

        boxes = results[0].boxes.xyxy.cpu().numpy()

        if self.debug:
            print("[DEBUG] detect_plates -> num boxes:", len(boxes))
            print("[DEBUG] detect_plates -> boxes:", boxes)

        return boxes

    # -------------------------
    # OCR
    # -------------------------
    def extract_text(self, im0: np.ndarray, bbox: np.ndarray):
        h, w = im0.shape[:2]

        x1, y1, x2, y2 = np.round(bbox).astype(int)

        # clamp
        x1 = max(0, min(x1, w))
        x2 = max(0, min(x2, w))
        y1 = max(0, min(y1, h))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            return ""

        roi = im0[y1:y2, x1:x2]

        if roi.size == 0:
            return ""

        if self.debug:
            print(f"[DEBUG] extract_text -> bbox: {bbox}")

        # OCR
        results = self.reader.run(roi)
        if not results:
            if self.debug:
                print("[DEBUG] extract_text -> no OCR result")
            return ""

        pred = results[0]

        if hasattr(pred, "text"):
            text = pred.text
        elif hasattr(pred, "plate"):
            text = pred.plate
        elif hasattr(pred, "chars"):
            text = "".join(pred.chars) if isinstance(pred.chars, list) else pred.chars
        else:
            text = str(pred)

        if self.debug:
            print("[DEBUG] extract_text -> text:", text)

        return text

    # -------------------------
    # PIPELINE
    # -------------------------
    def infer_image_array(self, im0: np.ndarray):
        boxes = self.detect_plates(im0)

        records = []
        vis = im0.copy()

        for bbox in boxes:
            text = self.extract_text(im0, bbox)

            if not text:
                continue

            x1, y1, x2, y2 = np.round(bbox).astype(int)

            records.append({
                "text": text,
                "bbox": [int(x1), int(y1), int(x2), int(y2)]
            })

            if self.debug:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if self.debug:
            print("[DEBUG] infer_image_array -> records:", records)
            cv2.imshow("ANPR Debug", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return records