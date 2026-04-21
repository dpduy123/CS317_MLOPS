from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from inference import ANPR

app = FastAPI()

model = ANPR("anpr_best.pt")


@app.post("/infer")
async def infer(file: UploadFile = File(...)):
    contents = await file.read()

    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    if img is None:
        return {"error": "Invalid image"}

    records = model.infer_image_array(img)

    return {
        "records": records
    }