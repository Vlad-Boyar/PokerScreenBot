# api_app/fastapi_app.py
# FastAPI app → receives image → processes via pipeline → returns processed image (or JSON)

import time
import logging
from io import BytesIO

import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Response
import uvicorn

from poker_pipeline.pipeline import process_screen
from poker_pipeline.draw_table import draw_table_from_json_pil  

# Global variable to store the last processed JSON (used by /last-json endpoint)
LAST_FINAL_JSON = None

# Initialize FastAPI app
app = FastAPI()

@app.post("/upload-screenshot")
async def upload_screenshot(file: UploadFile = File(...)):
    try:
        start_time = time.time()

        # Read image from UploadFile → convert to OpenCV format
        file_bytes = np.asarray(bytearray(await file.read()), dtype=np.uint8)
        image_cv = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        logging.info(f"📥 Received image: shape={image_cv.shape}")

        # Run pipeline → get final JSON
        final_json = process_screen(image_cv)
        logging.info("✅ Pipeline completed.")
        global LAST_FINAL_JSON
        LAST_FINAL_JSON = final_json

        # Generate visualization → as PIL.Image
        img_pil = draw_table_from_json_pil(final_json)

        # Convert PIL.Image to bytes
        buf = BytesIO()
        img_pil.save(buf, format="PNG")
        img_bytes = buf.getvalue()

        elapsed_time = time.time() - start_time
        logging.info(f"✅ Processing completed in {elapsed_time:.3f} sec.")

        # Return image to caller
        return Response(content=img_bytes, media_type="image/png")

    except ValueError as ve:
        logging.warning(f"⚠️ Invalid screen detected: {ve}")
        return Response(content=f"Invalid table: {ve}", media_type="text/plain", status_code=400)

    except Exception as e:
        logging.exception("❌ Internal error:")
        return Response(content=f"Internal error: {e}", media_type="text/plain", status_code=500)

@app.get("/last-json")
def get_last_json():
    global LAST_FINAL_JSON
    if LAST_FINAL_JSON is None:
        return {"error": "No JSON yet"}
    return LAST_FINAL_JSON

def run_fastapi():
    logging.info("🚀 Running FastAPI app...")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
