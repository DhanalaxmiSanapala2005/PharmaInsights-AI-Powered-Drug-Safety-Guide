import pytesseract
import cv2
import numpy as np
from PIL import Image
import os

# Set path to your Tesseract installation
pytesseract.pytesseract.tesseract_cmd = r"C:\Users\sanapala Dhanalaxmi\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"

def extract_text(uploaded_file):
    """
    Extract text from uploaded image file
    uploaded_file: werkzeug FileStorage object
    """
    # Convert uploaded file to OpenCV image
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Optional: convert to gray for better OCR
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # OCR
    text = pytesseract.image_to_string(gray)

    return text.strip()
