import cv2
import easyocr

reader = easyocr.Reader(['ar', 'en'], gpu=True)

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    blurred = cv2.bilateralFilter(enhanced, 9, 75, 75)
    return blurred

def extract_text(img):
    processed_img = preprocess_for_ocr(img)
    return reader.readtext(processed_img)