import easyocr
from deep_translator import GoogleTranslator

reader = easyocr.Reader(['ar', 'en'], gpu=True)
translator = GoogleTranslator(source='en', target='ar')
translated_cache = {}

def get_arabic(label_en):
    if label_en in translated_cache:
        return translated_cache[label_en]
    try:
        translated_text = translator.translate(label_en)
        translated_cache[label_en] = translated_text
        return translated_text
    except:
        return label_en

def scan_full_frame(frame_copy, send_alert_callback):
    try:
        ocr_res = reader.readtext(frame_copy)
        if ocr_res:
            detected_texts = [res[1] for res in ocr_res[:2]]
            full_text = " و ".join(detected_texts)
            send_alert_callback(f"نص يقول: {full_text}")
    except Exception as e:
        print(f"خطأ في الـ OCR الشامل: {e}")

def read_roi_text(roi):
    if roi.size > 0:
        try:
            ocr_res = reader.readtext(roi)
            if ocr_res:
                return " ومكتبه عليه " + ocr_res[0][1]
        except:
            pass
    return ""