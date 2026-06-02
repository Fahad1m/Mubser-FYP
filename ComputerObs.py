import cv2
import time
import os
import threading
from ultralytics import YOLO
import easyocr
from deep_translator import GoogleTranslator
from gtts import gTTS
from playsound3 import playsound

model_obstacle = YOLO('best.pt')      
model_general = YOLO('yolov8n.pt')    

reader = easyocr.Reader(['ar', 'en'], gpu=True)
translator = GoogleTranslator(source='en', target='ar')
yolo_arabic_dict = {}
translated_cache = {}
BLACKLIST = ['person', 'airplane', 'traffic light', 'car', 'bus', 'truck']

def get_arabic(label_en, class_id):
    if class_id in yolo_arabic_dict: return yolo_arabic_dict[class_id]
    if label_en in translated_cache: return translated_cache[label_en]
    try:
        translated_text = translator.translate(label_en)
        translated_cache[label_en] = translated_text
        return translated_text
    except:
        return label_en

def speak_async(text):
    def run():
        try:
            tts = gTTS(text=text, lang='ar')
            tts.save("temp.mp3")
            playsound("temp.mp3")
            os.remove("temp.mp3")
        except: pass
    threading.Thread(target=run, daemon=True).start()

def scan_full_frame_ocr(frame_copy):
    ocr_res = reader.readtext(frame_copy)
    if ocr_res:

        detected_texts = [res[1] for res in ocr_res[:2]]
        full_text = " و ".join(detected_texts)
        speak_async(f"نص  يقول: {full_text}")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
last_yolo_time = 0
last_full_ocr_time = time.time()  
label_last_spoken = {}  
COOLDOWN_SECONDS = 10

while cap.isOpened():

    success, frame = cap.read()
    if not success: break
    h, w, _ = frame.shape
    res_general = model_general(frame, stream=True, conf=0.50, verbose=False)
    res_obstacle = model_obstacle(frame, stream=True, conf=0.70, verbose=False)
    current_best = None
    max_area = 0

    for r in res_general:
        for box in r.boxes:
            label_en = model_general.names[int(box.cls[0])]
            if time.time() - label_last_spoken.get(label_en, 0) < COOLDOWN_SECONDS:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

            if area > max_area:
                max_area = area
                center_x = (x1 + x2) / 2
                direction = "على يسارك" if center_x < w / 3 else "على يمينك" if center_x > 2 * w / 3 else "أمامك"
                current_best = {'label': label_en, 'class_id': int(box.cls[0]), 'roi': frame[max(0, y1):y2, max(0, x1):x2], 'direction': direction, 'is_obstacle': False}
    for r in res_obstacle:

        for box in r.boxes:
            label_en = model_obstacle.names[int(box.cls[0])]
            if label_en in BLACKLIST: continue
            if time.time() - label_last_spoken.get(label_en, 0) < COOLDOWN_SECONDS:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

            if area > max_area:
                max_area = area
                center_x = (x1 + x2) / 2
                direction = "على يسارك" if center_x < w / 3 else "على يمينك" if center_x > 2 * w / 3 else "أمامك"
                current_best = {'label': label_en, 'class_id': int(box.cls[0]), 'roi': frame[max(0, y1):y2, max(0, x1):x2], 'direction': direction, 'is_obstacle': True}

    if current_best and (time.time() - last_yolo_time > 4):

        label_ar = get_arabic(current_best['label'], current_best['class_id'])  
        text_on_obj = ""

        if current_best['roi'].size > 0:
            ocr_res = reader.readtext(current_best['roi'])
            if ocr_res:
                text_on_obj = " ومكتوب عليه " + ocr_res[0][1]

        if current_best['is_obstacle']:
            full_msg = f"انتبه عائق، {label_ar} {current_best['direction']}{text_on_obj}"
            
        else:
            full_msg = f"{label_ar} {current_best['direction']}{text_on_obj}"
        speak_async(full_msg)
        last_yolo_time = time.time()
        label_last_spoken[current_best['label']] = time.time()
    if time.time() - last_full_ocr_time > 4.5:

        threading.Thread(target=scan_full_frame_ocr, args=(frame.copy(),), daemon=True).start()

        last_full_ocr_time = time.time()
    cv2.imshow("Mubser Vision System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break
cap.release()
cv2.destroyAllWindows()
