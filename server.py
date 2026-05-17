import cv2
import socket
import struct
import pickle
import threading
import time
from ultralytics import YOLO
from deep_translator import GoogleTranslator

from ocr import extract_text


HOST = '0.0.0.0'
PORT = 5000

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)
print("Server is Ready . .. ")

conn, addr = server_socket.accept()
print(f"✅ Connected with Raspbery{addr}")

model_obstacle = YOLO('best.pt')       
model_general = YOLO('yolov8s.pt')     

translator = GoogleTranslator(source='en', target='ar')

translated_cache = {}
BLACKLIST = ['person', 'airplane', 'traffic light', 'car', 'bus', 'truck']
is_busy = False 

def get_arabic(label_en): 
    if label_en in translated_cache: return translated_cache[label_en]
    try:
        translated_text = translator.translate(label_en)
        translated_cache[label_en] = translated_text
        return translated_text
    except:
        return label_en

def send_alert(message):
    try:
        conn.sendall(message.encode('utf-8'))
        print(f"📨 تم إرسال تنبيه: {message}")
    except Exception as e:
        print(f"خطأ في الإرسال: {e}")


def scan_full_frame_ocr(frame_copy):
    global is_busy
    is_busy = True 
    try:
        ocr_res = extract_text(frame_copy)
        if ocr_res:
            detected_texts = [res[1] for res in ocr_res[:2]]
            full_text = " و ".join(detected_texts)
            send_alert(f"نص يقول: {full_text}")
    finally:
        is_busy = False 

def process_roi_and_send(detected_obj):
    global is_busy
    is_busy = True 
    try:
        label_ar = get_arabic(detected_obj['label'])  
        text_on_obj = ""
        
        if detected_obj['roi'].size > 0:
            ocr_res = extract_text(detected_obj['roi'])
            if ocr_res:
                text_on_obj = " ومكتوب عليه " + ocr_res[0][1]

        if detected_obj['is_obstacle']:
            full_msg = f"انتبه عائق، {label_ar} {detected_obj['direction']}{text_on_obj}"
        else:
            full_msg = f"{label_ar} {detected_obj['direction']}{text_on_obj}"
            
        send_alert(full_msg)
    finally:
        is_busy = False 

data = b""
payload_size = struct.calcsize(">L")

last_obstacle_time = 0
last_general_time = 0
last_full_ocr_time = time.time()
frame_counter = 0

try:
    while True:
        while len(data) < payload_size:
            data += conn.recv(4096)
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]
        while len(data) < msg_size:
            data += conn.recv(4096)
        frame_data = data[:msg_size]
        data = data[msg_size:]
        
        frame_encoded = pickle.loads(frame_data)
        frame = cv2.imdecode(frame_encoded, cv2.IMREAD_COLOR)

        h, w, _ = frame.shape
        frame_counter += 1
        detected_obj = None
        max_area = 0

        is_obstacle_frame = (frame_counter % 2 == 0)
        
        if is_obstacle_frame:
            res = model_obstacle(frame, stream=True, conf=0.25, verbose=False)
            model_names = model_obstacle.names
        else:
            res = model_general(frame, stream=True, conf=0.25, verbose=False)
            model_names = model_general.names

        for r in res:
            for box in r.boxes:
                label_en = model_names[int(box.cls[0])]
                if is_obstacle_frame and label_en in BLACKLIST: continue 
                
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                area = (x2 - x1) * (y2 - y1)
                
                color = (0, 0, 255) if is_obstacle_frame else (255, 0, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                if area > max_area:
                    max_area = area
                    center_x = (x1 + x2) / 2
                    direction = "على يسارك" if center_x < w / 3 else "على يمينك" if center_x > 2 * w / 3 else "أمامك"
                    
                    detected_obj = {
                        'label': label_en, 
                        'confidence': conf,
                        'roi': frame[max(0, y1):y2, max(0, x1):x2], 
                        'direction': direction, 
                        'is_obstacle': is_obstacle_frame
                    }

        if detected_obj and not is_busy:
            current_time = time.time()
            if detected_obj['is_obstacle']:
                if (current_time - last_obstacle_time > 3.0):
                    last_obstacle_time = current_time
                    threading.Thread(target=process_roi_and_send, args=(detected_obj,), daemon=True).start()
            else:
                conf_score = detected_obj['confidence']
                delay_needed = 2.5 if conf_score >= 0.85 else 4.0 if conf_score >= 0.65 else 6.0
                if (current_time - last_general_time > delay_needed):
                    last_general_time = current_time
                    threading.Thread(target=process_roi_and_send, args=(detected_obj,), daemon=True).start()

        if (time.time() - last_full_ocr_time > 6) and not is_busy:
            last_full_ocr_time = time.time()
            threading.Thread(target=scan_full_frame_ocr, args=(frame.copy(),), daemon=True).start()

        cv2.imshow("Mubser Server - Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

except Exception as e:
    print(f"Diconnected! .{e}")
finally:
    conn.close()
    server_socket.close()
    cv2.destroyAllWindows()