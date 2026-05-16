import cv2
import zmq
import numpy as np
import threading
import time

import ocr
import obstacles
import object as obj_detect  

context = zmq.Context()

video_receiver = context.socket(zmq.SUB)
video_receiver.setsockopt_string(zmq.SUBSCRIBE, "")
video_receiver.setsockopt(zmq.CONFLATE, 1)  
video_receiver.bind('tcp://0.0.0.0:5555')

alert_sender = context.socket(zmq.PUB)
alert_sender.bind('tcp://0.0.0.0:5556')

print("✅ Server is Ready (waiting for the Camera). . .")

is_busy = False 

def send_alert(message):
    """إرسال التنبيه النصي للراسبيري باي"""
    try:
        alert_sender.send_string(message)
        print(f"📨 تم إرسال تنبيه: {message}")
    except: 
        pass

def background_full_ocr(frame_copy):
    global is_busy
    is_busy = True 
    try:
        ocr.scan_full_frame(frame_copy, send_alert)
    finally:
        is_busy = False 

def background_roi_process(detected_obj):
    global is_busy
    is_busy = True 
    try:
        label_ar = ocr.get_arabic(detected_obj['label'])  
        text_on_obj = ocr.read_roi_text(detected_obj['roi'])

        if detected_obj['is_obstacle']:
            full_msg = f"انتبه عائق، {label_ar} {detected_obj['direction']}{text_on_obj}"
        else:
            full_msg = f"{label_ar} {detected_obj['direction']}{text_on_obj}"
            
        send_alert(full_msg)
    finally:
        is_busy = False 

def main():
    global is_busy
    last_obstacle_time = 0
    last_general_time = 0
    last_full_ocr_time = time.time()
    frame_counter = 0

    try:
        while True:
            frame_data = video_receiver.recv()
            np_arr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            h, w, _ = frame.shape
            frame_counter += 1
            detected_obj = None
            max_area = 0

            is_obstacle_frame = (frame_counter % 2 == 0)
            
            if is_obstacle_frame:
                res, model_names, blacklist = obstacles.run_obstacle_detection(frame, conf=0.65)
            else:
                res, model_names = obj_detect.run_general_detection(frame, conf=0.60)
                blacklist = []

            for r in res:
                for box in r.boxes:
                    label_en = model_names[int(box.cls[0])]
                    if is_obstacle_frame and label_en in blacklist: continue 
                    
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
                        threading.Thread(target=background_roi_process, args=(detected_obj,), daemon=True).start()
                else:
                    conf_score = detected_obj['confidence']
                    delay_needed = 2.5 if conf_score >= 0.85 else 4.0 if conf_score >= 0.65 else 6.0
                    if (current_time - last_general_time > delay_needed):
                        last_general_time = current_time
                        threading.Thread(target=background_roi_process, args=(detected_obj,), daemon=True).start()

            if (time.time() - last_full_ocr_time > 6) and not is_busy:
                last_full_ocr_time = time.time()
                threading.Thread(target=background_full_ocr, args=(frame.copy(),), daemon=True).start()

            cv2.imshow("Mubser Vision System - Modular Live", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

    except KeyboardInterrupt:
        pass
    finally:
        video_receiver.close()
        alert_sender.close()
        context.term()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()