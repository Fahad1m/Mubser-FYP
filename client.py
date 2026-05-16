import cv2
import zmq
import threading
import os
import time  
from gtts import gTTS
from picamera2 import Picamera2

SERVER_IP ='xxx.xxx.xxx.xxx'  

print("⏳ Connecting to the server . . .")
context = zmq.Context()

video_sender = context.socket(zmq.PUB)
video_sender.connect(f'tcp://{SERVER_IP}:5555')

alert_receiver = context.socket(zmq.SUB)
alert_receiver.setsockopt_string(zmq.SUBSCRIBE, "")
alert_receiver.connect(f'tcp://{SERVER_IP}:5556')

print("✅ Connected")

def play_audio(filename):
    try:
        os.system(f"mpg321 -q {filename}")
        os.remove(filename)
    except: pass

def receive_alerts():
    while True:
        try:
            alert_msg = alert_receiver.recv_string()
            print(f"🔊 تنبيه: {alert_msg}")
            tts = gTTS(text=alert_msg, lang='ar')
            tts.save("alert.mp3")
            play_audio("alert.mp3")
        except: pass

threading.Thread(target=receive_alerts, daemon=True).start()


picam2 = Picamera2()
config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
picam2.configure(config)

try: 
    picam2.set_controls({"AfMode": 2}) 
except: pass

picam2.start()
print("🚀 Camera is ready (Streaming to the server) . .  .")

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]

last_send_time = 0

try:
    while True:
        frame = picam2.capture_array()
        if frame is None: continue
        
        current_time = time.time()
        if current_time - last_send_time > 0.15:
            result, frame_encoded = cv2.imencode('.jpg', frame, encode_param)
            video_sender.send(frame_encoded.tobytes())
            last_send_time = current_time

except KeyboardInterrupt:
    pass
finally:
    picam2.stop()
    picam2.close()
    video_sender.close()
    alert_receiver.close()
    context.term()
