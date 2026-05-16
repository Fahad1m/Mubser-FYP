# 👁️ Mubser (مبصر) - Vision System for the Visually Impaired

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-Client-red.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

**Mubser** is a real-time, AI-powered visual assistance system designed to help visually impaired individuals navigate their surroundings safely. The system detects street obstacles, identifies general objects, and reads environmental text, providing immediate auditory feedback in Arabic.

---

## 🎓 Project Overview
This project was developed and submitted as a **Final Year Project** to fulfill graduation requirements. It showcases the integration of computer vision and embedded systems to create an impactful assistive technology solution.

---

## ✨ Key Features
* **🚧 Real-Time Obstacle Detection:** Custom YOLOv8 model trained to detect navigational hazards like potholes, puddles, and roadblocks.
* **🚗 General Object Detection:** Identifies everyday objects (e.g., cars, pedestrians, chairs) using a secondary YOLO model.
* **📝 Optical Character Recognition (OCR):** Extracts text from the environment using EasyOCR.
* **🔊 Arabic Audio Feedback:** Converts warnings and extracted text into natural Arabic speech using `gTTS`.
* **⚡ Client-Server Architecture:** Offloads heavy AI computation from the edge device to a local server for zero-latency processing.

---

## 🏗️ System Architecture

To overcome the hardware limitations of embedded systems, Mubser utilizes a robust **Client-Server Architecture** via TCP Sockets:

1.  **The Client (Raspberry Pi):** Acts as the "Eyes and Ears." It captures video frames using `Picamera2`, compresses them, and streams them to the server over a local network. It simultaneously listens for incoming text alerts to synthesize into Arabic speech.
2.  **The Server (Local PC/Laptop):** Acts as the "Brain." It receives the video stream, runs the dual YOLO models and OCR concurrently, calculates the object's direction (Left, Right, Center), and sends concise text alerts back to the client.

---

## 🛠️ Technologies Used
* **Machine Learning:** Ultralytics YOLOv8
* **Computer Vision:** OpenCV
* **OCR & Translation:** EasyOCR, Deep Translator
* **Audio Processing:** gTTS, mpg321
* **Hardware:** Raspberry Pi (Camera Module 3)

---

## 👥 Team Members

| Name | GitHub |
| :--- | :--- |
| **[Fahad Al-Mutairi]** | [@Fahad1m](https://github.com/Fahad1m) |
| **[Mohammad Al-Neghimshi]** | [@muhammed-coder](https://github.com/muhammed-coder) |
| **[Waleed Al-Tuwaijri]** | [@WaleedTw](https://github.com/WaleedTw) |
| **[Waleed Al-Suwayyid]** | [@Lido-79](https://github.com/Lido-79) |
| **[Fahad Al-Fawzan]** | [@Fhod97](https://github.com/Fhod97) |

---

## 🚀 How to Run

### 1. Server-Side (PC/Laptop)
1. Install dependencies: `pip install ultralytics opencv-python easyocr deep-translator gTTS`
2. Place `obstacles.pt` and `yolov8s.pt` in the root directory.
3. Run the server script:
   ```bash
   python server.py
   ```

### 2. Client-Side (Raspberry Pi)
1. Ensure the Raspberry Pi is on the same local network (e.g., Mobile Hotspot) as the server.
2. Update the `SERVER_IP` variable in `client.py` to match the Server's IP address.
3. Run the client script:
   ```bash
   python client.py
   ```
