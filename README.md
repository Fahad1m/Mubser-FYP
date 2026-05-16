# 👁️ Mubser (مبصر) - Vision System for the Visually Impaired

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)
![Raspberry Pi](https://img.shields.io/badge/Raspberry%20Pi-Client-red.svg)
![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)

**Mubser** is a real-time, AI-powered visual assistance system designed to help visually impaired individuals navigate their surroundings safely. The system detects street obstacles, identifies general objects, and reads environmental text, providing immediate auditory feedback in Arabic.

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

## 👥 Meet the Team

This project was developed as a graduation project by a dedicated team of engineers/developers:

| Name | Role / Contribution | GitHub |
| :--- | :--- | :--- |
| **[Your Name]** | AI & System Architecture | [@YourGitHub](https://github.com/YourGitHub) |
| **[Colleague 1 Name]** | Object Detection & Datasets | [@Colleague1](https://github.com/Colleague1) |
| **[Colleague 2 Name]** | Raspberry Pi & Hardware | [@Colleague2](https://github.com/Colleague2) |
| **[Colleague 3 Name]** | OCR & Audio Integration | [@Colleague3](https://github.com/Colleague3) |

*(Note: Roles can be adjusted based on actual contributions).*

---

## 🚀 How to Run

### 1. Server-Side (PC/Laptop)
1. Install dependencies: `pip install ultralytics opencv-python easyocr deep-translator gTTS`
2. Place `best.pt` and `yolov8s.pt` in the root directory.
3. Run the server script:
   ```bash
   python server.py