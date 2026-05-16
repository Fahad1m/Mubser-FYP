from ultralytics import YOLO

model_general = YOLO('yolov8s.pt')

def run_general_detection(frame, conf=0.60):
    res = model_general(frame, stream=True, conf=conf, verbose=False)
    return res, model_general.names