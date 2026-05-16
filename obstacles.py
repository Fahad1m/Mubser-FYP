from ultralytics import YOLO

model_obstacle = YOLO('best.pt')

BLACKLIST = ['person', 'airplane', 'traffic light', 'car', 'bus', 'truck']

def run_obstacle_detection(frame, conf=0.65):
    res = model_obstacle(frame, stream=True, conf=conf, verbose=False)
    return res, model_obstacle.names, BLACKLIST