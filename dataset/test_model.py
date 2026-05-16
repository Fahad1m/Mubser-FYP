from ultralytics import YOLO
import torch

MODEL_PATH = "obstacles.pt"
DATA_YAML = "datasets/obstacles_dataset/output/data.yaml"

def main():


    model = YOLO(MODEL_PATH)

    metrics = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        batch=8,
        device=0,
        conf=0.25,
        iou=0.6,
        project="runs",
        name="obstacle_test_1",
        plots=True,
        verbose=True
    )

    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map50_95 = metrics.box.map

    if (precision + recall) > 0:
        f1_score = 2 * (precision * recall) / (precision + recall)
    else:
        f1_score = 0.0

    print("\n===== FINAL TEST RESULTS =====")
    print(f"Precision:   {precision:.4f}  ({precision*100:.2f}%)")
    print(f"Recall:      {recall:.4f}  ({recall*100:.2f}%)")
    print(f"F1-score:    {f1_score:.4f}  ({f1_score*100:.2f}%)")
    print(f"mAP@0.5:     {map50:.4f}  ({map50*100:.2f}%)")
    print(f"mAP@0.5:0.95 {map50_95:.4f}  ({map50_95*100:.2f}%)")

if __name__ == "__main__":
    main()