from ultralytics import YOLO
import torch

DATA_YAML = "datasets/obstacles_dataset/output/data.yaml"

def main():

     
    model = YOLO("yolov8n.pt")   
    
    results = model.train(
        data=DATA_YAML,
        epochs=300,        
        imgsz=768,         
        batch=16,           
        device=0,
        workers=2,
        pretrained=True,
        cache=False,
        patience=50,       
        project= "runs/",
        name="obstacle_train_v1", 
        verbose=True
    )

    print("\nTraining finished.")
    print(results)

if __name__ == "__main__":
    main()