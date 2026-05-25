import os
from ultralytics import YOLO

def run_manual_inference():
    model_path = r"D:\AI_Data\training_runs\Small_416_Augmented\weights\best.pt"

    test_images_dir = r"D:\AI_Data\test_normal"

    model = YOLO(model_path)

    results = model.predict(
        source=test_images_dir,
        save=True,       
        save_txt=True,     
        conf=0.25,          
        project=r"D:\AI_Data\inference_results_normal", 
        name="test_87_images"                    
    )

if __name__ == '__main__':
    run_manual_inference()