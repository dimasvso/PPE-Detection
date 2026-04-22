from ultralytics import YOLO

model = YOLO('best.pt') 

if __name__ == '__main__':
    model.train(
        data='data.yaml',    
        epochs=50,           
        imgsz=640,           
        batch=16,            
        device=0,            
        project='safety_v2', 
        name='heavy_dataset',
        mosaic=1.0
    )