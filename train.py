from ultralytics import YOLO

model = YOLO('./ultralytics/cfg/models/11/yolo11.yaml')   

if __name__ == '__main__':
    results = model.train(
        data='./data.yaml',  # 你的数据集配置文件路径（如果是相对路径）
        epochs=150,
        imgsz=640,
        batch=32,
        workers=16,  
        device=[0,1],
        optimizer='RAdam',
    )