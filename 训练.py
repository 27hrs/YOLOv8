from ultralytics import YOLO
if __name__ == '__main__':
    yolov8 = YOLO('./yolov8n.pt')

    yolov8.train(
        data='./fight.yaml',
        epochs=600,
        imgsz=640,
        batch=32,
        device='0'
    )










    print("训练完毕")