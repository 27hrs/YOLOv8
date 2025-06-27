from ultralytics import YOLO
import cv2
cap = cv2.VideoCapture(0)
model = YOLO('./best.pt')

# 遍历视频帧
while cap.isOpened():
    # 从视频中读取一帧
    success, frame = cap.read()

    if success:
        # 在该帧上运行YOLOv8推理
        results = model(frame)

        for result in results:
            print(f"检测到{result.boxes}个目标")
            boxes = result.boxes
            for box in boxes:
                confidence = box.conf.item()
                class_id = box.cls.item()
                class_name = result.names[class_id]
                print(f"类别: {class_name}({class_id}), 置信度: {confidence}")
        # 在帧上可视化结果
        annotated_frame = results[0].plot()
        # 显示带注释的帧
        cv2.imshow("YOLOv8推理", annotated_frame)
        # 如果按下'q'则中断循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果视频结束则中断循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()

# cs('rtsp://127.0.0.1:554/live',show=True,save=True)