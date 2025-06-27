from ultralytics import YOLO
import yaml
with open('./hosts.yaml','r',encoding='utf-8') as f:
    result = yaml.load(f.read(),Loader=yaml.FullLoader)
#加载预训练模型
yolov8 = YOLO('./best.pt')#yolov8n.pt为官方提供的预训练模型
#检测目标
yolov8('./11111.jpg',show=True,save=True)
# yolov8('https://www.onlinemictest.com/zh/webcam-test/',show=True,save=True)

#show显示关键结果、save保存检查结果