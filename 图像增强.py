import cv2
import numpy as np
import random

def augment_data(image):
    # 随机水平翻转（50%概率）
    if random.random() > 0.5:
        image = cv2.flip(image, 1)  # 1表示水平翻转\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

    # 对比度与亮度调整
    alpha = random.uniform(0.8, 1.2)  # 对比度系数（0.8~1.2）
    beta = random.randint(-30, 30)    # 亮度偏移（-30~30）
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # 随机饱和度调整（HSV色彩空间）
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = hsv[:, :, 1] * random.uniform(0.7, 1.3)  # 饱和度缩放
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 高斯噪声（模拟低光照噪点）
    if random.random() > 0.6:
        noise = np.random.normal(0, 15, image.shape).astype(np.uint8)
        image = cv2.add(image, noise)

    return image