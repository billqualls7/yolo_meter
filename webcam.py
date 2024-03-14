'''
Author: wuyao 1955416359@qq.com
Date: 2024-03-14 09:42:41
LastEditors: wuyao 1955416359@qq.com
LastEditTime: 2024-03-14 09:43:02
FilePath: /yolo_meter/webcam.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import numpy as np
import pyvirtualcam

# 打开摄像头设备
cap = cv2.VideoCapture(0)

# 创建虚拟摄像头
with pyvirtualcam.Camera(width=640, height=480, fps=30) as cam:
    while True:
        ret, frame = cap.read()  # 读取摄像头视频流
        
        # 在这里对frame进行任何处理或图像处理操作
        
        cam.send(frame)  # 将帧发送到虚拟摄像头
        
        # 显示原始视频流
        # cv2.imshow('Original', frame)
        
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

cap.release()
cv2.destroyAllWindows()
