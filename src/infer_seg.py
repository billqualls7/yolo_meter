'''
Author: wuyao sss
Date: 2024-03-05 17:24:47
LastEditors: wuyao sss
LastEditTime: 2024-03-05 20:39:30
FilePath: /rqh/Detect-and-read-meters/yolov8/infer_seg.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import math
from videocapture import VideoCapture
import time
import threading
import queue
import os



class Find_Pointer():
    def __init__(self, model) :
        self.categories = ['pointer', 'digital']
        # self.cam = VideoCapture(0)
        self.model = YOLO(model)

        self.conf = 0.5
        self.iou = 0.7
    
    def infer(self,img):
        t0 = time.time()
        results = self.model.predict(img, save=False, 
                        imgsz=640, conf=self.conf,
                        iou = self.iou, 
                        visualize=False
                        )
        for result in results:
            masks = result.masks
            data = masks.data
            xy = masks.xy
            x = [point[0] for point in xy]
            y = [point[1] for point in xy]
            for i in range(len(x)):
                cv2.circle(img, (int(x[i]), int(y[i])), 5, (0, 0, 255), -1)
            cv2.imwrite("mask.jpg" , img)
            print((masks))
            for index, mask in enumerate(data):
                
                print(index)
                print("------------------")
                mask = mask.cpu().numpy() * 255
                
                


            # print(masks)
        #     boxes = result.boxes.data.cpu().numpy()




if __name__ == "__main__":  
    model = '/home/rqh/Detect-and-read-meters/yolov8/pointer.pt'
    img  = cv2.imread('/home/rqh/Detect-and-read-meters/demo1/8.jpg')
    FP = Find_Pointer(model)
    t0 = time.time()
    FP.infer(img)
    t1 = time.time()
    print(t1-t0)