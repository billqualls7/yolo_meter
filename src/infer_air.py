'''
Author: wuyao 1955416359@qq.com
Date: 2024-03-20 15:55:53
LastEditors: wuyao 1955416359@qq.com
LastEditTime: 2024-03-20 16:16:17
FilePath: /yolo_meter/src/infer_air.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import math
import time
import threading
import queue
import os
from ultralytics.utils import ops


class Find_AirKnob():
    """
    Find air conditioner knob
    """
    def __init__(self, model) :
        self.categories = ['button', 'square_dail']
        # self.cam = VideoCapture(0)
        self.model = YOLO(model)

        self.conf = 0.3
        self.iou = 0.5

    def infer(self,img):

        results = self.model.predict(img, save=True, 
                        imgsz=640, conf=self.conf,
                        iou = self.iou, 
                        visualize=False,
                        verbose = True,
                        stream=True
                        )
        






if __name__ == "__main__":

    img = cv2.imread('/home/rqh/yolo_meter/airknob/2024_1.jpg')
    model = '/home/rqh/yolo_model/air.pt'
    FAir = Find_AirKnob(model)
    FAir.infer(img)

