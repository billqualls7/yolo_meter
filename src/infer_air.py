'''
Author: wuyao 1955416359@qq.com
Date: 2024-03-20 15:55:53
LastEditors: wuyao 1955416359@qq.com
LastEditTime: 2024-03-23 19:51:50
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
from math import sqrt, atan2, cos, sin
from infer import Find_Meters


class Colormap:
    # 定义类属性，每个属性对应一个BGR颜色值
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    yellow = (0, 255, 255)
    purple = (128, 0, 128)
    cyan = (255, 255, 0)
    magenta = (255, 0, 255)
    orange = (0, 165, 255)
    black = (0, 0, 0)
    white = (255, 255, 255)
    brown = (165, 42, 42)
    pink = (255, 192, 203)
    lime = (50, 205, 50)
    skyblue = (135, 206, 235)
    gray = (128, 128, 128)

    def __init__(self):
        # 这里可以添加初始化代码，如果需要的话
        pass


class Find_Air(Find_Meters):
    def __init__(self, model):
        super().__init__(model) 
        self.categories = ['air']


class Find_AirKnob():
    """
    Find air conditioner knob
    """
    def __init__(self, model) :
        self.categories = ['air_dail_10', 'air_dail_15', 'knob']
        # self.cam = VideoCapture(0)
        self.model = YOLO(task="segment", model= model)

        self.conf = 0.3
        self.iou = 0.5
        self.colormap = Colormap()
        



    def check_result(self,air_map):
        if len(air_map) != 3:
            return False
        if all(category in air_map for category in self.categories):
            # print("1st infer ok")
            return True

        else:
            missing_categories = [category for category in self.categories if category not in air_map]
            print(f"can not find：{missing_categories}")
            return False

    def midperpendicular(self,A,B):
        Mx = (A[0] + B[0]) // 2
        My = (A[1] + B[1]) // 2

        # 如果线段是垂直的，我们不能直接计算斜率，需要特殊处理
        if A[0] == B[0]:
            slope_AB = float('inf')  # 斜率无穷大，线段是垂直的
        else:
            slope_AB = (B[1] - A[1]) / (B[0] - A[0])

        # 计算中垂线的斜率
        if slope_AB == float('inf'):
            # 如果线段是垂直的，中垂线的斜率是0（水平线）
            slope_perpendicular = 0
        else:
            slope_perpendicular = -1 / slope_AB

        # 计算中垂线上的偏移量
        # 这里我们使用线段的一半长度作为中垂线的长度
        distance = sqrt((B[0] - A[0]) ** 2 + (B[1] - A[1]) ** 2) / 2

        # 计算中垂线的端点
        # 根据中垂线的斜率和中点来确定端点
        if slope_AB == float('inf'):
            # 如果线段是垂直的，中垂线是水平的，所以端点的x坐标与中点相同
            point1 = ((Mx - distance), (My))
            point2 = ((Mx + distance), (My))
        else:
            # 如果线段不是垂直的，我们可以使用斜率和中点来计算端点
            # 这里我们使用反正切函数来计算角度，然后根据角度计算端点的x坐标
            angle = atan2(slope_perpendicular, 1)  # 中垂线的角度
            dx = distance * cos(angle)  # x方向的偏移量
            dy = distance * sin(angle)  # y方向的偏移量
            point1 = ((Mx - dx), (My - dy))
            point2 = ((Mx + dx), (My + dy))





        return point1, point2


    def find_intersection(self, point1, point2, point3, point4):
        # 计算两条直线的斜率
        m1 = (point2[1] - point1[1]) / (point2[0] - point1[0])
        m2 = (point4[1] - point3[1]) / (point4[0] - point3[0])
        
        # 计算两条直线的截距
        b1 = point1[1] - m1 * point1[0]
        b2 = point3[1] - m2 * point3[0]
        
        # 检查斜率是否相同（即直线是否平行）
        if m1 == m2:
            # 如果直线平行，没有交点
            return None
        
        # 解方程找到x的值
        x = (b2 - b1) / (m1 - m2)
        
        # 使用x的值找到y的值
        y = m1 * x + b1
        
        return ((x), (y))


    def clc_value(self, A, B, C, O):
        std = 45
        A, B, C, O = map(np.array, (A, B, C, O))
        vector_AO = O - A
        vector_BO = O - B
        vector_CO = O - C

        dot_product_std = np.dot(vector_AO, vector_BO)
        dot_product_pre = np.dot(vector_AO, vector_CO)
        norms = np.array([np.linalg.norm(vector_AO), np.linalg.norm(vector_BO), np.linalg.norm(vector_CO)])


        cos_theta_std = dot_product_std / (norms[0] * norms[1])
        cos_theta_pre = dot_product_pre / (norms[0] * norms[2])

        theta_radians_std = np.arccos(cos_theta_std)
        theta_radians_pre = np.arccos(cos_theta_pre)
        theta_degrees_std = np.degrees(theta_radians_std)
        theta_degrees_pre = np.degrees(theta_radians_pre)

        k = theta_degrees_std/std
        print(k)
        value = ((theta_degrees_pre / theta_degrees_std)/(k) * 5 + 10)
        value = round(value, 3)

        print(theta_degrees_std)
        print(theta_degrees_pre)
        print(value)








    def infer(self,img):
        h, w = img.shape[:2]
        results = self.model.predict(img, save=False, 
                        imgsz=640, conf=self.conf,
                        iou = self.iou, 
                        visualize=False,
                        verbose = True,
                        stream=False
                        )
        

        air_map = {}
        for result in results:
            boxes = result.boxes.data.cpu().numpy() 
            for i in range(len(boxes)):
                x1, y1, x2, y2, score, cls = boxes[i]
                cls = self.categories[int(cls)]
                # if cls =='air_dail_10':

                mask = result.masks[i].xy[0]
                center_x = (np.mean(mask[:, 0]))
                center_y = (np.mean(mask[:, 1]))
                cv2.circle(img, (int(center_x), int(center_y)), radius=5, color=self.colormap.red, thickness=-1)
                air_map[cls] = (center_x, center_y)


        # print(air_map)
        # print(len(air_map))
        if self.check_result(air_map):
            scale_0 = air_map['air_dail_10']
            scale_1 = air_map['air_dail_15']
            knob_p = air_map['knob']
            y_0 = scale_0[1]
            p1 = (0, int(y_0))
            p2 = (w, int(y_0))
            p3, p4 = self.midperpendicular(scale_0, scale_1)
            (knob_x, knob_y) = self.find_intersection(p1, p2, p3, p4)
            knob_center = (knob_x, knob_y)
            # print(knob_center)
            # knob_center = (w//2, h//2)

            value = self.clc_value(scale_0, scale_1, knob_p, knob_center)
            

            # cv2.circle(img, (w//2, h//2), radius=5, color=(255, 0, 0), thickness=-1)
            # cv2.circle(img, (int(knob_x), int(knob_y)), radius=5, color=self.colormap.orange, thickness=-1)
            # cv2.line(img, (p1), (p2), color=(0, 255, 0), thickness=2)  
            # cv2.line(img, (p3), (p4), color=(0, 255, 0), thickness=2)
            # cv2.line(img, int(scale_0), int(scale_1), color=(0, 255, 0), thickness=2)
            return value
        else: return -1




















if __name__ == "__main__":
    
    model_1 = '/home/rqh/yolo_model/air_det_n.engine'
    model_2 = '/home/rqh/yolo_model/air_seg_m.engine'
    
    FAir = Find_Air(model_1)
    FAir_knob = Find_AirKnob(model_2)
    

    # for i in range(10):
    img = cv2.imread('/home/rqh/yolo_meter/airknob/frame_0.jpg')
    cls, cropped_image = FAir.infer_trt(img)
    FAir_knob.infer(cropped_image)
    cv2.imwrite('../airknob/infer_air.jpg', cropped_image)
