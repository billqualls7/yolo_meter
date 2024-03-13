'''
Author: wuyao 1955416359@qq.com
Date: 2024-03-13 16:35:09
LastEditors: wuyao 1955416359@qq.com
LastEditTime: 2024-03-13 16:54:15
FilePath: /yolo_meter/src/output_trt.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''


import os
import cv2
import time
import queue
import torch
import threading
import traceback
from queue import Queue
from src.angle_trt import Find_Angles
from src.angle import get_value
from src.infer import Find_Meters
from src.infer import Find_Number
from src.videocapture import VideoCapture
from concurrent.futures import ThreadPoolExecutor

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class INFER():
    def __init__(self):
        model_1 = '/home/rqh/yolo_model/meter.engine'
        model_2 = '/home/rqh/yolo_model/num.engine'
        model_3 = '/home/rqh/yolo_model/pointer.engine'
        self.FA = Find_Angles(model_3)
        self.FM = Find_Meters(model_1)
        self.FN = Find_Number(model_2)
        print("Model is loading...")
    

    def results(self, img):

        try:  
            start_time = time.time()

            meter_cls, cropped_image = self.FM.infer_trt(img)
            # print(cropped_image.shape)

            if meter_cls == 'pointer':
                std_point, pointer_line= self.FA.infer(cropped_image)
                number, *_ = self.FN.infer(cropped_image)
                value = get_value(cropped_image, std_point, pointer_line, number)
                
                font = cv2.FONT_HERSHEY_SIMPLEX
                cropped_image = cv2.putText(cropped_image, str(value), (30, 30), font, 1.2, (255, 0,255), 2)
                # cv2.line(cropped_image, pointer_line[0], pointer_line[1], (0, 255, 0), 2)
                # cv2.imwrite(save_path, cropped_image)  
                end_time = time.time()
                execution_time = (end_time - start_time)*1000
                print("第一个非0刻度: {} ".format(number))
                print("当前读数: {:.3f}".format(value))
                print("std_point", std_point)
                print("pointer_lines", (pointer_line))
                print("执行时间: {:.2f} ms".format(execution_time))
                print('---------------------------------------------------------------------')
                
                
                return cropped_image, value
            else: return None, None
        except Exception as e:
            print("An exception occurred:", e)
            # raise ValueError("Something went wrong during inference with FA.infer") from e
        except KeyboardInterrupt:
            cap.terminate()

    



if __name__ == "__main__":  

    cap = VideoCapture(0)

    infer = INFER()
    while True:
        img = cap.read()
        cv2.imshow("img.jpg", img)
        cropped_image, value = infer.results(img)
        if value != None:
            cv2.imshow("infer", cropped_image)
        cv2.waitKey(1)














