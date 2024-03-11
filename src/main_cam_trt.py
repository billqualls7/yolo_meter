'''
Author: wuyao 1955416359@qq.com
Date: 2024-03-08 16:54:49
LastEditors: wuyao 1955416359@qq.com
LastEditTime: 2024-03-11 13:06:52
FilePath: /yolo_meter/src/main _cam.py
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
from angle_trt import Find_Angles
from angle import get_value
from infer import Find_Meters
from infer import Find_Number
from videocapture import VideoCapture
from concurrent.futures import ThreadPoolExecutor

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


# model_1 = '/home/rqh/yolo_model/meter.pt'
# model_2 = '/home/rqh/yolo_model/num.pt'
# model_3 = '/home/rqh/yolo_model/pointer.pt'


# FA = Find_Angles(model_3)
# FM = Find_Meters(model_1)
# FN = Find_Number(model_2)
# cap = VideoCapture(0)

# def find_m(output_queue):
#     while True:
#         img = cap.read()
#         meter_cls, cropped_image = FM.infer_trt(img)
#         output_queue.put((meter_cls, cropped_image))

# def find_a(input_queue, output_queue):
#     while True:
#         meter_cls, cropped_image = input_queue.get()
#         if meter_cls == 'pointer':
#             std_point, pointer_line= FA.infer(cropped_image)
#             output_queue.put((std_point, pointer_line))


# def find_n(input_queue, output_queue):
#     while True:
#         meter_cls, cropped_image = input_queue.get()
#         if meter_cls == 'pointer':
#             number, *_ = FN.infer(cropped_image)
#             output_queue.put((number))

# Met_queue = queue.Queue()
# Num_queue = queue.Queue()
# Ang_queue = queue.Queue()


# thread1 = threading.Thread(target=find_m, args=(Met_queue,))
# thread2 = threading.Thread(target=find_a, args=(Met_queue, Ang_queue))
# thread3 = threading.Thread(target=find_n, args=(Met_queue, Num_queue))

# thread1.start()
# thread2.start()
# thread3.start()

# while True:
    
#     meter_cls, cropped_image = Met_queue.get()
#     if meter_cls == 'pointer':
#         std_point, pointer_line = Ang_queue.get()
#         number = Num_queue.get()
#         value = get_value(cropped_image, std_point, pointer_line, number)
#         cv2.imshow("infer", cropped_image)
#         cv2.waitKey(1)

if __name__ == "__main__":  

    cap = VideoCapture(0)

    model_1 = '/home/rqh/yolo_model/meter.engine'
    model_2 = '/home/rqh/yolo_model/num.engine'
    model_3 = '/home/rqh/yolo_model/pointer.engine'
    
    
    FA = Find_Angles(model_3)
    FM = Find_Meters(model_1)
    FN = Find_Number(model_2)
    
    print("infer...")

    while True:
        img = cap.read()
        # try:
            

        cv2.imshow("img", img)
        
        start_time = time.time()

        try:
            
            meter_cls, cropped_image = FM.infer_trt(img)
            # print(cropped_image.shape)

            if meter_cls == 'pointer':


                std_point, pointer_line= FA.infer(cropped_image)
                # raise ValueError("Something went wrong")
                
                number, *_ = FN.infer(cropped_image)
                

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
                
                cv2.imshow("infer", cropped_image)
        
        except Exception as e:
            print("An exception occurred:", e)
            # raise ValueError("Something went wrong during inference with FA.infer") from e

        if chr(cv2.waitKey(1)&255) == 'q':  # 按 q 退出
            cap.terminate()
            break











