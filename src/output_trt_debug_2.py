'''
Author: wuyao 1955416359@qq.com
Date: 2024-03-13 16:35:09
LastEditors: wuyao 1955416359@qq.com
LastEditTime: 2024-03-15 18:13:46
FilePath: /yolo_meter/src/output_trt.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''


import os
import cv2
import time
import json
import queue
import torch
import threading
import traceback
from queue import Queue
import matplotlib.pyplot as plt
from angle_trt import Find_Angles
from angle_trt import get_value
from infer import Find_Meters
from infer import Find_Number
from videocapture import VideoCapture
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
                # print("第一个非0刻度: {} ".format(number))
                # print("当前读数: {:.3f}".format(value))
                # print("std_point", std_point)
                # print("pointer_lines", (pointer_line))
                # print("执行时间: {:.2f} ms".format(execution_time))
                # print('---------------------------------------------------------------------')
                
                
                return cropped_image, value
            else: return None, None
        except Exception as e:
            print("An exception occurred:", e)
            # raise ValueError("Something went wrong during inference with FA.infer") from e
        except KeyboardInterrupt:
            cap.terminate()

    



if __name__ == "__main__":  

    # cap = VideoCapture(0)
    img_path = '/home/rqh/yolo_meter/demo1'
    image_list=os.listdir(img_path)
    sorted_image_list = sorted(image_list)
    infer = INFER()
    count = 0
    val_num = len(sorted_image_list)
    execution_times = []
    first_iteration = True
    values = []
    image_names = []

    for img_name in sorted_image_list: 
    # while True:
        # img = cap.read()
        img = cv2.imread(os.path.join(img_path, img_name))
        # cv2.imshow("img.jpg", img)
        start_time = time.time()
        
        cropped_image, value = infer.results(img)
        values.append(value)
        image_names.append(img_name) 

        end_time = time.time()
        execution_time = (end_time - start_time)*1000
        # execution_times.append(execution_time)

        print("执行时间: {:.2f} ms".format(execution_time))
        if not first_iteration:
            execution_times.append(execution_time)
        else:
            first_iteration = False
        if value == -1:
            save_path = "../result_new/"+img_name
            cv2.imwrite(save_path, cropped_image)
        else :
            save_path = "../result_right/"+img_name
            cv2.imwrite(save_path, cropped_image)
        # if value != None:
        #     h, w, _ = cropped_image.shape
        #     img_h, img_w, _ = img.shape
        #     offset_x = img_w - w
        #     offset_y = 0
        #     offset_x = max(offset_x, 0)
        #     offset_y = max(offset_y, 0)
        #     img_resized = cv2.resize(img, (32*4, 24*4))
        #     img_cropped = cropped_image.copy()

        #     # 修改偏移量以将小图片放置在右上角
        #     offset_x_new = img_w - (32*4)
        #     offset_y_new = 0

        #     # 调整小图片位置
        #     img_cropped[offset_y_new:offset_y_new+24*4, offset_x_new:offset_x_new+32*4] = img_resized
        #     save_path = "../result_new/"+img_name
        #     cv2.imwrite(save_path, cropped_image)
        #     # cv2.imshow("infer", img_cropped)
        # else:
        #     count += 1
        #     cv2.imshow("infer", img)
    average_execution_time = sum(execution_times) / len(execution_times)
    print("平均执行时间: {:.2f} ms".format(average_execution_time))

    # print("val_num:", val_num)
    # print("count:", count)
    # detet_num = (1-(count/val_num))*100
    # print("能够正确识别的图片（%）: {:.2f}%".format(detet_num))
        # cv2.waitKey(1)
    max_execution_time = max(execution_times)
    min_execution_time = min(execution_times)
    print("最大执行时间: {:.2f} ms".format(max_execution_time))
    print("最小执行时间: {:.2f} ms".format(min_execution_time))


    # Save values to a txt file
    data_dict = dict(zip(image_names, values))

    with open('log.json', 'w') as f:
        json.dump(data_dict, f)


    # plt.figure(figsize=(10, 6))
    # plt.plot(execution_times, color='g', marker='o', linestyle='-')
    # plt.axhline(y=average_execution_time, color='r', linestyle='--', label='average_execution_time')
    # plt.text(len(execution_times) * 0.8, average_execution_time * 1.05, 
    #          'average_execution_time: {:.2f} ms'.format(average_execution_time), 
    #          color='r',fontsize=12)

    # plt.xlabel('Image Index', fontsize=12)
    # plt.ylabel('Execution Time (milliseconds)', fontsize=12)
    # plt.title('Execution Time for Each Image', fontsize=14)
    # plt.grid(True)
    # plt.tight_layout()
    # plt.legend()

    # plt.show()













