'''
Author: wuyao 1955416359@qq.com
Date: 2024-03-08 16:54:49
LastEditors: wuyao 1955416359@qq.com
LastEditTime: 2024-03-08 17:19:20
FilePath: /yolo_meter/src/main _cam.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''


import os
import cv2
import time
from angle import Find_Angles
from angle import get_value
from infer import Find_Meters
from infer import Find_Number
from videocapture import VideoCapture












if __name__ == "__main__":  

    cap = VideoCapture(1)

    model_1 = '/home/rqh/yolo_model/meter.pt'
    model_2 = '/home/rqh/yolo_model/num.pt'
    model_3 = '/home/rqh/yolo_model/pointer.pt'
    
    

    FM = Find_Meters(model_1)
    FN = Find_Number(model_2)
    FA = Find_Angles(model_3)
   

    while True:
        img = cap.read()
        # try:
            
        cv2.imshow("img", img)
        print("infer...")
        start_time = time.time()

        FM.infer(img)
        try:
            if not FM.image_queue.empty():
                cropped_image = FM.image_queue.get()

                # print(img.shape)
                # print(cropped_image.shape)
                number, *_ = FN.infer(cropped_image)
                std_point, pointer_line = FA.key_point(cropped_image)

                value = get_value(cropped_image, std_point, pointer_line, number)
                # print(value)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cropped_image = cv2.putText(cropped_image, str(value), (30, 30), font, 1.2, (255, 0,255), 2)
                cv2.line(cropped_image, pointer_line[0], pointer_line[1], (0, 255, 0), 2)
                # cv2.imwrite(save_path, cropped_image)  

                end_time = time.time()
                execution_time = (end_time - start_time)*1000
                print("std_point", std_point)
                print("pointer_lines", (pointer_line))
                print("执行时间: {:.2f} ms".format(execution_time))
                print('---------------------------------------------------------------------')
                
                cv2.imshow("infer", cropped_image)
            else: pass
        except Exception as e:
            print("An exception occurred:", e)
        if chr(cv2.waitKey(1)&255) == 'q':  # 按 q 退出
            cap.terminate()
            break
    cap.terminate()
        # except: 
        #     print('*******************')









