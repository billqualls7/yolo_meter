'''
Author: wuyao 1955416359@qq.com
Date: 2024-03-08 10:13:20
LastEditors: wuyao 1955416359@qq.com
LastEditTime: 2024-03-08 11:39:07
FilePath: /yolo_meter/src/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import os
import cv2
import time
from angle import Find_Angles
from angle import get_value
from infer import Find_Meters
from infer import Find_Number













if __name__ == "__main__":  

    img = cv2.imread('/home/rqh/Detect-and-read-meters/demo1/1032.jpg')
    img_path = '/home/rqh/yolo_meter/demo1'
    image_list=os.listdir(img_path)
    sorted_image_list = sorted(image_list)
    model_1 = '/home/rqh/yolo_model/meter.pt'
    model_2 = '/home/rqh/yolo_model/num.pt'
    model_3 = '/home/rqh/yolo_model/pointer.pt'
    
    

    FM = Find_Meters(model_1)
    FN = Find_Number(model_2)
    FA = Find_Angles(model_3)
   

    count = 0
    val_num = len(sorted_image_list)
    for img_name in sorted_image_list:
        try:
            
            img = cv2.imread(os.path.join(img_path, img_name))
            print("infer...", img_name)
            start_time = time.time()

            FM.infer(img)
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
            save_path = "../result/"+img_name
            cv2.imwrite(save_path, cropped_image)  

            end_time = time.time()
            execution_time = (end_time - start_time)*1000
            print("std_point", std_point)
            print("pointer_lines", (pointer_line))
            print("执行时间: {:.2f} ms".format(execution_time))
            print('---------------------------------------------------------------------')

        except: 
            print('*******************')
            print(img_name)
            print('*******************')

            count += 1
    print("val_num:", val_num)
    print("count:", count)
    detet_num = (1-(count/val_num))*100
    print("能够正确识别的图片（%）: {:.2f}%".format(detet_num))







