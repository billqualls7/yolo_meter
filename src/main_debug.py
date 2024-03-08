'''
Author: wuyao 1955416359@qq.com
Date: 2024-03-08 10:13:20
LastEditors: wuyao 1955416359@qq.com
LastEditTime: 2024-03-08 11:43:26
FilePath: /yolo_meter/src/main.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import os
import cv2
import time
from tqdm import tqdm 
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
    with open('debug_info.txt', 'w') as debug_file:

    # 获取图像路径
        img_path = '/home/rqh/yolo_meter/demo1'

        # 列出图像文件列表
        image_list = os.listdir(img_path)

        # 按照文件名的顺序对图像文件列表进行排序
        sorted_image_list = sorted(image_list)

        # 记录正确识别的图片数量和出错的图片数量
        val_num = len(sorted_image_list)
        count = 0

        # 逐个读取图像并进行处理
        for img_name in tqdm(sorted_image_list, desc='Processing images', unit='image'):
            try:
                img = cv2.imread(os.path.join(img_path, img_name))
                debug_file.write("infer... {}\n".format(img_name))
                start_time = time.time()

                # 进行处理...
                # 假设你的处理函数为FM.infer()
                FM.infer(img)
                cropped_image = FM.image_queue.get()

                number, *_ = FN.infer(cropped_image)
                std_point, pointer_line = FA.key_point(cropped_image)

                value = get_value(cropped_image, std_point, pointer_line, number)
                cropped_image = cv2.putText(cropped_image, str(value), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 255), 2)
                cv2.line(cropped_image, pointer_line[0], pointer_line[1], (0, 255, 0), 2)
                save_path = "../result/" + img_name
                cv2.imwrite(save_path, cropped_image)

                end_time = time.time()
                execution_time = (end_time - start_time) * 1000
                debug_file.write("std_point: {}\n".format(std_point))
                debug_file.write("pointer_lines: {}\n".format(pointer_line))
                debug_file.write("执行时间: {:.2f} ms\n".format(execution_time))
                debug_file.write('---------------------------------------------------------------------\n')

            except Exception as e:
                debug_file.write('*******************\n')
                debug_file.write("Error occurred in {}\n".format(img_name))
                debug_file.write("Error message: {}\n".format(str(e)))
                debug_file.write('*******************\n')
                count += 1

        debug_file.write("val_num: {}\n".format(val_num))
        debug_file.write("count: {}\n".format(count))
        detet_num = (1 - (count / val_num)) * 100
        debug_file.write("能够正确识别的图片（%）: {:.2f}%\n".format(detet_num))







