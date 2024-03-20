'''
Author: wuyao 1955416359@qq.com
Date: 2024-03-08 15:22:22
LastEditors: wuyao 1955416359@qq.com
LastEditTime: 2024-03-20 16:11:17
FilePath: /yolo_meter/src/rename.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import shutil

def rename_images(input_folder, output_folder):
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 获取输入文件夹中所有文件
    files = os.listdir(input_folder)
    
    # 设置一个计数器
    count = 1
    
    # 循环处理每个文件
    for file in files:
        # 检查文件是否为图片文件
        if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png") or file.endswith(".gif"):
            # 构建新的文件名
            new_name = "2024_" + str(count) + os.path.splitext(file)[1]
            
            # 构建完整的路径
            old_path = os.path.join(input_folder, file)
            new_path = os.path.join(output_folder, new_name)
            
            # 重命名并移动文件
            shutil.copy(old_path, new_path)
            
            # 更新计数器
            count += 1

# 调用函数并传入输入和输出文件夹路径
input_folder = "/home/rqh/yolo_meter/airknob"
output_folder = "/home/rqh/yolo_meter/airknob"
rename_images(input_folder, output_folder)
