'''
Author: wuyao 1955416359@qq.com
Date: 2024-03-08 15:25:22
LastEditors: wuyao 1955416359@qq.com
LastEditTime: 2024-03-08 15:25:59
FilePath: /yolo_meter/src/split_dataset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import shutil

def split_dataset(input_folder, output_folders):
    # 创建输出文件夹
    for folder in output_folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
    # 获取文件列表并按文件名排序
    files = sorted(os.listdir(input_folder))
    
    # 计算每个文件夹应该包含的文件数量
    files_per_folder = len(files) // len(output_folders)
    
    # 循环处理每个输出文件夹
    for i, folder in enumerate(output_folders):
        # 计算当前文件夹应该包含的文件索引范围
        start_index = i * files_per_folder
        end_index = (i + 1) * files_per_folder
        
        # 获取当前文件夹应该包含的文件列表
        files_in_folder = files[start_index:end_index]
        
        # 将文件复制到当前文件夹
        for file in files_in_folder:
            src = os.path.join(input_folder, file)
            dst = os.path.join(folder, file)
            shutil.copy(src, dst)

# 调用函数并传入输入文件夹路径和输出文件夹列表
input_folder = "/home/rqh/dataset/meter_cc"
output_folders = ["/home/rqh/dataset/meter_c1",
                  "/home/rqh/dataset/meter_c2",
                  "/home/rqh/dataset/meter_c3",
                  "/home/rqh/dataset/meter_c4",
                  "/home/rqh/dataset/meter_c5",
                  "/home/rqh/dataset/meter_c6"]
split_dataset(input_folder, output_folders)
