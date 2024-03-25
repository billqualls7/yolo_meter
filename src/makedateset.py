'''
Author: wuyao 1955416359@qq.com
Date: 2024-03-20 12:31:35
LastEditors: wuyao 1955416359@qq.com
LastEditTime: 2024-03-25 15:42:48
FilePath: /rqh/dataset/makedateset.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''



import cv2
import os
from tqdm import tqdm

# 视频文件路径
video_path = '/home/rqh/dataset/v20.mp4'
# 保存帧的文件夹路径
output_folder = '/home/rqh/dataset/v20'
os.makedirs(output_folder, exist_ok=True)

# 打开视频文件
cap = cv2.VideoCapture(video_path)

# 获取视频总帧数
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 定义进度条
pbar = tqdm(total=total_frames, desc='Processing Frames')

# 循环读取视频帧
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 每四帧保存一次
    if frame_count % 4 == 0:
        frame_filename = f'{output_folder}/frame_{frame_count}.jpg'
        cv2.imwrite(frame_filename, frame)

    frame_count += 1
    pbar.update(1)

pbar.close()
cap.release()
cv2.destroyAllWindows()
