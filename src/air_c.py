'''
Author: wuyao 1955416359@qq.com
Date: 2024-03-20 21:42:08
LastEditors: wuyao 1955416359@qq.com
LastEditTime: 2024-03-20 22:03:50
FilePath: /yolo_meter/src/air_c.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import numpy as np

# 读取图片
image = cv2.imread('/home/rqh/yolo_meter/airknob/frame_0.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)





gray_blurred = cv2.GaussianBlur(gray, (9, 9), 0)
# 使用霍夫变换检测圆
circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=60,
                           param1=50, param2=50, minRadius=150, maxRadius=200)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # 画圆
        cv2.circle(image, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # 画圆心
        cv2.circle(image, (i[0], i[1]), 2, (0, 0, 255), 3)
        # 标注圆心坐标
        cv2.putText(image, f"({i[0]}, {i[1]})", (i[0] + 10, i[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# 显示结果
cv2.imwrite('../airknob/Circles.jpg', image)

