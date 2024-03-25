'''
Author: wuyao 1955416359@qq.com
Date: 2024-03-20 15:55:53
LastEditors: wuyao 1955416359@qq.com
LastEditTime: 2024-03-25 20:55:16
FilePath: /yolo_meter/src/infer_air.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import numpy as np
from air_4c import Find_Air
from air_4c import Find_AirKnob


def resize_image(image, height=480, width=640):
    # 使用 cv2.resize 函数调整图像尺寸
    resized_image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return resized_image

def correct_image(image):
    # 灰度化
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 二值化
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        # 形态学开运算去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)        
        cv2.imwrite('../airknob/binary.jpg', closed_edges)
        # 轮廓检测
        contours, _ = cv2.findContours(closed_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 筛选最大轮廓
        max_contour = max(contours, key=cv2.contourArea)
        
        # 获取最小外接矩形
        rect = cv2.minAreaRect(max_contour)
        points = cv2.boxPoints(rect)
        points = np.int0(points)  # 转换为整数坐标
        
        # 定义矫正后的四顶点
        dst_points = np.array([
            [0, image.shape[0]],
            [0, 0],
            [image.shape[1], 0],
            [image.shape[1], image.shape[0]]
        ], dtype=np.float32)
        
        # 确保 points 和 dst_points 都是包含 4 个点的数组
        if len(points) == 4 and len(dst_points) == 4:
            # 计算透视变换矩阵
            M = cv2.getPerspectiveTransform(points, dst_points)
            # 应用透视变换
            warped = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]))
            return warped
        else:
            raise ValueError("The points arrays do not contain exactly 4 points each.")

if __name__ == "__main__":
    video_input_path = '/home/rqh/dataset/kongtiao.mp4'
    video_output_path = '../airknob/video_output.avi'
    model_1 = '/home/rqh/yolo_model/air_det_n.engine'
    model_2 = '/home/rqh/yolo_model/air_seg_4_m.engine'
    
    FAir = Find_Air(model_1)
    FAir_knob = Find_AirKnob(model_2)
    cap = cv2.VideoCapture(video_input_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 可以选择不同的编码器
    # out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))
    
    # test one pic-----------------------------------------------------------------
    img = cv2.imread('/home/rqh/yolo_meter/airknob/frame_420.jpg')

    cls, cropped_image, xyxy = FAir.infer_trt(img)
    # gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=100, maxRadius=200)
    # if circles is not None:
    #     circles = np.uint16(np.around(circles))
    #     circles = circles[0, :]

    #     # 找到距离图像中心最近的圆
    #     center = circles[:, 0:2]
    #     radius = circles[:, 2]
    #     min_radius = np.min(radius)
    #     min_center = center[radius == min_radius, :]

    #     # 计算图像中心点
    #     height, width = gray.shape
    #     center_image = (width / 2, height / 2)

    #     # 计算最近的圆的中心和半径
    #     closest_center = min_center[0]
    #     closest_radius = min_radius

    #     # 计算透视变换矩阵
    #     src_points = np.float32([closest_center, (width, closest_center), (closest_center, height)])
    #     dst_points = np.float32([center_image, (center_image[0] + closest_center[0] - center_image[0], center_image[1]), (center_image[0], center_image[1] + closest_center[1] - center_image[1])])
    #     M = cv2.getPerspectiveTransform(src_points, dst_points)

    #     # 应用透视变换来矫正图像
    #     warped_image = cv2.warpPerspective(cropped_image, M, (width, height))
    # # 保存矫正后的图像
    # # cv2.imwrite('../airknob/infer_air_edges.jpg', edges)
    # cv2.imwrite('../airknob/infer_air_cropped_image.jpg', cropped_image)
    # cv2.imwrite('../airknob/infer_air_warped_image.jpg', warped_image)
    value = FAir_knob.infer(cropped_image) 
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(cropped_image, str(value), (30, 30), font, 1.5, FAir_knob.colormap.white, 2)
    cv2.imwrite('../airknob/infer_air_4.jpg', cropped_image)
    # test one pic-----------------------------------------------------------------

    
    # for i in range(10):
#     while True:
#         ret, frame = cap.read() 
#         # frame = cv2.resize(frame, (640, 480))
#         if not ret:
#             break 
#         try:
#         # img = cv2.imread('/home/rqh/yolo_meter/airknob/frame_4.jpg')
#             cls, cropped_image = FAir.infer_trt(frame)
#             value = FAir_knob.infer(cropped_image)
#             # cv2.imwrite('../airknob/infer_air.jpg', cropped_image)
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             cv2.putText(cropped_image, str(value), (30, 30), font, 1.5, FAir_knob.colormap.white, 2)

#             out.write(cropped_image)
#         except: continue
# cap.release()
# out.release()
# cv2.destroyAllWindows()