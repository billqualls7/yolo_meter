'''
Author: wuyao sss
Date: 2024-02-22 17:11:14
LastEditors: wuyao 1955416359@qq.com
LastEditTime: 2024-03-08 17:08:03
FilePath: /rqh/YOLOv8/src/infer.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np
import math
from videocapture import VideoCapture
import time
import threading
import queue
import os


class Find_Meters:
    def __init__(self, model) :
        self.categories = ['pointer', 'digital']
        # self.cam = VideoCapture(0)
        self.model = YOLO(model)

        self.conf = 0.5
        self.iou = 0.7
        self.image_queue = queue.Queue()


    def outimg_queue(self, img, x1, y1, x2, y2):
        h, w = img.shape[:2]
        cropped_image = img[int(y1):int(y2), int(x1):int(x2)]
        cropped_image_resized = cv2.resize(cropped_image, (w, h))  # 假设 original_width 和 original_height 是之前图像的尺寸

        self.image_queue.put(cropped_image_resized)



    def infer(self,img):

        t0 = time.time()
        results = self.model.predict(img, save=False, 
                        imgsz=640, conf=self.conf,
                        iou = self.iou, 
                        visualize=False
                        )
        for result in results:
            boxes = result.boxes.data.cpu().numpy()  # Boxes object for bounding box outputs
            # data = boxes.data.cpu().numpy()

            for i in range(len(boxes)):
                # data = boxes[i]
                x1, y1, x2, y2, score, cls = boxes[i]
                # x1 = data[0]
                # y1 = data[1]
                # x2 = data[2]
                # y2 = data[3]
                # score = data[4]
                cls = self.categories[int(cls)]
                if cls == 'pointer':
                    threading.Thread(target=self.outimg_queue, args=(img, x1, y1, x2, y2)).start()
                
        t1 = time.time()
        # print((t1-t0)*1000)


class Find_Number(Find_Meters):
    def __init__(self, model):
        super().__init__(model) 
        self.categories = ['0.4', '2', '5']

    def infer(self, img):
        x1 = -1
        y1 = -1
        x2 = -1
        y2 = -1
        max_score = -1
        max_score_cls = None
        results = self.model.predict(img, save=False, 
                        imgsz=640, conf=self.conf,
                        iou = self.iou, 
                        visualize=False
                        )
        for result in results:
            boxes = result.boxes.data.cpu().numpy()
            
            for i in range(len(boxes)):
                x1, y1, x2, y2, score, cls = boxes[i]
                cls = self.categories[int(cls)]
                if score > max_score:
                    max_score = score
                    max_score_cls = cls
                
                threading.Thread(target=self.outimg_queue, args=(img, x1, y1, x2, y2)).start()
                # cls_list.append(cls)
                # cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                # cv2.imwrite('img.jpg',img)
        return max_score_cls, x1, y1, x2, y2
                
if __name__ == "__main__":  
    # FM = Find_Meters()
    # while True:
    #     FM.infer()
    #     if not FM.image_queue.empty():
    #         cropped_image = FM.image_queue.get()
    #         cv2.imshow('cropped_image', cropped_image)
    img_path = '/home/rqh/Detect-and-read-meters/demo1/'
    image_list=os.listdir(img_path)
    
    model = '/home/rqh/Detect-and-read-meters/yolov8/num.pt'
    FN = Find_Number(model)
    # img  = cv2.imread('/home/rqh/Detect-and-read-meters/demo1/8.jpg')
    # for img_name in image_list:
    #     if img_name.endswith('.jpg') or img_name.endswith('.png'):
    img = cv2.imread('/home/rqh/Detect-and-read-meters/demo1/1032.jpg')
    max_score_cls, *_ = FN.infer(img)
    print(max_score_cls)
    print(type(max_score_cls))
    # cv2.imwrite("../images/num.jpg",img)

    
  
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


# categories = ['pointer', 'digital']

# cam = VideoCapture(0)
# while True:
#     img = cam.read()
#     model = YOLO("dashboardbest.pt")
#     # img_path = '../img/76.jpg'
#     # img_path = '../demo/76.jpg'
#     # img = cv2.imread(img_path)
#     # results = model([img_path], stream=True)  # return a generator of Results objects
#     results = model.predict(img, save=False, 
#                             imgsz=640, conf=0.5, 
#                             visualize=False
#                             )
#     '''
#     data: tensor([[ 26.3736,  18.5083, 394.1399, 399.2077,   0.9480,   0.0000]], device='cuda:0')
#     分别代表xyxy conf cls 
#     '''
#     for result in results:
#         boxes = result.boxes.data.cpu().numpy()  # Boxes object for bounding box outputs
#         # data = boxes.data.cpu().numpy()

#         for i in range(len(boxes)):
#             data = boxes[i]
#             x1 = data[0]
#             y1 = data[1]
#             x2 = data[2]
#             y2 = data[3]
#             score = data[4]
#             cls = categories[int(data[5])]
#             cropped_image = img[int(y1):int(y2), int(x1):int(x2)]
#             cv2.imwrite("YOLOv8推理.jpg", cropped_image)
#         # if chr(cv2.waitKey(1)&255) == 'q':  # 按 q 退出
#         #     cam.terminate()
#         #     break

    
      

