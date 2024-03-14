'''
Author: wuyao 1955416359@qq.com
Date: 2024-03-13 15:01:10
LastEditors: wuyao 1955416359@qq.com
LastEditTime: 2024-03-14 21:03:23
FilePath: /yolo_meter/web/Node_red.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
import json
import base64
import asyncio
import websockets
from src.output_trt import INFER
from src.videocapture import VideoCapture



class Node_RED():
    def __init__(self, url="ws://localhost:1880/ws/video"):
        self.websocket = None
        self.url = url

    async def connect(self):
        try:
            self.websocket = await websockets.connect(self.url)
            print("WebSocket connected")
        except Exception as e:
            print("Failed to connect:", e)

    async def send_img(self, img):
        if self.websocket is None:
            print("WebSocket connection not established")
            self.connect()
            return
        
        _, img_encoded = cv2.imencode('.jpg', img)
        img_base64 = base64.b64encode(img_encoded.tobytes()).decode('utf-8')


        
        if self.websocket:
            try:
                await self.websocket.send(img_base64)
                # print("Image sent")
            except Exception as e:
                print("Failed to send image:", e)



async def main():
    NR = Node_RED()
    # NR_orin = Node_RED(url="ws://localhost:1880/ws/video_orin")
    # img_paths = ["../result/1.jpg", "../result/2.jpg", "../result/3.jpg"]  # 外部图像路径列表，你可以根据需要修改
    cap = VideoCapture(0)
    infer = INFER()
    # 创建并启动一个事件循环
    await NR.connect()
    # await NR_orin.connect()
    while True:
        img = cap.read()
        # await NR_orin.send_img(img)
        if img is None:
            print("Failed to load image")
        else:
            cropped_image, value = infer.results(img) 
            if value != None:

                h, w, _ = cropped_image.shape
                img_h, img_w, _ = img.shape
                offset_x = img_w - w
                offset_y = 0
                offset_x = max(offset_x, 0)
                offset_y = max(offset_y, 0)
                img_resized = cv2.resize(img, (32*4, 24*4))
                img_cropped = cropped_image.copy()

                # 修改偏移量以将小图片放置在右上角
                offset_x_new = img_w - (32*4)
                offset_y_new = 0

                # 调整小图片位置
                
                img_cropped[offset_y_new:offset_y_new+24*4, offset_x_new:offset_x_new+32*4] = img_resized
                
                await NR.send_img(img_cropped)
                
            else: await NR.send_img(img)
            # await NR.send_img(img)  
        # await asyncio.sleep(1)  # 1秒延时


if __name__ == "__main__":

    asyncio.run(main())