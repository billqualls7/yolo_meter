'''
Author: wuyao sss
Date: 2024-02-28 18:36:16
LastEditors: wuyao 1955416359@qq.com
LastEditTime: 2024-03-08 17:05:19
FilePath: /rqh/Detect-and-read-meters/videocapture.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import cv2
import queue
import threading
import time

# 自定义无缓存读视频类
class VideoCapture:
    """Customized VideoCapture, always read latest frame """
    
    def __init__(self, camera_id):
        # "camera_id" is a int type id or string name
        self.cap = cv2.VideoCapture(camera_id)
        self.q = queue.Queue(maxsize=3)
        self.stop_threads = False    # to gracefully close sub-thread
        th = threading.Thread(target=self._reader)
        th.daemon = True             # 设置工作线程为后台运行
        th.start()

    # 实时读帧，只保存最后一帧
    def _reader(self):
        while not self.stop_threads:
            ret, frame = self.cap.read()
            if not ret:
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait() 
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()
    
    def terminate(self):
        self.stop_threads = True
        self.cap.release()


        
if __name__ == "__main__":        
    # 测试自定义VideoCapture类
    cap = VideoCapture(1)
    while True:
        t0 = time.time()
        frame = cap.read()
        t1 = time.time()
        print('cam time->{:.2f}ms'.format((t1-t0 )* 1000))
        # time.sleep(0.05)   # 模拟耗时操作，单位：秒   
        cv2.imshow("frame", frame)
        if chr(cv2.waitKey(1)&255) == 'q':  # 按 q 退出
            cap.terminate()
            break
