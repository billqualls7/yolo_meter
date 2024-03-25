'''
Author: wuyao 1955416359@qq.com
Date: 2024-03-20 15:55:53
LastEditors: wuyao 1955416359@qq.com
LastEditTime: 2024-03-25 16:45:03
FilePath: /yolo_meter/src/infer_air.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import cv2
from air_4c import Find_Air
from air_4c import Find_AirKnob
from videocapture import VideoCapture



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

    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 可以选择不同的编码器
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))
    img = cv2.imread('/home/rqh/yolo_meter/airknob/frame_4.jpg')
    cls, cropped_image = FAir.infer_trt(img)
    FAir_knob.infer(cropped_image)
    cv2.imwrite('../airknob/infer_air.jpg', cropped_image)
    # for i in range(10):
    cap = VideoCapture(0)
    while True:
        frame = cap.read() 
        # frame = cv2.resize(frame, (640, 480))

        try:
        # img = cv2.imread('/home/rqh/yolo_meter/airknob/frame_4.jpg')
            cls, cropped_image = FAir.infer_trt(frame)
            value = FAir_knob.infer(cropped_image)
            # cv2.imwrite('../airknob/infer_air.jpg', cropped_image)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(cropped_image, str(value), (30, 30), font, 1.5, FAir_knob.colormap.white, 2)
            cv2.imshow("infer", cropped_image)

            if chr(cv2.waitKey(1)&255) == 'q':  # 按 q 退出
                cap.terminate()
                break
            # out.write(cropped_image)
        except:
            cv2.imshow("infer", frame) 
            if chr(cv2.waitKey(1)&255) == 'q':  # 按 q 退出
                cap.terminate()
                break
            continue
cap.release()
out.release()
cv2.destroyAllWindows()