import cv2
import torch
import numpy as np
import torch.nn.functional as F
from ultralytics.data.augment import LetterBox
from ultralytics.nn.autobackend import AutoBackend
import timeit
from time_code import time_code_execution
from skimage import morphology
from videocapture import VideoCapture
import time

def preprocess_letterbox(image):
    letterbox = LetterBox(new_shape=640, stride=32, auto=True)
    image = letterbox(image=image)
    image = (image[..., ::-1] / 255.0).astype(np.float32) # BGR to RGB, 0 - 255 to 0.0 - 1.0
    image = image.transpose(2, 0, 1)[None]  # BHWC to BCHW (n, 3, h, w)
    image = torch.from_numpy(image)
    return image
 
def preprocess_warpAffine(image, dst_width=640, dst_height=640):
    scale = min((dst_width / image.shape[1], dst_height / image.shape[0]))
    ox = (dst_width  - scale * image.shape[1]) / 2
    oy = (dst_height - scale * image.shape[0]) / 2
    M = np.array([
        [scale, 0, ox],
        [0, scale, oy]
    ], dtype=np.float32)
    
    img_pre = cv2.warpAffine(image, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    IM = cv2.invertAffineTransform(M)
 
    img_pre = (img_pre[...,::-1] / 255.0).astype(np.float32)
    img_pre = img_pre.transpose(2, 0, 1)[None]
    img_pre = torch.from_numpy(img_pre)
    return img_pre, IM
 
def iou(box1, box2):
    def area_box(box):
        return (box[2] - box[0]) * (box[3] - box[1])
    
    left, top = max(box1[:2], box2[:2])
    right, bottom = min(box1[2:4], box2[2:4])
    union = max((right-left), 0) * max((bottom-top), 0)
    cross = area_box(box1) + area_box(box2) - union
    if cross == 0 or union == 0:
        return 0
    return union / cross
 
def NMS(boxes, iou_thres):
    
    remove_flags = [False] * len(boxes)
 
    keep_boxes = []
    for i, ibox in enumerate(boxes):
        if remove_flags[i]:
            continue
 
        keep_boxes.append(ibox)
        for j in range(i + 1, len(boxes)):
            if remove_flags[j]:
                continue
 
            jbox = boxes[j]
            if(ibox[5] != jbox[5]):
                continue
            if iou(ibox, jbox) > iou_thres:
                remove_flags[j] = True
    return keep_boxes
 
def postprocess(pred, conf_thres=0.25, iou_thres=0.45):
 
    # 输入是模型推理的结果，即8400个预测框
    # 1,8400,116 [cx,cy,w,h,class*80,32]
    boxes = []
    for item in pred[0]:
        cx, cy, w, h = item[:4]
        label = item[4:-32].argmax()
        confidence = item[4 + label]
        if confidence < conf_thres:
            continue
        left    = cx - w * 0.5
        top     = cy - h * 0.5
        right   = cx + w * 0.5
        bottom  = cy + h * 0.5
        boxes.append([left, top, right, bottom, confidence, label, *item[-32:]])
        
    boxes = sorted(boxes, key=lambda x:x[4], reverse=True)
 
    return NMS(boxes, iou_thres)
 
def crop_mask(masks, boxes):
    
    # masks -> n, 160, 160  原始 masks
    # boxes -> n, 4         检测框，映射到 160x160 尺寸下的
    n, h, w = masks.shape
    x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(n,1,1)
    r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,1,w)
    c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(1,h,1)
 
    return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
 
def process_mask(protos, masks_in, bboxes, shape, upsample=False):
 
    # protos   -> 32, 160, 160 分割头输出
    # masks_in -> n, 32        检测头输出的 32 维向量，可以理解为 mask 的权重
    # bboxes   -> n, 4         检测框
    # shape    -> 640, 640     输入网络中的图像 shape
    # unsample 一个 bool 值，表示是否需要上采样 masks 到图像的原始形状
    c, mh, mw = protos.shape  # CHW
    ih, iw = shape
    # 矩阵相乘 nx32 @ 32x(160x160) -> nx(160x160) -> sigmoid -> nx160x160
    masks = (masks_in.float() @ protos.float().view(c, -1)).sigmoid().view(-1, mh, mw)  # CHW
 
    downsampled_bboxes = bboxes.clone()
    downsampled_bboxes[:, 0] *= mw / iw
    downsampled_bboxes[:, 2] *= mw / iw
    downsampled_bboxes[:, 3] *= mh / ih
    downsampled_bboxes[:, 1] *= mh / ih
 
    masks = crop_mask(masks, downsampled_bboxes)  # CHW
    if upsample:
        masks = F.interpolate(masks[None], shape, mode='bilinear', align_corners=False)[0]  # CHW
    return masks.gt_(0.5)
 
def hsv2bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    r, g, b = 0, 0, 0
 
    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q
 
    return int(b * 255), int(g * 255), int(r * 255)
 
def random_color(id):
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)


def judge(p1,p2,p3):
    A=p2[1]-p1[1]
    B=p1[0]-p2[0]
    C=p2[0]*p1[1] - p1[0]*p2[1]

    value=A*p3[0] + B*p3[1] +C

    return value

def angle(v1, v2):
    lx=np.sqrt(v1.dot(v1))
    ly=np.sqrt(v2.dot(v2))
    cos_angle=v1.dot(v2) / (lx * ly)

    angle=np.arccos(cos_angle)
    angle2=angle*360 / 2 / np.pi

    return angle2

def getangle(ori_img, std_point, pointer_line, number):
    a1 = std_point[0]
    a2 = std_point[1]
    cv2.circle(ori_img, a1, 2, (255, 0, 0), 2)
    cv2.circle(ori_img, a2, 2, (255, 0, 0), 2)
    one = [[pointer_line[0][0], pointer_line[0][1]], [a1[0], a1[1]]]
    two = [[pointer_line[0][0], pointer_line[0][1]], [a2[0], a2[1]]]
    three = [[pointer_line[0][0], pointer_line[0][1]], [pointer_line[1][0], pointer_line[1][1]]]
    # print("one", one)
    # print("two", two)
    # print("three",three)

    one=np.array(one)
    two=np.array(two)
    three = np.array(three)

    v1=one[1]-one[0]
    v2=two[1]-two[0]
    v3 = three[1] - three[0]

    distance=get_distance_point2line([a1[0], a1[1]],[pointer_line[0][0], pointer_line[0][1], pointer_line[1][0], pointer_line[1][1]])
    # print("dis",distance)

    flag=judge(pointer_line[0],std_point[0],pointer_line[1])
    # print("flag",flag)

    std_ang = angle(v1, v2)
    
    now_ang = angle(v1, v3)
    print("std_result: {:.3f}".format(std_ang))
    print("now_ang:   {:.3f}".format(now_ang))

    # if flag >0:
    #     now_ang=360-now_ang
    #     print("now_result", now_ang)


    # calculate value
    if number!=None and number[0]!="":
        two_value = float(number[0])
        # print(two_value)
    else:
        return "can not recognize number"
    
    if two_value == 0.4:
        correction_value = 0.3333*1.05
    if two_value == 5:
        correction_value = 0.3333*17


    if flag>0: 
        k1 = 0.9
        k2 = 1 - k1
        two_value = ((k1 * correction_value)+(k2 * two_value))
    else :
        k1 = 0.5
        k2 = 1 - k1
        two_value = ((k1 * correction_value)+(k2 * two_value))

    if std_ang * now_ang !=0:
        value = (two_value / std_ang)
        value=value*now_ang

    else:
        return "angle detect error"

    value=round(value,3)

    # if flag>0 and distance<40:
    #     value=0.00
    # else:
    #     value=round(value,3)

    return value

def get_distance_point2line(point, line):
    """
    Args:
        point: [x0, y0]
        line: [x1, y1, x2, y2]
    """
    line_point1, line_point2 = np.array(line[0:2]), np.array(line[2:])
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
    return distance



class GetAngle():
    """
    计算指针在表盘中的角度 单位 °
    
    """
    def __init__(self,model, w = 640, h = 480):
        self.model = AutoBackend(weights=model)
        self.h = h
        self.w = w
        self.img_center = (0.5 * w, 0.5 * h)
        
        # self.__img_params__()

        

    
    def __img_params__(self):
        # self.h, self.w = img.shape[:2]
        # self.img_center = (0.5 * self.w , 0.5 * self.h)
        print("Height:", self.h)
        print("Width:", self.w)
        print("Img_center:", self.img_center)

    @staticmethod    
    def _img_params_(img):
        h, w = img.shape[:2]
        img_center = (0.5 * w , 0.5 * h)
        print("Height:", h)
        print("Width:", w)
        print("Img_center:", img_center)
        return h, w

    
    def infer(self,img):
        """
        result[0] -> 1, 116, 8400 -> det head
        result[1][0][0] -> 1, 144, 80, 80
        result[1][0][1] -> 1, 144, 40, 40
        result[1][0][2] -> 1, 144, 20, 20
        result[1][1] -> 1, 32, 8400
        result[1][2] -> 1, 32, 160, 160 -> seg head
        """
        img_pre, IM = preprocess_warpAffine(img)
        result = self.model(img_pre)
        output0 = result[0].transpose(-1, -2) # 1,8400,116 检测头输出
        output1 = result[1][2][0]             # 32,160,160 分割头输出
        pred = postprocess(output0)
        pred = torch.from_numpy(np.array(pred).reshape(-1, 38))
        # pred -> nx38 = [cx,cy,w,h,conf,label,32]
        masks = process_mask(output1, pred[:, 6:], pred[:, :4], img_pre.shape[2:], True)
        boxes = np.array(pred[:,:6])
        lr = boxes[:, [0, 2]]
        tb = boxes[:,[1, 3]]
        boxes[:,[0, 2]] = IM[0][0] * lr + IM[0][2]
        boxes[:,[1, 3]] = IM[1][1] * tb + IM[1][2]
        # h, w = img.shape[:2]
        
        return masks, boxes, IM
    
    def mask2keys(self, mask ,label, std_point, pointer_line):

        if label =='dail':
            indices = np.argwhere(mask == 1)
            center = np.mean(indices, axis=0)
            center = tuple(center[::-1].astype(int))
            std_point.append(center)
        if label == 'pointer':
            # indices = np.argwhere(mask_resized == 1)
            pointer_skeleton = morphology.skeletonize(mask)
            pointer_edges = pointer_skeleton * 255
            pointer_edges = pointer_edges.astype(np.uint8)
            # cv2.imwrite("pointer_edges.jpg", pointer_edges)
            pointer_lines = cv2.HoughLinesP(pointer_edges, 1, np.pi / 180, 10, np.array([]), minLineLength=10,
                                            maxLineGap=400)
            try:
                for x1, y1, x2, y2 in pointer_lines[0]:
                    coin1 = (x1, y1)
                    coin2 = (x2, y2)
                    # cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            except TypeError:
                return "can not detect pointer"
            dis1 = (coin1[0] - self.img_center[0]) ** 2 + (coin1[1] - self.img_center[1]) ** 2
            dis2 = (coin2[0] - self.img_center[0]) ** 2 + (coin2[1] - self.img_center[1]) ** 2
            if dis1 <= dis2:
                pointer_line.append ([coin1, coin2])
            else:
                pointer_line.append ([coin2, coin1])


        

    def key_point(self, img, categories = ['dail', 'pointer']):
        masks, boxes, IM = self.infer(img)

        std_point = []
        pointer_lines = []

        for i, mask in enumerate(masks):
            label =int(boxes[i][5])
            label_ =categories[label]
            
            mask = mask.cpu().numpy().astype(np.uint8) # 640x640   
            mask_resized = cv2.warpAffine(mask, IM, (self.w, self.h), flags=cv2.INTER_LINEAR)
            self.mask2keys(mask_resized, label_, std_point, pointer_lines)
            # if center != []:std_point.append(center)
            # if pointer_line != []:  



            #---------------------------colored_mask------------------
            color = np.array(random_color(label))
        
            colored_mask = (np.ones((h, w, 3)) * color).astype(np.uint8)
            masked_colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=mask_resized)
    
            mask_indices = mask_resized == 1
            img[mask_indices] = (img[mask_indices] * 0.6 + masked_colored_mask[mask_indices] * 0.4).astype(np.uint8)
            #---------------------------colored_mask------------------

        print(len(std_point))
        print((std_point))
        if std_point==None:
            return "can not detect dail"
        if std_point[0][1] >= std_point[1][1]:
            pass
        else:
            std_point[0], std_point[1] = std_point[1], std_point[0]
        
        if len(pointer_lines)>1: return "can not detect pointer"

        print("std_point", std_point)
        print("pointer_lines", (pointer_lines))
        return std_point, pointer_lines[0]
        # cv2.imwrite("../images/infer-seg.jpg", img)    
            


def main():
    img = cv2.imread('/home/rqh/Detect-and-read-meters/demo1/8.jpg')
    model = '/home/rqh/Detect-and-read-meters/yolov8/pointer.pt'
    categories = ['dail', 'pointer']
    # img_pre = preprocess_letterbox(img)
    img_pre, IM = preprocess_warpAffine(img)
 
    model  = AutoBackend(weights=model)
    names  = model.names
    result = model(img_pre)
    """
    result[0] -> 1, 116, 8400 -> det head
    result[1][0][0] -> 1, 144, 80, 80
    result[1][0][1] -> 1, 144, 40, 40
    result[1][0][2] -> 1, 144, 20, 20
    result[1][1] -> 1, 32, 8400
    result[1][2] -> 1, 32, 160, 160 -> seg head
    """
 
    output0 = result[0].transpose(-1, -2) # 1,8400,116 检测头输出
    output1 = result[1][2][0]             # 32,160,160 分割头输出
 
    pred = postprocess(output0)
    pred = torch.from_numpy(np.array(pred).reshape(-1, 38))
 
    # pred -> nx38 = [cx,cy,w,h,conf,label,32]
    masks = process_mask(output1, pred[:, 6:], pred[:, :4], img_pre.shape[2:], True)
 
    boxes = np.array(pred[:,:6])
    lr = boxes[:, [0, 2]]
    tb = boxes[:,[1, 3]]
    boxes[:,[0, 2]] = IM[0][0] * lr + IM[0][2]
    boxes[:,[1, 3]] = IM[1][1] * tb + IM[1][2]
 
    # draw mask
    h, w = img.shape[:2]
    std_point = []
    pointer_lines = []
    img_center = (0.5 * w, 0.5 * h)
    coin1, coin2 = None, None
    # print(img.shape[:2])
    for i, mask in enumerate(masks):
        label_ =categories[int(boxes[i][5])]
        

        mask = mask.cpu().numpy().astype(np.uint8) # 640x640
        mask_resized = cv2.warpAffine(mask, IM, (w, h), flags=cv2.INTER_LINEAR)  # 1080x810
        print(mask_resized.shape)
        print(mask.shape)
        # cv2.imwrite("mask_resized.jpg", mask_resized*255)
        label =int(boxes[i][5])
        
        print(label_)
        if label_ =='dail':
            indices = np.argwhere(mask_resized == 1)
            center = np.mean(indices, axis=0)
            center = tuple(center[::-1].astype(int))
            std_point.append(center)
            print(tuple(center[::-1]) )
            cv2.circle(img, center, 3, (0, 255, 0), -1)
        if label_ == 'pointer':
            # indices = np.argwhere(mask_resized == 1)
            pointer_skeleton = morphology.skeletonize(mask_resized)
            pointer_edges = pointer_skeleton * 255
            pointer_edges = pointer_edges.astype(np.uint8)
            # cv2.imwrite("pointer_edges.jpg", pointer_edges)
            pointer_lines = cv2.HoughLinesP(pointer_edges, 1, np.pi / 180, 10, np.array([]), minLineLength=10,
                                        maxLineGap=400)
            

            try:
                for x1, y1, x2, y2 in pointer_lines[0]:
                    coin1 = (x1, y1)
                    coin2 = (x2, y2)
                    cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            except TypeError:
                return "can not detect pointer"


        color = np.array(random_color(label))
        
        colored_mask = (np.ones((h, w, 3)) * color).astype(np.uint8)
        masked_colored_mask = cv2.bitwise_and(colored_mask, colored_mask, mask=mask_resized)
 
        mask_indices = mask_resized == 1
        img[mask_indices] = (img[mask_indices] * 0.6 + masked_colored_mask[mask_indices] * 0.4).astype(np.uint8)
 
        # contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cv2.drawContours(img, contours, -1, random_color(label), 2)
    dis1 = (coin1[0] - img_center[0]) ** 2 + (coin1[1] - img_center[1]) ** 2
    dis2 = (coin2[0] - img_center[0]) ** 2 + (coin2[1] - img_center[1]) ** 2
    if dis1 <= dis2:
        pointer_line = (coin1, coin2)
    else:
        pointer_line = (coin2, coin1)
    
    
    print("pointer_line", pointer_line)
    if std_point==None:
            return "can not detect dail"
    if std_point[0][1] >= std_point[1][1]:
        pass
    else:
        std_point[0], std_point[1] = std_point[1], std_point[0]
    number =['5']
    value = getangle(img, std_point, pointer_line, number)
    # draw box
    # for obj in boxes:
    #     left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
    #     confidence = obj[4]
    #     label = int(obj[5])
    #     color = random_color(label)
    #     cv2.rectangle(img, (left, top), (right, bottom), color = color ,thickness=2, lineType=cv2.LINE_AA)
    #     caption = f"{names[label]} {confidence:.2f}"
    #     w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
    #     cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
    #     cv2.putText(img, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, str(value), (30, 30), font, 1.2, (255, 0,255), 2)
    cv2.imwrite("infer-seg.jpg", img)
    print("save done")    


if __name__ == "__main__":
    # main()
    # cap = VideoCapture(0)
    # frame = cap.read()
    img = cv2.imread('/home/rqh/Detect-and-read-meters/demo1/1032.jpg')
    model = '/home/rqh/yolo_model/pointer.pt'
    h, w = GetAngle._img_params_(img)
    GA = GetAngle(model, w = w, h = h)
    
    for i in range(5):
        start_time = time.time()
        img = cv2.imread('/home/rqh/Detect-and-read-meters/demo1/1032.jpg')
        std_point, pointer_line = GA.key_point(img)
        number =['0.4']
        value = getangle(img, std_point, pointer_line, number)
        print(value)


    
        

        end_time = time.time()
        execution_time = (end_time - start_time)*1000
        print("执行时间: {:.2f} ms".format(execution_time))


    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, str(value), (30, 30), font, 1.2, (255, 0,255), 2)
    cv2.line(img, pointer_line[0], pointer_line[1], (0, 255, 0), 2)
    
    cv2.imwrite("../images/infer-seg.jpg", img)  

    # cap.terminate()
    # time_code_execution(main())