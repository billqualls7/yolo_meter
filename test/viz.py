import os
import cv2

# 定义类别名称
class_names = ['0.4', '2', '5']

def visualize_yolo_labels(image_folder, label_folder):
    # 获取文件夹中的所有图片和标签文件
    image_files = sorted(os.listdir(image_folder))
    label_files = sorted(os.listdir(label_folder))
    
    # 遍历图片文件
    for image_file, label_file in zip(image_files, label_files):
        # 图片路径
        image_path = os.path.join(image_folder, image_file)
        
        # 对应的标签文件路径
        label_path = os.path.join(label_folder, label_file)
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            continue

        image_h, image_w, _ = image.shape

        # 如果没有对应的标签文件，则跳过
        if not os.path.exists(label_path):
            print(f"No label file found for image: {image_path}")
            continue

        # 读取标签
        with open(label_path, 'r') as f:
            lines = f.readlines()

        for line in lines:
            class_idx, center_x, center_y, width, height = map(float, line.split())
            
            # 确保类别索引在有效范围内
            class_idx = int(class_idx)
            if class_idx < 0 or class_idx >= len(class_names):
                print(f"Invalid class index {class_idx} in label file: {label_path}")
                continue
            
            # 将坐标转换为图像上的坐标
            x = int((center_x - width / 2) * image_w)
            y = int((center_y - height / 2) * image_h)
            w = int(width * image_w)
            h = int(height * image_h)
            
            # 绘制边界框
            color = (0, 255, 0)  # 绿色边界框
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            
            # 标注类别名称
            label = class_names[class_idx]
            cv2.putText(image, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # 显示图像
        cv2.imshow('Image with Labels', image)
        
        # 按任意键进入下一张图片
        cv2.waitKey(0)
    
    cv2.destroyAllWindows()

# 示例用法
image_folder = 'images'  # 图片文件夹路径
label_folder = 'labels'  # 标签文件夹路径
visualize_yolo_labels(image_folder, label_folder)
