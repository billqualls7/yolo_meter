# 基于YOLOv8的仪表检测





环境依赖见

environment.yml

安装TRT

```bash
python3 -m pip install nvidia-pyindex

python3 -m pip install --upgrade nvidia-tensorrt

pip3 install nvidia-tensorrt==8.4.3.1
```

检测圆表

src/output_trt_debug.py

检测空调

src/infer_air.py



参考

https://github.com/shouxieai/tensorRT_Pro

https://github.com/shuyansy/Detect-and-read-meters

https://github.com/Melody-Zhou/tensorRT_Pro-YOLOv8

[YOLOv8-Seg推理详解及部署实现-CSDN博客](https://blog.csdn.net/qq_40672115/article/details/134277752)