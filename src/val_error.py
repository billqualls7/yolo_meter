'''
Author: wuyao 1955416359@qq.com
Date: 2024-03-15 16:44:03
LastEditors: wuyao 1955416359@qq.com
LastEditTime: 2024-03-15 17:56:36
FilePath: /yolo_meter/src/val_error.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


with open('log.json', 'r') as f:
    data_dict = json.load(f)
# print(data_dict)

data_array = [(key, value) for key, value in data_dict.items()]
data_array = [(name.split('.')[0], value) for name, value in data_array]
# print(data_array)


val_label = []
with open('val_label.txt', 'r') as f:
    for line in f:
        val = float(line.strip())  # 将每行的值转换为浮点数
        val_label.append(val)

# print(val_label)


result_array = [(name, value, val) for (name, value), val in zip(data_array, val_label)]
print(result_array)
# print(type(result_array[0]))


# print(len(result_array))


def calculate_error(predicted, actual):
    return abs(predicted - actual)

# 遍历每个元组，计算预测值和真实值之间的误差
errors = []
for data_tuple in result_array:
    name, predicted, actual = data_tuple
    error = calculate_error(predicted, actual)
    errors.append(error)

# 输出误差列表
print("误差列表:", errors)



# 计算总体误差
total_error = sum(errors)
print("总体误差:", total_error)

# 计算均方误差
mean_squared_error = np.mean(np.square(errors))
print("均方误差:", mean_squared_error)

# 计算平均误差
mean_error = np.mean(errors)
print("平均误差:", mean_error)

# 计算最大误差和最小误差
max_error = max(errors)
min_error = min(errors)
print("最大误差:", max_error)
print("最小误差:", min_error)

# 计算误差的标准差
std_deviation = np.std(errors)
print("误差标准差:", std_deviation)


selected_data = [(name, pred, true) for name, pred, true in result_array if pred < 2 ]
print(selected_data)
print(len(selected_data))
errors = [pred - true for _, pred, true in selected_data]


# 计算总体误差
total_error = sum(errors)
print("25总体误差:", total_error)

# 计算均方误差
mean_squared_error = np.mean(np.square(errors))
print("25均方误差:", mean_squared_error)

# 计算平均误差
mean_error = np.mean(errors)
print("25平均误差:", mean_error)

# 计算最大误差和最小误差
max_error = max(errors)
min_error = min(errors)
print("25最大误差:", max_error)
print("25最小误差:", min_error)

# 计算误差的标准差
std_deviation = np.std(errors)
print("25误差标准差:", std_deviation)