'''
Author: wuyao sss
Date: 2024-03-05 20:50:33
LastEditors: wuyao sss
LastEditTime: 2024-03-07 17:46:41
FilePath: /rqh/Detect-and-read-meters/yolov8/time_code.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import time

def time_code_execution(code_to_test, num=5):
    """
    测试给定代码段的执行时间，并打印结果（保留两位小数）。

    参数:
    code_to_test (callable): 要测试的代码段（函数或代码块）。

    返回:
    None
    """
    for i in range(num):
        start_time = time.time()
        code_to_test()
        end_time = time.time()
        execution_time = (end_time - start_time)*1000
        print("执行时间: {:.2f} ms".format(execution_time))


def time_codeblock_execution(code_to_test, num=5):
    """
    测试给定代码段的执行时间，并打印结果（保留两位小数）。

    参数:
    code_to_test (str): 要测试的代码段（字符串形式）。
    num (int): 执行测试的次数。

    返回:
    None
    """
    for i in range(num):
        start_time = time.time()
        exec(code_to_test)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        print("执行时间: {:.2f} ms".format(execution_time))