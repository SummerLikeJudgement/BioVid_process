import os
import logging


# 确保路径存在
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 日志设置
def set_log(log_name):
    # 设置日志输出
    log_dir = './log'
    check_path(log_dir)

    logging.basicConfig(
        filename=os.path.join(log_dir, log_name),
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

def getlabel(filename):
    """根据文件名获取标签"""
    labels = {
        'BL1': 'P0',
        'PA1': 'P1',
        'PA2': 'P2',
        'PA3': 'P3',
        'PA4': 'P4'
    }

    for key, value in labels.items():
        if key in filename:
            return value
    return 'P0'
