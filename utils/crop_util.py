import os
import logging


# 确保路径存在
def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 日志设置，可重复调用
def set_log(log_name):
    # 设置日志输出
    log_dir = './log'
    check_path(log_dir)

    # 获取或创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 移除已有的handler，避免重复
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # 创建文件handler
    file_handler = logging.FileHandler(os.path.join(log_dir, log_name))
    file_handler.setLevel(logging.INFO)

    # 创建控制台handler（可选）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 设置格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 添加handler
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def getlabel(filename):
    """根据文件名获取标签"""
    labels = {
        'BL1': 0,
        'PA1': 1,
        'PA2': 2,
        'PA3': 3,
        'PA4': 4
    }
    for key, value in labels.items():
        if key in filename:
            return value
    return 0


# def get_files(folder):
#     for subject in os.listdir(folder):
#         subject_path = os.path.join(folder, subject)
#         if os.path.isdir(subject_path):
#             for file in os.listdir(subject_path):
#                 label = getlabel(file)
