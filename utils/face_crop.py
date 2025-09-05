import os
import numpy as np
import cv2
from mtcnn import MTCNN
import logging

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 设置日志输出
log_dir = './log'
check_path(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, 'face_crop.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)

# 裁剪人脸
def crop_face(img, output_folder_path, image_name):
    detector = MTCNN()
    faces = detector.detect_faces(img)
    # 无法识别面部的图片
    if len(faces) == 0:
        return False

    for face in faces:
        # 获取人脸坐标
        x1, y1, width, height = face['box']
        # 截取对齐人脸
        cropped_img = img[y1:y1 + height, x1:x1 + width]
        resized = cv2.resize(cropped_img, (224, 224), interpolation=cv2.INTER_AREA)
        # 将图像保存到输出目录
        output_path = os.path.join(output_folder_path, image_name)
        cv2.imwrite(output_path, resized)
        return True

# 视频转图片
def getpic(input_folder_path, output_folder_path):
    cap = cv2.VideoCapture(input_folder_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps  # 视频总时长(秒)

    start_frame = int(2.0 * fps)
    end_frame = int(fps * (duration - 0.5))

    count = 0
    numFrame = 0
    while True:
        if cap.grab():
            flag, frame = cap.retrieve()
            if not flag:
                continue
            numFrame += 1
            if numFrame >= start_frame and numFrame <= end_frame:
                count += 1
                image_name = f"{str(count).zfill(5)}.jpg"
                detected = crop_face(frame, output_folder_path, image_name)  # 直接裁剪并保存
                if not detected:
                    logging.warning(f"{output_folder_path}/{image_name} no face_data!")
            else:
                continue
        else:
            break

def getlabel(filename):
    if 'BL1' in filename:
        return 'P0'
    elif 'PA1' in filename:
        return 'P1'
    elif 'PA2' in filename:
        return 'P2'
    elif 'PA3' in filename:
        return 'P3'
    elif 'PA4' in filename:
        return 'P4'
    else:
        return 'P0'


def main(vedio_folder_path, output_folder_path):
    vedio_folder = vedio_folder_path
    face_folder = output_folder_path

    # 创建输出目录
    check_path(face_folder)

    logging.info("====getpic====")

    for subject in os.listdir(vedio_folder):
        vedio_path = os.path.join(vedio_folder, subject)
        if os.path.isdir(vedio_path):
            for i, vedio in enumerate(os.listdir(vedio_path)):
                logging.info(f"{subject}_video_{i}")
                label = getlabel(vedio)
                vedio_name = subject + "_" + label + "_" + str(i)
                output_path = os.path.join(face_folder, subject, vedio_name)
                check_path(output_path)  # 确保输出文件夹存在
                getpic(os.path.join(vedio_path, vedio), output_path)

    logging.info('Done!')