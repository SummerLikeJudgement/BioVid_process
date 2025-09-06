import os
import numpy as np
import cv2
import mediapipe as mp
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

# 初始化 MediaPipe 人脸检测
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


# 裁剪人脸
def crop_face(img, output_folder_path, image_name):
    with mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        # 转换图像为 RGB（MediaPipe 需要 RGB 格式）
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_detection.process(rgb_img)

        # 无法识别面部的图片
        if not results.detections:
            return False

        for detection in results.detections:
            # 获取人脸边界框
            bbox = detection.location_data.relative_bounding_box
            h, w, _ = img.shape

            # 计算实际像素坐标
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)

            # 扩展边界框以获得更好的裁剪（包含更多面部区域）
            expand_factor = 0.2  # 扩展20%
            x_expand = max(0, x - int(width * expand_factor))
            y_expand = max(0, y - int(height * expand_factor))
            width_expand = min(w - x_expand, int(width * (1 + 2 * expand_factor)))
            height_expand = min(h - y_expand, int(height * (1 + 2 * expand_factor)))

            # 截取对齐人脸
            cropped_img = img[y_expand:y_expand + height_expand, x_expand:x_expand + width_expand]

            # 确保裁剪的图像不为空
            if cropped_img.size == 0:
                continue

            resized = cv2.resize(cropped_img, (224, 224), interpolation=cv2.INTER_AREA)

            # 将图像保存到输出目录
            output_path = os.path.join(output_folder_path, image_name)
            cv2.imwrite(output_path, resized)
            return True

    return False


# 视频转图片
def getpic(input_folder_path, output_folder_path):
    cap = cv2.VideoCapture(input_folder_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video: {input_folder_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 处理视频总帧数为0的情况
    if total_frames == 0 or fps == 0:
        logging.error(f"Invalid video parameters: {input_folder_path} (fps: {fps}, total_frames: {total_frames})")
        cap.release()
        return

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
            elif numFrame > end_frame:
                break
        else:
            break

    cap.release()


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
                logging.info(f"Processing {subject}_video_{i}: {vedio}")
                label = getlabel(vedio)
                vedio_name = subject + "_" + label + "_" + str(i)
                output_path = os.path.join(face_folder, subject, vedio_name)
                check_path(output_path)  # 确保输出文件夹存在

                video_file_path = os.path.join(vedio_path, vedio)
                if os.path.isfile(video_file_path):
                    getpic(video_file_path, output_path)
                else:
                    logging.error(f"Video file not found: {video_file_path}")

    logging.info('Done!')


# 如果直接运行这个文件，可以使用默认路径
if __name__ == "__main__":
    vedio_folder = r""
    face_folder = r""
    main(vedio_folder, face_folder)