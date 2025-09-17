import os
import cv2
import logging
import subprocess
import tempfile
import pandas as pd
import glob

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

# 设置日志输出
log_dir = './log'
check_path(log_dir)
# openface设置
openface_path = 'D:\Code\OpenFace_2.2.0_win_x64\FeatureExtraction.exe'

logging.basicConfig(
    filename=os.path.join(log_dir, 'face_crop.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)


# openface提取AU
def extract_au_from_images(image_dir, output_dir):
    try:
        # 构建OpenFace命令
        cmd = [
            openface_path,
            '-fdir', image_dir,  # 输入图片目录
            '-out_dir', output_dir,  # 输出目录
            '-aus',  # 输出AU特征
        ]

        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=r'D:\Code\OpenFace_2.2.0_win_x64')

        if result.returncode != 0:
            logging.error(f"OpenFace failed for images in {image_dir}: {result.stderr}")
            return None

        # 查找生成的CSV文件 - FaceLandmarkImg会生成一个汇总的CSV文件
        csv_files = glob.glob(os.path.join(output_dir, "*.csv"))
        if csv_files:
            return csv_files[0]  # 返回第一个找到的CSV文件
        else:
            logging.error(f"No CSV file found in {output_dir}")
            return None

    except Exception as e:
        logging.error(f"Error running OpenFace for images: {str(e)}")
        return None

# 截取视频前2秒和后0.5秒之间的帧并保存为图片，返回保存的图片数量
def save_video_frames(video_path, temp_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Cannot open video: {video_path}")
        return 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0 or fps == 0:
        logging.error(f"Invalid video parameters: {video_path}")
        cap.release()
        return 0

    duration = total_frames / fps
    start_frame = int(2.0 * fps)
    end_frame = int(fps * (duration - 0.5))

    saved_count = 0

    for frame_num in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if frame_num >= start_frame and frame_num < end_frame:
            # 保存帧图像到临时目录
            frame_path = os.path.join(temp_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

    cap.release()
    return saved_count

# 处理单个视频：截取帧 → 保存为图片 → 使用OpenFace提取AU → 清理图片
def process_video_with_openface(video_path, output_folder):
    # 创建临时目录保存帧图像
    with tempfile.TemporaryDirectory() as temp_dir:
        # 1. 截取视频帧并保存为图片
        saved_count = save_video_frames(video_path, temp_dir)

        if saved_count == 0:
            logging.warning(f"No frames extracted from {video_path}")
            return

        # 2. 使用OpenFace处理图片目录
        temp_folder = os.path.abspath('./temp')
        csv_file = extract_au_from_images(temp_dir, temp_folder)

        if csv_file:
            # 3. 处理AU数据
            try:
                au_data = pd.read_csv(csv_file)
                # print(f"Extracted columns from CSV: {au_data.columns.tolist()}")

                # 提取AU相关的列（AU开头的列）
                au_data.columns = au_data.columns.str.strip() # 去掉列名的空格
                au_columns = [col for col in au_data.columns if col.startswith('AU')]
                # 选择需要的列
                selected_columns = ['frame', 'success'] + au_columns
                available_columns = [col for col in selected_columns if col in au_data.columns]
                # print(available_columns)

                if available_columns:
                    selected_data = au_data[available_columns]

                    # 保存处理后的AU数据
                    video_name = os.path.splitext(os.path.basename(video_path))[0]
                    output_csv = os.path.join(output_folder, f"{video_name}_au.csv")
                    selected_data.to_csv(output_csv, index=False)
                    logging.info(f"Processed AU data saved to {output_csv}")
                else:
                    logging.warning(f"No AU columns found in {csv_file}")

            except Exception as e:
                logging.error(f"Error processing AU data from {csv_file}: {str(e)}")
            # 4. 临时目录清理
            try:
                for file in glob.glob(os.path.join(temp_folder, '*')):
                    os.remove(file)  # 删除文件
                logging.info(f"Deleted all files in temporary folder: {temp_folder}")
            except Exception as e:
                logging.error(f"Error deleting files in temporary folder: {str(e)}")


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


def main(video_folder_path, output_folder_path):
    """主函数"""
    check_path(output_folder_path)
    logging.info("==== Processing videos with OpenFace ====")

    # 支持的视频格式
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv')

    for subject in os.listdir(video_folder_path):
        subject_path = os.path.join(video_folder_path, subject)
        if os.path.isdir(subject_path):
            for video_file in os.listdir(subject_path):
                label = getlabel(video_file)
                video_name = os.path.splitext(video_file)[0]
                output_path = os.path.join(output_folder_path, subject, f"{video_name}_{label}")
                check_path(output_path)

                video_file_path = os.path.join(subject_path, video_file)
                if os.path.isfile(video_file_path):
                    logging.info(f"Processing {video_file_path}")
                    process_video_with_openface(video_file_path, output_path)
                else:
                    logging.error(f"Video file not found: {video_file_path}")

            logging.info(f"Processed {subject} videos")

    logging.info('Done!')

# 如果直接运行这个文件，可以使用默认路径
if __name__ == "__main__":
    video_folder = r""  # 输入视频文件夹路径
    output_folder = r""  # 输出文件夹路径

    if video_folder and output_folder:
        main(video_folder, output_folder)
    else:
        print("Please set video_folder and output_folder paths")