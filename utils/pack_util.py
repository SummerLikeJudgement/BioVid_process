import os, pickle, csv
import numpy as np
import pandas as pd
from .crop_util import getlabel

def find_path(base_dir, subject):
    sub_path = os.path.join(base_dir, subject)
    res = []
    for feat in os.listdir(sub_path):
        feat_path = os.path.join(sub_path, feat)
        res.append(feat_path)
    # print(res)
    return res

def find_label(base_dir, subject):
    sub_path = os.path.join(base_dir, subject)
    res = []
    for feat in os.listdir(sub_path):
        res.append(getlabel(feat))
    return res

def find_id(base_dir, subject):
    sub_path = os.path.join(base_dir, subject)
    res = []
    for feat in os.listdir(sub_path):
        res.append(feat[:-8])
    return res


def ecg_read(subject_path):
    """读取ecg特征并连接"""
    data = []
    for path in subject_path:
        df = pd.read_csv(path, index_col=0)# 第一列为索引
        data.append(df.to_numpy())
    return np.stack(data, axis=0)

def gsr_read(subject_path):
    """读取gsr特征并连接"""
    mfcc = []
    for path in subject_path:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            mfcc.append(data["mfcc"])
    return np.stack(mfcc, axis=0)

def vision_read(subject_path):
    """读取vision特征并连接"""
    data = []
    for path in subject_path:
        df = pd.read_csv(path)  # 第一列为索引
        data.append(df.iloc[:, 2:].to_numpy())
    return np.stack(data, axis=0)


if __name__ == '__main__':
    ecg_read(r"D:\Code\python\DeepLearning\BioVid_process\BioVid process\processed\ecg\071309_w_21\P0_071309_w_21-BL1-081_bio.csv")