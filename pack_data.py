import logging, itertools, pickle
import numpy as np
from utils.crop_util import set_log
from utils.pack_util import find_id, find_label, find_path, ecg_read, gsr_read, vision_read

set_log("pack_data.log")

# test:19*100 valid:7*100 train:61*100
subject = {
    "test":["100914_m_39", "101114_w_37", "082315_w_60", "083114_w_55", "080309_m_29", "112016_m_25", "112310_m_20", "092813_w_24", "112809_w_23", "071313_m_41", "101309_m_48", "091809_w_43", "102214_w_36", "102316_w_50", "101814_m_58", "101908_m_61", "102309_m_61", "112610_w_60", "112914_w_51"],
    "valid":["083109_m_60", "112909_w_20", "072514_m_27", "112009_w_43", "101609_m_36", "120514_w_56", "112209_m_51"],
    "train":['071309_w_21', '071614_m_20', '071709_w_23', '071814_w_23', '071911_w_24', '072414_m_23', '072609_w_23', '072714_m_23', '073109_w_28', '073114_m_25', '080209_w_26', '080314_w_25', '080609_w_27', '080614_m_24', '080709_m_24', '080714_m_23', '081014_w_27', '081609_w_40', '081617_m_27', '081714_m_36', '082014_w_24', '082109_m_53', '082208_w_45', '082414_m_64', '082714_m_22', '082809_m_26', '082814_w_46', '082909_m_47', '083009_w_42', '083013_w_47', '091814_m_37', '091914_m_46', '092009_m_54', '092014_m_56', '092509_w_51', '092514_m_50', '092714_m_64', '092808_m_51', '100117_w_36', '100214_m_50', '100417_m_44', '100509_w_43', '100514_w_51', '100909_w_65', '101015_w_43', '101209_w_61', '101216_m_40', '101514_w_36', '101809_m_59', '101916_m_40', '102008_w_22', '102414_w_58', '102514_w_40', '110614_m_42', '110810_m_62', '110909_m_29', '111313_m_64', '111409_w_63', '111609_m_65', '111914_w_63', '120614_w_61']
}

ecg_path = "./processed/ecg/"
gsr_path = "./processed/gsr/"
vedio_path = "./processed/vedio/"



def pack(mode = "train"):
    logging.info(f"====Processing {mode} set====")
    subs = subject[mode]
    ecg, gsr, vision = [], [], []
    label, id = [], []

    # 查找所有匹配的文件路径
    for sub in subs:
        logging.info(f"Processing {sub} features")
        label = list(itertools.chain(label, find_label(ecg_path, sub)))
        id = list(itertools.chain(id, find_id(gsr_path, sub)))
        ecg = list(itertools.chain(ecg, find_path(ecg_path, sub)))
        gsr = list(itertools.chain(gsr, find_path(gsr_path, sub)))
        vision = list(itertools.chain(vision, find_path(vedio_path, sub)))

    set = {}
    set["id"] = id
    set["classification_labels"] = np.array(label)
    set["ecg"] = ecg_read(ecg)
    set["gsr"] = gsr_read(gsr)
    set["vision"] = vision_read(vision)
    set["ecg_lengths"] = [29]*(len(subs)*100)
    set["gsr_lengths"] = [8]*(len(subs)*100)
    set["vision_lengths"] = [75]*(len(subs)*100)

    return set

if __name__ == "__main__":
    train = pack("train")
    test = pack("test")
    valid = pack("valid")
    dataset = {
        "train": train,
        "test": test,
        "valid": valid
    }
    with open("./processed/unaligned.pkl", "wb") as f:
        pickle.dump(dataset, f)
    logging.info("pkl saved!")