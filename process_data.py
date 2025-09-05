import os
from utils.face_crop import main as crop

data_path = {
    'vedio':os.path.abspath("./face_data")
}
out_path = {
    'vedio':os.path.abspath("./processed/face")
}


modal = input("which modal process?(vedio, ecg, gsr)")
if modal == "vedio":
    crop(data_path['vedio'], out_path['vedio'])
