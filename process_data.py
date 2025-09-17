import os
from utils.face_crop import main as crop

data_path = {
    'vedio':r"D:\EmoData\BioVid\PartA\video"
}
out_path = {
    'vedio':os.path.abspath("./processed/vedio")
}


modal = input("which modal process?(vedio, ecg, gsr)")
if modal == "vedio":
    crop(data_path['vedio'], out_path['vedio'])
else:
    print("no modal!")