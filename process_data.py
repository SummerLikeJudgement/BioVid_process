import os
from face_crop import main as AU_crop
from gsr_crop import main as GSR_crop

data_path = {
    'vedio':r"D:\EmoData\BioVid\PartA\temp",
    'gsr':r"D:\EmoData\BioVid\PartA\biosignals_filtered",
    'ecg':r"D:\EmoData\BioVid\PartA\biosignals_filtered"
}
out_path = {
    'vedio':os.path.abspath("./processed/vedio"),
    'gsr':os.path.abspath("./processed/gsr"),
    'ecg':os.path.abspath("./processed/ecg")
}


modal = input("which modal process?(vedio, ecg, gsr)")
if modal == "vedio":
    AU_crop(data_path['vedio'], out_path['vedio'])
elif modal == "gsr":
    GSR_crop(data_path['gsr'], out_path['gsr'])
# todo:elif modal == "ecg":
#     ECG_crop(data_path['ecg'], out_path['ecg'])
else:
    print("no modal!")