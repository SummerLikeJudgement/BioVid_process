import os
from face_crop import main as AU_crop
from gsr_crop import main as GSR_crop

data_path = {
    'vedio':r"D:\EmoData\BioVid\PartA\video",
    'gsr':r"D:\EmoData\BioVid\PartA\biosignals_filtered"
}
out_path = {
    'vedio':os.path.abspath("./processed/vedio"),
    'gsr':os.path.abspath("./processed/gsr")
}


modal = input("which modal process?(vedio, ecg, gsr)")
if modal == "vedio":
    AU_crop(data_path['vedio'], out_path['vedio'])
elif modal == "gsr":
    GSR_crop(data_path['gsr'], out_path['gsr'])
else:
    print("no modal!")