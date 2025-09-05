import os
from VEDIO.train import main as vision_train

data_dir = {
    'vision':'./processed/face'
}

def model_train(model_name = "vision"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if model_name == 'vision':
        abs_data_dir = os.path.join(base_dir, data_dir['vision'])
        vision_train(abs_data_dir, 20)
    else:
        raise ValueError('Model name must be ecg or gsr or vision')