import os 
import cv2
import numpy as np
from configs import BG_COLOR

class Obs_Processor():
    def __init__(self) -> None:
        self.downsample_rate = 4 
        self.n_channels = 3

    def process_img(self, img):
        processed_img = self.change_bg_color(img)
        processed_img = cv2.resize(processed_img, (img.shape[0]//self.downsample_rate, img.shape[1]//self.downsample_rate))
        processed_img = processed_img/255.0

        return processed_img

    def change_bg_color(self, img):
        processed_img = img.copy()
        bg_pos = img==BG_COLOR[:3]
        bg_pos = (np.sum(bg_pos,axis=-1) == 3)
        processed_img[bg_pos] = (0,0,0)
        return processed_img
    
    def save_processed_img(self, img, path):
        save_path = './env/log/img'
        num_files = len([f for f in os.listdir(save_path) if os.path.isfile(os.path.join(save_path, f))])
        filename = f'image_{num_files}.png'
        path = os.path.join(save_path, filename)
        img = (img * 255).astype(np.uint8) 
        bgr_image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr_image)
    