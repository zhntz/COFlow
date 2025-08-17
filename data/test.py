from PIL import Image
import numpy as np
import cv2

ads = '/home/dell/Flyingchairs/mv/frame_'
save = '/home/dell/Flyingchairs/mv/frame_'

img = Image.open('/home/dell/flo_gan/SPI_data_QP22/alley_1/occ/train/frame_0002.png')
img = np.array(img)
print(img)
print(np.max(img))