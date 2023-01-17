import os
import cv2
import sys
from tqdm import tqdm     # python 进度条库

image_folder_dir = "/workspace/MultiObjTracking/tflite_test/videoDR/"
fps = 11     # fps: frame per seconde 每秒帧数，数值可根据需要进行调整
size = (506,478)     # (width, height) 数值可根据需要进行调整
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
video = cv2.VideoWriter("/workspace/MultiObjTracking/tflite_test/video/video_DR_1.mp4", fourcc, fps, size, isColor=True)

image_list = []
for i in range(18,463,2):
    name = image_folder_dir + 'frame_' + str(i) + '.jpg'
    image_list.append(name)

for image_name in tqdm(image_list):     # 遍历 image_list 中所有图像并添加进度条
 image = cv2.imread(image_name)     # 读取图像
 video.write(image)     # 将图像写入视频

video.release()
cv2.destroyAllWindows()