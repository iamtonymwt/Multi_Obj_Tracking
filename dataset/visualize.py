import os, glob
from scipy.io import loadmat
from collections import defaultdict
import numpy as np
from lxml import etree, objectify
import cv2

def visualize_bbox(xml_file, img_file):
    tree = etree.parse(xml_file)
    # load image
    image = cv2.imread(img_file)
    origin = cv2.imread(img_file)
    # 获取一张图片的所有bbox
    for bbox in tree.xpath('//bndbox'):
        coord = []
        for corner in bbox.getchildren():
            coord.append(int(float(corner.text)))
        print(coord)
        cv2.rectangle(image, (coord[0], coord[1]), (coord[2], coord[3]), (0, 0, 255), 2)
    # visualize image
    cv2.imwrite("/workspace/dataset/caltech/set07_V000_442_bbox.jpg", image)

if __name__ == "__main__":
    xml_file = "/workspace/dataset/caltech/set07_V000_442.xml"
    img_file = "/workspace/dataset/caltech/set07_V000_442.jpg"
    visualize_bbox(xml_file, img_file)
