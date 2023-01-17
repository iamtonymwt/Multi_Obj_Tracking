import os
import sys

cnt = 0

xmlfilepath = "./dataset/caltech/annotations_voc_select"
image_train_path="./dataset/caltech/JPEGImages"

def del_files(path,path2):
    global cnt
    total_xml = os.listdir(path)#所有标签名
    total_image = os.listdir(path2)#遍所有图片名
    
    for i in range(len(total_xml)):
          total_xml[i] = total_xml[i].split('.')[0]
    for i in range(len(total_image)):
          total_image[i] = total_image[i].split('.')[0]
    for a in total_image:
        if a not in total_xml:
            del_path=image_train_path + '/' + a + '.jpg'
            print('图片不存在',cnt)
            cnt += 1
            os.remove(del_path)
    print('del success!')



del_files(xmlfilepath,image_train_path)
