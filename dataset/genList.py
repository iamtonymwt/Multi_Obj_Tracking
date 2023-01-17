import os

img_path = "./dataset/caltech/annotations_voc_select"
list_path = "./dataset/caltech/Imagesets"

def gen_list():
    cnt = 0
    Note=open(list_path + '/train.txt',mode='a')
    for file_name in os.listdir(img_path):
          file_name = file_name.split('.')[0]
          Note.write(file_name + '\n')
          print('write: ', cnt)
          cnt += 1
    Note.close()      
      


if __name__ == '__main__':
    gen_list()