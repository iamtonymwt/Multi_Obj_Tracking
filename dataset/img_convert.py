import os.path
import fnmatch
import shutil
import sys


def open_save(file, savepath, set_path, v_path):
    """
    read .seq file, and save the images into the savepath

    :param file: .seq文件路径
    :param savepath: 保存的图像路径
    :return:
    """

    # 读入一个seq文件，然后拆分成image存入savepath当中
    f = open(file, 'rb+')
    # 将seq文件的内容转化成str类型
    string = f.read().decode('latin-1')

    # splitstring是图片的前缀，可以理解成seq是以splitstring为分隔的多个jpg合成的文件
    splitstring = "\xFF\xD8\xFF\xE0\x00\x10\x4A\x46\x49\x46"

    # split .seq file into segment with the image prefix
    strlist = string.split(splitstring)
    f.close()
    count = 0
    # create the image folder path
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    # deal with file segment, every segment is an image except the first one
    for img in strlist:
        filename = set_path + '_' + v_path + '_' + str(count) + '.jpg'
        filenamewithpath = os.path.join(savepath, filename)
        # abandon the first one, which is filled with .seq header
        if count > 0:
            i = open(filenamewithpath, 'wb+')
            i.write(splitstring.encode('latin-1'))
            i.write(img.encode('latin-1'))
            i.close()
        count += 1


if __name__ == "__main__":
    rootdir = "./dataset/caltech"
    saveroot = "./dataset/caltech/JPEGImages"

    # walk in the rootdir, take down the .seq filename and filepath
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            # check .seq file with suffix
            # fnmatch 全称是 filename match，主要是用来匹配文件名是否符合规则的
            if fnmatch.fnmatch(filename, '*.seq'):
                # take down the filename with path of .seq file
                thefilename = os.path.join(parent, filename)
                # create the image folder by combining .seq file path with .seq filename
                parent_path = parent
                parent_path = parent_path.replace('\\', '/')
                thesavepath = saveroot
                set_path = parent_path.split('/')[-1]
                v_path = filename.split('.')[0]
                print("Filename=" + thefilename)
                print("Savepath=" + thesavepath)
                print(set_path)
                print(v_path)
                open_save(thefilename, thesavepath, set_path, v_path)
          



