import xml.etree.ElementTree as ET 
import  os
from os import getcwd
import numpy as np
import imgaug as ia 
from imgaug import augmenters as iaa
import cv2
import random
import shutil


def renamexml(imgpath,xmlpath):
    imgfilelist=os.listdir(xmlpath)
    index=1
    for files in imgfilelist:
        if files.endswith('.xml'):
            print(files)
            ##重命名文件
            filenamex=files.split('.')
            #filenamex=files[:-4]
            imgnew_name=str(index)+'.jpg'
            xmlnew_name=str(index)+'.xml'
            img=os.path.join(imgpath,str(filenamex[0])+'.jpg')
            os.rename(img,os.path.join(imgpath,imgnew_name))
            xml=os.path.join(xmlpath,files)
            os.rename(xml,os.path.join(xmlpath,xmlnew_name))
            path=os.path.join(xmlpath,xmlnew_name)
            in_file = open(path)
            tree = ET.parse(in_file)
            root = tree.getroot()
            filename=root.find('filename')
            filename.text=imgnew_name
            tree.write(path)
            index+=1
            
            ##重命名xml内部名称


if __name__ == "__main__":
    print('expanding ...')
    xmlpath='data/VOCdevkit2007/VOC2007/Annotations'
    imgpath='data/VOCdevkit2007/VOC2007/JPEGImages'
    renamexml(imgpath,xmlpath)
    print('finish!')