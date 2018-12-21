#coding=utf-8
'''
author : EllonLi
使用说明:
图像预处理工具在expanddata.py文件中
包括扩充工具_func0()
重命名工具renamefile()
将所有类别的数据移动到VOC文件夹下moveFiletoVOC()
将扩充的数据移动到对应类别的文件夹atrophy-jpg-dst下_moveFile()
将原始数据移动到对应类别的文件夹atrophy-jpg-dst下_moveFile_root()
扩充时更改xml文件中的坐标值change_xml_annotation（）
获取xml中的坐标值read_xml_annotation（）

实例化对象OperateData需要传入的参数
rootpath是自定义数据地址，一般放在data/mydata下，结构是
--data
  |--mydata
        |--atrophy-jpg
        |--atrophy-xml
        |--cancer-jpg
        |--cancer-xml
        |--cache
            |--atrophy-jpg-dst
            |--atrophy-xml-dst
            |--cancer-jpg-dst
            |--cancer-xml-dst
            |--atrophy-jpg-train
            |--atrophy-xml-train
            |--cancer-jpg-train
            |--cancer-xml-train
cache文件夹及其所有的内容是py自动生成的，如果要重新建立一批数据，需要把这个删除掉
vocpath是VOC数据库的地址，一般放在data/VOCdevkit2007/VOC2007，结构是
--VOC2007
    |--JPEGImages
    |--Annotations
    |--ImageSets
JPEGImages是存图片的，Annotations是存xml的，也是自动生成的，如果重新建立一批数据，也需要把这个删除掉
'''
from expanddata import OpenrateData
import os


if __name__ == "__main__":
    print('expanding ...')
    rootpath='data/mydata/'
    vocpath='data/VOCdevkit2007/VOC2007/'
    clses=['atrophy','cancer']
    jpgpath_voc=vocpath+'JPEGImages/'
    xmlpath_voc=vocpath+'Annotations/'
    countpercls=10
    ed=OpenrateData(rootpath,vocpath,countpercls)
    
    for cla in clses:
        imgpath_src=rootpath+'{}-jpg/'.format(cla)
        xmlpath_src=rootpath+'{}-xml/'.format(cla)
        ed.expanddata(imgpath_src,xmlpath_src)
        xmlpath=rootpath+'cache/{}-xml-train'.format(cla)
        imgpath=rootpath+'cache/{}-jpg-train'.format(cla)
        ed.renamefile(False,imgpath,xmlpath)
        if not os.path.exists(jpgpath_voc):
            os.mkdir(jpgpath_voc)
        if not os.path.exists(xmlpath_voc):
            os.mkdir(xmlpath_voc)
        ed.moveFiletoVOC(imgpath,xmlpath)
    ed.renamefile(True,jpgpath_voc,xmlpath_voc)
    print('finish!')