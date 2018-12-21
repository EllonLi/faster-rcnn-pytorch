# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 17:00:54 2018
@author: EllonLi
增强方法:
1 水平镜像
2 垂直镜像
3 裁剪
4 高斯模糊
5 亮度
6 噪声
7 消除反光

"""
import xml.etree.ElementTree as ET 
import  os
from os import getcwd
import numpy as np
import imgaug as ia 
from imgaug import augmenters as iaa
import cv2
import random
import shutil
#读取xml获取坐标值
class OpenrateData(object):
    def __init__(self,root,vocroot,cout_train_percls=10000):
        self.cout_train_percls=cout_train_percls
        self.vocroot=vocroot
        self.root=root
        print(self.cout_train_percls)
    def read_xml_annotation(self,root,image_id):
        in_file = open(os.path.join(root,image_id))
        tree = ET.parse(in_file)
        root = tree.getroot()
        objects=root.findall('object')
        bndboxs=[]
        i=1
        for i in range(len(objects)) :
            xobject=objects[i]
            bndbox=xobject.find('bndbox')
            xmin = int(bndbox.find('xmin').text)
            xmax = int(bndbox.find('xmax').text)
            ymin = int(bndbox.find('ymin').text)
            ymax = int(bndbox.find('ymax').text)
            bndboxs.append((xmin,ymin,xmax,ymax))
        return bndboxs

    def change_xml_annotation(self,root,droot, image_id, new_name,newboxlist):
        in_file = open(os.path.join(root, str(image_id)+'.xml')) #这里root分别由两个意思
        tree = ET.parse(in_file)
        xmlroot = tree.getroot() 
        sizenode=xmlroot.find('size')
        width=int(sizenode.find('width').text)
        height=int(sizenode.find('height').text)
    
        filename=xmlroot.find('filename')
        strname=new_name+'.jpg'
        filename.text=strname
        for i in range(len(newboxlist)):
            new_target=newboxlist[i]
            new_xmin = new_target[0]
            if new_xmin<=0:
                new_xmin=2
            new_ymin = new_target[1]
            if new_ymin<=0:
                new_ymin=2
            new_xmax=new_target[2]
            if new_xmax>=width:
                new_xmax=width-2
            new_ymax=new_target[3]
            if new_ymax>=height:
                new_ymax=height-2
            if new_xmin>new_xmax:
                print('ERRO:%s'%filename)
                return
            if new_ymin>new_ymax:
                print('ERRO:%s'%filename)
                return
            objects = xmlroot.findall('object')
            xobject=objects[i]
            bndbox = xobject.find('bndbox')
            xmin = bndbox.find('xmin')
            xmin.text = str(new_xmin)
            ymin = bndbox.find('ymin')
            ymin.text = str(new_ymin)
            xmax = bndbox.find('xmax')
            xmax.text = str(new_xmax)
            ymax = bndbox.find('ymax')
            ymax.text = str(new_ymax)
        tree.write(os.path.join(droot,new_name+'.xml'))   
    def _func0(self,nameprex,seq,imgpath_src, xmlpath_src,imgpath_dst,xmlpath_dst):
        filelist=os.listdir(imgpath_src)
        for files in filelist:
            if files.endswith('.jpg'):
                print("current image files:%s"%files)
                filenamex=files[:-4]
                new_imgname=nameprex+str(files)
                dstimgpath=os.path.join(imgpath_dst,new_imgname)
                imagefile=os.path.join(imgpath_src,files)
                img=cv2.imread(imagefile)
            
                img=np.array(img)
            
                bndboxlist = self.read_xml_annotation(xmlpath_src, str(filenamex)+'.xml')
            
                seq_det = seq.to_deterministic() # 保持坐标和图像同步改变，而不是随机
                image_aug = seq_det.augment_images([img])[0]
            
                cv2.imwrite(dstimgpath,image_aug)
                newboxlist=[]
                if bndboxlist!=None:
                    for m in range(len(bndboxlist)):
                        bndbox=bndboxlist[m]
                        bbs = ia.BoundingBoxesOnImage([ia.BoundingBox(x1=bndbox[0], y1=bndbox[1], x2=bndbox[2], y2=bndbox[3])], shape=img.shape)
                        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
                        new_bndbox = []
                        new_bndbox.append(int(bbs_aug.bounding_boxes[0].x1))
                        new_bndbox.append(int(bbs_aug.bounding_boxes[0].y1))
                        new_bndbox.append(int(bbs_aug.bounding_boxes[0].x2))
                        new_bndbox.append(int(bbs_aug.bounding_boxes[0].y2))
                        newboxlist.append(new_bndbox)
                        # 修改xml tree 并保存
                        newname=nameprex+str(filenamex)
                    self.change_xml_annotation(xmlpath_src, xmlpath_dst,filenamex,newname, newboxlist)

    def renamefile(self,vocmode,imgpath,xmlpath):
        prefix=imgpath.split('/')[-1].split('-')[0][0:1]#注意这里的阶段首字母只适用于现在的病变分类
        imgfilelist=os.listdir(xmlpath)
        index=1
        for files in imgfilelist:
            if files.endswith('.xml'):
                print(files)
                    ##重命名文件
                filenamex=files.split('.')
                    #filenamex=files[:-4]
                if vocmode==False:
                    imgnew_name=prefix+str(index)+'.jpg'
                    xmlnew_name=prefix+str(index)+'.xml'
                else:
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
    def _moveFile_root(self,imgdir,xmldir,imgpath,xmlpath):
        for file in os.listdir(imgdir):
            if file.endswith('.jpg'):
                shutil.copy(os.path.join(imgdir,file),os.path.join(imgpath,file))
        for file in os.listdir(xmldir):
            if file.endswith('.xml'):
                shutil.copy(os.path.join(xmldir,file),os.path.join(xmlpath,file))
    def _moveFile(self,imgdir,xmldir,imgpath,xmlpath,picknumber):
        pathdir=os.listdir(imgdir)
        filenumber=len(pathdir)
        if filenumber<picknumber:
            sample=random.sample(pathdir,filenumber)
        else:
            sample=random.sample(pathdir,picknumber)
        for name in sample:
            xmlname=name.split('.')[0]+'.xml'
            src=os.path.join(imgdir,name)
            dst=os.path.join(imgpath,name)
            shutil.move(src,dst)
            src=os.path.join(xmldir,xmlname)
            dst=os.path.join(xmlpath,xmlname)
            shutil.move(src,dst)

        return
    def moveFiletoVOC(self,imgpath_user,xmlpath_user):
        jpgpath_voc=self.vocroot+'JPEGImages/'
        xmlpath_voc=self.vocroot+'Annotations/'
        for fi in os.listdir(imgpath_user):
            if fi.endswith('.jpg'):
                shutil.move(os.path.join(imgpath_user,fi),os.path.join(jpgpath_voc,fi))
        for fi in os.listdir(xmlpath_user):
            if fi.endswith('.xml'):
                shutil.move(os.path.join(xmlpath_user,fi),os.path.join(xmlpath_voc,fi))

    def expanddata(self,imgpath_src,xmlpath_src):
        print('expanding ...')
        imgcout_temp=0
        cachepath=self.root+'cache/'
        if not os.path.exists(cachepath):
            os.mkdir(cachepath)
        cla=imgpath_src.split('/')[-2].split('-')[0]
        imgpath_dst=cachepath+'{}-jpg-dst/'.format(cla)
        xmlpath_dst=cachepath+'{}-xml-dst/'.format(cla)
        imgpath=cachepath+'{}-jpg-train/'.format(cla)
        xmlpath=cachepath+'{}-xml-train/'.format(cla)
        for fi in os.listdir(imgpath_src):
            imgcout_temp+=1
        cout_move=self.cout_train_percls-imgcout_temp
        if cout_move<0:
            cout_move=0
        if not os.path.exists(imgpath_dst):
            os.mkdir(imgpath_dst) 
        if not os.path.exists(xmlpath_dst):
            os.mkdir(xmlpath_dst)
        if not os.path.exists(imgpath):
            os.mkdir(imgpath)
        if not os.path.exists(xmlpath):
            os.mkdir(xmlpath)
        # up垂直镜像
        seq = iaa.Sequential(iaa.Flipud(1))
        self._func0('u', seq,imgpath_src, xmlpath_src,imgpath_dst,xmlpath_dst)
        #水平镜像
        seq=iaa.Sequential(iaa.Fliplr(1.0))
        self._func0('l',seq,imgpath_dst, xmlpath_dst,imgpath_dst,xmlpath_dst)
        self._func0('l',seq,imgpath_src, xmlpath_src,imgpath_dst,xmlpath_dst)
        # 高斯模糊`：
        #seq = iaa.Sequential(iaa.GaussianBlur(sigma=(1, 3.0)))
        #func0('a', seq, imgpath=path, dimgpath=path)
        #func0('a',seq,imgpath=croppath, dimgpath=path)
        # 加高斯噪声
        seq = iaa.AdditiveGaussianNoise(4)
        self._func0('n', seq,imgpath_dst, xmlpath_dst,imgpath_dst,xmlpath_dst)
        self._func0('n',seq,imgpath_src, xmlpath_src,imgpath_dst,xmlpath_dst)
        # 加高斯噪声
        seq = iaa.AdditiveGaussianNoise(1)
        self._func0('n1', seq,imgpath_dst, xmlpath_dst,imgpath_dst,xmlpath_dst)
        self._func0('n1', seq,imgpath_src, xmlpath_src,imgpath_dst,xmlpath_dst)
        # 增亮
        seq = iaa.Sequential(iaa.Multiply((0.8, 0.8)))
        self._func0('e', seq,imgpath_dst, xmlpath_dst,imgpath_dst,xmlpath_dst)
        self._func0('e', seq,imgpath_src, xmlpath_src,imgpath_dst,xmlpath_dst)
        print ('expand finish')
        print('starting move file')
        self._moveFile(imgpath_dst,xmlpath_dst, imgpath,xmlpath, cout_move)
        self._moveFile_root(imgpath_src,xmlpath_src,imgpath,xmlpath)
