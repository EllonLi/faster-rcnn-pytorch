#coding=utf-8
import os
import random
trainval_percent=0.8
train_percent=0.7
xmlfilepath='data/VOCdevkit2007/VOC2007/Annotations'
txtsavepath='data/VOCdevkit2007/VOC2007/ImageSets/Main'
total_xml=os.listdir(xmlfilepath)

num=len(total_xml)
list=range(num)#list=[1,2,3,4,5,...]
tv=int(num*trainval_percent)
tr=int(tv*train_percent)
trainval=random.sample(list,tv)#从list中随机采样
train=random.sample(trainval,tr)

ftrainval=open('data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt','w')
ftest=open('data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt','w')
ftrain=open('data/VOCdevkit2007/VOC2007/ImageSets/Main/train.txt','w')
fval=open('data/VOCdevkit2007/VOC2007/ImageSets/Main/val.txt','w')

for i in list:
    name=total_xml[i][:-4]+'\n'#截取名字
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()

