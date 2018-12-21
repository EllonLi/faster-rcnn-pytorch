"""Transform a roidb into a trainable roidb by adding a bunch of metadata."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import datasets
import numpy as np
from model.utils.config import cfg
from datasets.factory import get_imdb
import PIL
import pdb

def prepare_roidb(imdb):
  """Enrich the imdb's roidb by adding some derived quantities that
  are useful for training. This function precomputes the maximum
  overlap, taken over ground-truth boxes, between each ROI and
  each ground-truth box. The class with maximum overlap is also
  recorded.
  """

  roidb = imdb.roidb
  #获取图片尺寸大小,还是循环遍历
  if not (imdb.name.startswith('coco')):
    sizes = [PIL.Image.open(imdb.image_path_at(i)).size
         for i in range(imdb.num_images)]
  #image_index是图片数量
  for i in range(len(imdb.image_index)):
    roidb[i]['img_id'] = imdb.image_id_at(i)#向roidb列表中元素添加字典新段img_id,图片位置
    roidb[i]['image'] = imdb.image_path_at(i)#图片的路径地址
    if not (imdb.name.startswith('coco')):
      roidb[i]['width'] = sizes[i][0]#图片的宽高
      roidb[i]['height'] = sizes[i][1]
    # need gt_overlaps as a dense array for argmax
    gt_overlaps = roidb[i]['gt_overlaps'].toarray()
    # max overlap with gt over classes (columns)
    max_overlaps = gt_overlaps.max(axis=1)
    # gt class that had the max overlap
    max_classes = gt_overlaps.argmax(axis=1)
    roidb[i]['max_classes'] = max_classes
    roidb[i]['max_overlaps'] = max_overlaps
    # sanity checks
    # max overlap of 0 => class should be zero (background)
    zero_inds = np.where(max_overlaps == 0)[0]
    assert all(max_classes[zero_inds] == 0)
    # max overlap > 0 => class should not be zero (must be a fg class)
    nonzero_inds = np.where(max_overlaps > 0)[0]
    assert all(max_classes[nonzero_inds] != 0)


def rank_roidb_ratio(roidb):
    # rank roidb based on the ratio between width and height.
    ratio_large = 2 # largest ratio to preserve.
    ratio_small = 0.5 # smallest ratio to preserve.    
    
    ratio_list = []
    for i in range(len(roidb)):
      width = roidb[i]['width']
      height = roidb[i]['height']
      ratio = width / float(height)

      if ratio > ratio_large:
        roidb[i]['need_crop'] = 1
        ratio = ratio_large
      elif ratio < ratio_small:
        roidb[i]['need_crop'] = 1
        ratio = ratio_small        
      else:
        roidb[i]['need_crop'] = 0

      ratio_list.append(ratio)

    ratio_list = np.array(ratio_list)
    ratio_index = np.argsort(ratio_list)
    return ratio_list[ratio_index], ratio_index

def filter_roidb(roidb):
    # filter the image without bounding box.
    print('before filtering, there are %d images...' % (len(roidb)))
    i = 0
    while i < len(roidb):
      if len(roidb[i]['boxes']) == 0:
        del roidb[i]
        i -= 1
      i += 1

    print('after filtering, there are %d images...' % (len(roidb)))
    return roidb

def combined_roidb(imdb_names, training=True):
  """
  Combine multiple roidbs
  """

  def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    if cfg.TRAIN.USE_FLIPPED:
      print('Appending horizontally-flipped training examples...')
      imdb.append_flipped_images()#就是将图像进行水平翻转处理一下,这是个可选操作
      print('done')

    print('Preparing training data...')

    prepare_roidb(imdb)#这个时候就获取了roidb了
    #ratio_index = rank_roidb_ratio(imdb)
    print('done')

    return imdb.roidb
  
  def get_roidb(imdb_name):
    imdb = get_imdb(imdb_name)#返回的是__sets{}一个字典,根本是字典指向了一个pascal_voc的类,所以返回的是一个pascal_voc实例,这个实例中包括了
    #图像名称列表,还有roidb(主要是每一张图的box信息,class信息等)
    #imdb.name就是数据集名称,voc_2007_test
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    imdb.set_proposal_method(cfg.TRAIN.PROPOSAL_METHOD)#在imdb类中的成员函数
    print('Set proposal method: {:s}'.format(cfg.TRAIN.PROPOSAL_METHOD))
    roidb = get_training_roidb(imdb)#获取训练集的roidb
    return roidb
  #对输入的训练集名称进行拆分,有的训练集是由两部分组成的,比如:"voc_2007_trainval+voc_2012_trainval"
  #测试集没有组合的,有也没关系作用相同voc_2007_test
  roidbs = [get_roidb(s) for s in imdb_names.split('+')]#得到数据集名称,然后调用上面的函数
  roidb = roidbs[0]#获取第一份数据集,如果只有voc_2007_test这么一个则就一个

  if len(roidbs) > 1:
    for r in roidbs[1:]:
      roidb.extend(r)
    tmp = get_imdb(imdb_names.split('+')[1])
    imdb = datasets.imdb.imdb(imdb_names, tmp.classes)
  else:
    imdb = get_imdb(imdb_names)#这个get_imdb函数很关键

  if training:
    roidb = filter_roidb(roidb)

  ratio_list, ratio_index = rank_roidb_ratio(roidb)

  return imdb, roidb, ratio_list, ratio_index
