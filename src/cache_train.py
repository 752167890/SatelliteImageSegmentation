# coding=utf-8
# coding:utf-8
"""
Script that caches train data for future training
"""
from __future__ import division
import os
import pandas as pd
import extra_functions
from tqdm import tqdm
import h5py
import numpy as np


data_path = '../data'
# 读取训练集的轮廓
train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
# 筛选出crops的轮廓
train_wkt = train_wkt[(train_wkt['ClassType'] == 6) & (train_wkt['MultipolygonWKT'] != 'MULTIPOLYGON EMPTY')]
# 将Type值重新赋值为1
train_wkt['ClassType'] = 1
# 读取边界坐标信息
gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
# 获取所有图片的w,h,image ID
shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))


def cache_train_16():
    print('num_train_images =', train_wkt['ImageId'].nunique())

    # 选出给出训练集答案的Shapes
    train_shapes = shapes[shapes['image_id'].isin(train_wkt['ImageId'].unique())]
    # 给出最小的shapes的图像大小
    min_train_height = train_shapes['height'].min()
    min_train_width = train_shapes['width'].min()
    # 训练集的大小，25张图
    num_train = train_shapes.shape[0]

    image_rows = min_train_height
    image_cols = min_train_width

    num_channels = 3

    num_mask_channels = 1

    # 创建一个HDF5文件, 其中create_dataset用于创建给定形状和数据类型的空dataset
    f = h5py.File(os.path.join(data_path, 'train_3.h5'), 'w', compression='blosc:lz4', compression_opts=9)

    # imgs存放读取的数据，imgs_mask存放正确答案所生成的图像
    imgs = f.create_dataset('train', (num_train, num_channels, image_rows, image_cols), dtype=np.float16)
    imgs_mask = f.create_dataset('train_mask', (num_train, num_mask_channels, image_rows, image_cols), dtype=np.uint8)

    ids = []

    i = 0
    # tqdm用于实现进度条
    for image_id in tqdm(sorted(train_wkt['ImageId'].unique())):
        # 读取图片，3维的数据
        image = extra_functions.read_image_3(image_id)
        _, height, width = image.shape
        # 截取最小的范围，其余舍弃
        imgs[i] = image[:, :min_train_height, :min_train_width]
        # 利用给定的train_wkt.csv里面的边界，生成正确答案图像。10个channel对应10个类别的正确答案
        imgs_mask[i] = extra_functions.generate_mask(image_id,
                                                     height,
                                                     width,
                                                     num_mask_channels=num_mask_channels,
                                                     train=train_wkt)[:, :min_train_height, :min_train_width]

        ids += [image_id]
        i += 1
    # 好像是为了兼容py3,好像是hdf5的问题。
    # fix from there: https://github.com/h5py/h5py/issues/441
    f['train_ids'] = np.array(ids).astype('|S9')

    f.close()


if __name__ == '__main__':
    cache_train_16()
