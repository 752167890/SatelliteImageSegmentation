# coding:utf-8
from __future__ import division

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,3"
from tqdm import tqdm
import pandas as pd
import extra_functions
import shapely
import matplotlib.pyplot as plt
from numba import jit

from keras.models import model_from_json
import numpy as np
from keras.models import model_from_json
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import cv2
config = tf.ConfigProto()
config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
sess = tf.Session(config=config)
# 设置session
KTF.set_session(sess)


def read_model(cross=''):
    json_name = 'architecture_128_50_crops_3_' + cross + '.json'
    weight_name = 'model_weights_128_50_crops_3_' + cross + '.h5'
    model = model_from_json(open(os.path.join('../src/cache', json_name)).read())
    model.load_weights(os.path.join('../src/cache', weight_name))
    return model

# 读取模型参数
model = read_model()

# sample = pd.read_csv('../data/sample_submission.csv')

data_path = '../data'
num_channels = 3
num_mask_channels = 1
threashold = 0.3


# 读取训练集的轮廓
train_wkt = pd.read_csv(os.path.join(data_path, 'contours.csv'))
ImageOutDirectory = '../data/image_tiles/'
result = []


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


@jit
def mask2poly(predicted_mask, threashold, x_scaler, y_scaler):
    polygons = extra_functions.mask2polygons_layer(predicted_mask[0] > threashold, epsilon=0, min_area=10)
    polygons = shapely.affinity.scale(polygons, xfact=1.0 / x_scaler, yfact=1.0 / y_scaler, origin=(0, 0, 0))
    return shapely.wkt.dumps(polygons)


for file_name in tqdm(sorted(os.listdir(ImageOutDirectory))):
    if file_name[0:-4] not in train_wkt['file_name']:
        # 读取图片
        image = extra_functions.read_image_new_3(file_name)
        # # 读取自定义训练图片
        # image = np.transpose(plt.imread("../data/image_tiles{}.tif".format(image_id)), (2, 0, 1)) / 2047.0
        # image=image.astype(np.float16)
        H = image.shape[1]
        W = image.shape[2]

        # 预测图片
        predicted_mask = extra_functions.make_prediction_cropped(model, image, initial_size=(112, 112),
                                                                 final_size=(112-32, 112-32),
                                                                 num_masks=num_mask_channels, num_channels=num_channels)
        # 将图片水平翻转然后预测
        image_v = flip_axis(image, 1)
        predicted_mask_v = extra_functions.make_prediction_cropped(model, image_v, initial_size=(112, 112),
                                                                   final_size=(112 - 32, 112 - 32),
                                                                   num_masks=1,
                                                                   num_channels=num_channels)
        # 将图片竖直翻转然后预测
        image_h = flip_axis(image, 2)
        predicted_mask_h = extra_functions.make_prediction_cropped(model, image_h, initial_size=(112, 112),
                                                                   final_size=(112 - 32, 112 - 32),
                                                                   num_masks=1,
                                                                   num_channels=num_channels)
        # 将图片交换维度，然后预测
        image_s = image.swapaxes(1, 2)
        predicted_mask_s = extra_functions.make_prediction_cropped(model, image_s, initial_size=(112, 112),
                                                                   final_size=(112 - 32, 112 - 32),
                                                                   num_masks=1,
                                                                num_channels=num_channels)
        # 将以上4种预测结果合并
        new_mask = np.power(predicted_mask *
                            flip_axis(predicted_mask_v, 1) *
                            flip_axis(predicted_mask_h, 2) *
                            predicted_mask_s.swapaxes(1, 2), 0.25)
        new_mask=new_mask*2047
        new_maks=new_mask.astype(np.int32)
        new_mask=np.transpose(new_mask,(1,2,0))
        cv2.imwrite("../data/image_train/%s.jpg" % file_name, new_mask)
        # # 得到kaggle所要求的grid坐标
        # x_scaler, y_scaler = extra_functions.get_scalers(H, W, x_max, y_min)

#     mask_channel = 5
#     result += [(image_id, mask_channel + 1, mask2poly(new_mask, threashold, x_scaler, y_scaler))]
#
# submission = pd.DataFrame(result, columns=['ImageId', 'ClassType', 'MultipolygonWKT'])
#
#
# sample = sample.drop('MultipolygonWKT', 1)
# submission = sample.merge(submission, on=['ImageId', 'ClassType'], how='left').fillna('MULTIPOLYGON EMPTY')
#
# submission.to_csv('temp_crops_4.csv', index=False)
