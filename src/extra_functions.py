# coding:utf-8
from __future__ import division

from shapely.wkt import loads as wkt_loads
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import shapely
import shapely.geometry
import shapely.affinity
from shapely.validation import explain_validity
from skimage import measure, data, color
import h5py
import pandas as pd
import tifffile as tiff
from numba import jit, njit
from tqdm import tqdm
from collections import defaultdict
import csv
import sys

import cv2
from shapely.geometry import MultiPolygon, Polygon
import shapely.wkt
import shapely.affinity
import numpy as np
import tifffile as tiff

import skimage.color as color
from skimage.transform import rescale

# dirty hacks from SO to allow loading of big cvs's
# without decrement loop it crashes with C error
# http://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
maxInt = sys.maxsize
decrement = True

while decrement:
    # decrease the maxInt value by factor 10
    # as long as the OverflowError occurs.

    decrement = False
    try:
        csv.field_size_limit(maxInt)
    except OverflowError:
        maxInt = int(maxInt/10)
        decrement = True

data_path = '../data'
# train_wkt = pd.read_csv(os.path.join(data_path, 'train_wkt_v4.csv'))
# gs = pd.read_csv(os.path.join(data_path, 'grid_sizes.csv'), names=['ImageId', 'Xmax', 'Ymin'], skiprows=1)
# shapes = pd.read_csv(os.path.join(data_path, '3_shapes.csv'))

epsilon = 1e-15


def get_scalers(height, width, x_max, y_min):
    """

    :param height:
    :param width:
    :param x_max:
    :param y_min:
    :return: (xscaler, yscaler)
    """
    w_ = width * (width / (width + 1))
    h_ = height * (height / (height + 1))
    return w_ / x_max, h_ / y_min


def polygons2new_mask_layer(height, width, polygons):

    img_mask = np.zeros((height, width), np.uint8)
    if not polygons:
        return img_mask
    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]
    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def polygons2mask_layer(height, width, polygons, image_id):
    """

    :param height:
    :param width:
    :param polygons:
    :return:
    """

    x_max, y_min = _get_xmax_ymin(image_id)
    x_scaler, y_scaler = get_scalers(height, width, x_max, y_min)

    polygons = shapely.affinity.scale(polygons, xfact=x_scaler, yfact=y_scaler, origin=(0, 0, 0))
    img_mask = np.zeros((height, width), np.uint8)

    if not polygons:
        return img_mask

    int_coords = lambda x: np.array(x).round().astype(np.int32)
    exteriors = [int_coords(poly.exterior.coords) for poly in polygons]
    interiors = [int_coords(pi.coords) for poly in polygons for pi in poly.interiors]

    cv2.fillPoly(img_mask, exteriors, 1)
    cv2.fillPoly(img_mask, interiors, 0)
    return img_mask


def polygons2mask(height, width, polygons, image_id):
    num_channels = len(polygons)
    result = np.zeros((num_channels, height, width))
    for mask_channel in range(num_channels):
        result[mask_channel, :, :] = polygons2mask_layer(height, width, polygons[mask_channel], image_id)
    return result


def generate_mask(image_id, height, width, num_mask_channels, train):
    """

    :param image_id:
    :param height:
    :param width:
    :param num_mask_channels: numbers of channels in the desired mask
    :param train: polygons with labels in the polygon format
    :return: mask corresponding to an image_id of the desired height and width with desired number of channels
    """

    mask = np.zeros((num_mask_channels, height, width))

    for mask_channel in range(num_mask_channels):
        poly = train.loc[(train['ImageId'] == image_id)
                         & (train['ClassType'] == mask_channel + 1), 'MultipolygonWKT'].values[0]
        polygons = shapely.wkt.loads(poly)
        mask[mask_channel, :, :] = polygons2mask_layer(height, width, polygons, image_id)
    return mask


# 生成自己抓取数据的mask
def generate_new_mask(file_name, height, width, train):
    mask = np.zeros((height, width))
    poly = train['MultipolygonWKT'][(train['file_name'] == file_name)].values[0]
    polygons = shapely.wkt.loads(poly)
    mask[:, :] = polygons2new_mask_layer(height, width, polygons)
    return mask


def png2polygons_layer(img):
    # first, find contours with cv2: it's much faster than shapely
    # 如果没有轮廓，则不需要处理
    padding = 200
    if img.min() == img.max():
        img[: , :] = 0
    else:
        # 需要检测的为0，不需要的为255
        _, img = cv2.threshold(img, 150, 255, cv2.THRESH_BINARY)
        # 100为padding部分，解决边界问题
        image = np.ones((4096+2*padding, 4096+2*padding), dtype=np.uint8) * 255
        image[padding:(4096+padding), padding:(4096+padding)] = img
        img = image
    contours = measure.find_contours(img, 0.8)  # 绘制轮廓
    if not contours:
        return MultiPolygon()
    else:
        polygonList = []
        for k in range(len(contours)):
            i = 0
            if len(contours[k]) < 6:
                continue
            pointList = []
            while i < contours[k].shape[0]:
                xy = contours[k][i]
                if xy[0] < padding / 2 or xy[0] > 4096 + padding * 3 / 2 or xy[1] < padding / 2 or xy[1] > 4096 + padding * 3 / 2:
                    contours[k] = np.delete(contours[k], i, 0)
                elif padding / 2 <= xy[0] < padding:
                        xy[0] = padding
                elif 4096 + padding < xy[0] <= 4096 + padding * 3 / 2:
                        xy[0] = 4096 + padding
                elif padding / 2 <= xy[1] < padding:
                        xy[1] = padding
                elif 4096 + padding < xy[1] <= 4096 + padding * 3 / 2:
                        xy[1] = 4096 + padding
                xy[0] = (xy[0] - padding) / 4
                xy[1] = (xy[1] - padding) / 4
                temp = xy[0]
                xy[0] = xy[1]
                xy[1] = temp
                pointList.append(xy)
                i += 1
            p = Polygon(pointList)
            if not p.is_valid:
                p = p.buffer(0)
                assert p.is_valid, "Contour %r did not make valid polygon %s because %s" % (
                p, p.wkt, explain_validity(p))
            polygonList.append(p)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(polygonList)

    all_polygons = fix_invalid_polygons(all_polygons)

    return all_polygons


def mask2polygons_layer_new(mask, min_area=200.0):
    # first, find contours with cv2: it's much faster than shapely
    # 如果没有轮廓，则不需要处理
    image_row=mask.shape[1]
    image_col=mask.shape[2]
    mask = 255-np.round(mask)
    mask = mask > 200
    mask = mask*255
    padding = 200
    # 100为padding部分，解决边界问题
    image = np.ones((image_row+2*padding, image_col+2*padding), dtype=np.uint8) * 255
    image[padding:(image_row+padding), padding:(image_col+padding)] = mask
    img = image
    contours = measure.find_contours(img, 0.8) # 绘制轮廓
    # figure, ax = plt.subplots(1, 2)
    # ax0, ax1 = ax.ravel()
    # ax0.imshow(img, interpolation='nearest', cmap=plt.cm.gray)
    # ax1.set_aspect(1)
    # for n, contour in enumerate(contours):
    #     ax1.plot(contour[:, 1], contour[:, 0], linewidth=2)
    # plt.xlim((0,1024+2*padding))
    # plt.ylim((1024+2*padding,0))
    # plt.plot()
    # plt.savefig("test.png")
    if not contours:
        return MultiPolygon()
    else:
        polygonList = []
        for k in range(len(contours)):
            i = 0
            if len(contours[k]) < 6:
                continue
            pointList = []
            while i < contours[k].shape[0]:
                xy = contours[k][i]
                if xy[0] < padding / 2 or xy[0] > image_row + padding * 3 / 2 or xy[1] < padding / 2 or xy[1] > image_col + padding * 3 / 2:
                    contours[k] = np.delete(contours[k], i, 0)
                elif padding / 2 <= xy[0] < padding:
                        xy[0] = padding
                elif image_row + padding < xy[0] <= image_row + padding * 3 / 2:
                        xy[0] = 1024 + padding
                elif padding / 2 <= xy[1] < padding:
                        xy[1] = padding
                elif image_col + padding < xy[1] <= image_col + padding * 3 / 2:
                        xy[1] = image_col + padding
                xy[0] = xy[0] - padding
                xy[1] = xy[1] - padding
                temp = xy[0]
                xy[0] = xy[1]
                xy[1] = temp
                pointList.append(xy)
                i += 1
            p = Polygon(pointList)
            if not p.is_valid:
                p = p.buffer(0)
                assert p.is_valid, "Contour %r did not make valid polygon %s because %s" % (
                p, p.wkt, explain_validity(p))
            if p.area<min_area:
                continue
            polygonList.append(p)
    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(polygonList)

    all_polygons = fix_invalid_polygons(all_polygons)

    return all_polygons


def mask2polygons_layer(mask, epsilon=1.0, min_area=10.0):
    # first, find contours with cv2: it's much faster than shapely
    contours, hierarchy = cv2.findContours(((mask == 1) * 255).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    # create approximate contours to have reasonable submission size
    if epsilon != 0:
        approx_contours = simplify_contours(contours, epsilon)
    else:
        approx_contours = contours

    if not approx_contours:
        return MultiPolygon()

    all_polygons = find_child_parent(hierarchy, approx_contours, min_area)

    # approximating polygons might have created invalid ones, fix them
    all_polygons = MultiPolygon(all_polygons)

    all_polygons = fix_invalid_polygons(all_polygons)

    return all_polygons


def find_child_parent(hierarchy, approx_contours, min_area):
    # now messy stuff to associate parent and child contours
    cnt_children = defaultdict(list)
    child_contours = set()
    assert hierarchy.shape[0] == 1
    # 遍历修改层次结构，去除最外层的轮廓(应该只有一个，因为扩展边界产生的)
    temp = hierarchy[0]
    for idx, (_, _, _, parent_idx) in enumerate(temp):
        if parent_idx == -1:
            temp = np.delete(temp, idx, 0)
            break
    # 遍历修改层次结构，去除最外层的轮廓
    for idx, (_, _, _, parent_idx) in enumerate(temp):
        if parent_idx == 0:
            temp[idx][3] = -1
    hierarchy = temp
    # http://docs.opencv.org/3.1.0/d9/d8b/tutorial_py_contours_hierarchy.html
    for idx, (_, _, _, parent_idx) in enumerate(hierarchy):
        if parent_idx != -1:
            # 将子轮廓的编号存入set集合
            child_contours.add(idx)
            # cnt_children词典中，key为上一级轮廓的编号，内容为子轮廓
            cnt_children[parent_idx].append(approx_contours[idx])

    # create actual polygons filtering by area (removes artifacts)
    all_polygons = []
    # 遍历所有的轮廓，idx为轮廓的编号，cnt为轮廓的内容
    for idx, cnt in enumerate(approx_contours):
        # 找到第一级的轮廓，也就是其他轮廓都是其子轮廓
        if idx not in child_contours and cv2.contourArea(cnt) >= min_area:
            assert cnt.shape[1] == 1
            holes = [c[:, 0, :] for c in cnt_children.get(idx, []) if cv2.contourArea(c) >= min_area]
            contour = cnt[:, 0, :]

            poly = Polygon(shell=contour, holes=holes)

            if poly.area >= min_area:
                all_polygons.append(poly)

    return all_polygons


def simplify_contours(contours, epsilon):
    return [cv2.approxPolyDP(cnt, epsilon, True) for cnt in contours]


def fix_invalid_polygons(all_polygons):
    if not all_polygons.is_valid:
        all_polygons = all_polygons.buffer(0)
        # Sometimes buffer() converts a simple Multipolygon to just a Polygon,
        # need to keep it a Multi throughout
        if all_polygons.type == 'Polygon':
            all_polygons = MultiPolygon([all_polygons])
    return all_polygons


def _get_xmax_ymin(image_id, gs):
    xmax, ymin = gs[gs['ImageId'] == image_id].iloc[0, 1:].astype(float)
    return xmax, ymin


def get_shape(image_id, band, shapes):
    if band == 3:
        height = shapes.loc[shapes['image_id'] == image_id, 'height'].values[0]
        width = shapes.loc[shapes['image_id'] == image_id, 'width'].values[0]
        return height, width


def read_image_16(image_id):
    img_m = np.transpose(tiff.imread("../data/sixteen_band/{}_M.tif".format(image_id)), (1, 2, 0)) / 2047.0
    img_3 = np.transpose(tiff.imread("../data/three_band/{}.tif".format(image_id)), (1, 2, 0)) / 2047.0
    img_p = tiff.imread("../data/sixteen_band/{}_P.tif".format(image_id)).astype(np.float32) / 2047.0

    height, width, _ = img_3.shape
    # 图像扩大，采用的插值算法是cv2.INTER_CUBIC,调整到和RGB图一样的大小
    rescaled_M = cv2.resize(img_m, (width, height), interpolation=cv2.INTER_CUBIC)
    rescaled_P = cv2.resize(img_p, (width, height), interpolation=cv2.INTER_CUBIC)

    # 数据的预处理
    rescaled_M[rescaled_M > 1] = 1
    rescaled_M[rescaled_M < 0] = 0

    rescaled_P[rescaled_P > 1] = 1
    rescaled_P[rescaled_P < 0] = 0

    image_r = img_3[:, :, 0]
    image_g = img_3[:, :, 1]
    image_b = img_3[:, :, 2]
    nir = rescaled_M[:, :, 7]
    re = rescaled_M[:, :, 5]

    L = 1.0
    C1 = 6.0
    C2 = 7.5
    evi = (nir - image_r) / (nir + C1 * image_r - C2 * image_b + L)
    evi = np.expand_dims(evi, 2)

    ndwi = (image_g - nir) / (image_g + nir)
    ndwi = np.expand_dims(ndwi, 2)

    savi = (nir - image_r) / (image_r + nir)
    savi = np.expand_dims(savi, 2)

    ccci = (nir - re) / (nir + re) * (nir - image_r) / (nir + image_r)
    ccci = np.expand_dims(ccci, 2)

    rescaled_P = np.expand_dims(rescaled_P, 2)

    result = np.transpose(np.concatenate([rescaled_M, rescaled_P, ndwi, savi, evi, ccci, img_3], axis=2), (2, 0, 1))
    return result.astype(np.float16)


def read_image_3(image_id):
    img_3 = np.transpose(tiff.imread("../data/three_band/{}.tif".format(image_id)), (1, 2, 0)) / 2047.0

    height, width, _ = img_3.shape

    result = np.transpose(np.concatenate([img_3], axis=2), (2, 0, 1))
    return result.astype(np.float16)


# 读取自己抓取的数据
def read_image_new_3(file_name):
    file_name=file_name+".tif"
    # img_3=np.transpose(cv2.imread("../data/image_tiles/{}".format(file_name)), (2, 0, 1)) / 2047.0
    img_3 = np.transpose(tiff.imread("../data/image_tiles/{}".format(file_name)), (2, 0, 1)) / 2047.0
    return img_3.astype(np.float16)


def make_prediction_cropped(model, X_train, initial_size=(572, 572), final_size=(388, 388), num_channels=19, num_masks=10):
    # padding距离：16
    shift = int((initial_size[0] - final_size[0]) / 2)

    height = X_train.shape[1]
    width = X_train.shape[2]

    # 根据final_size作为滑动窗口，对图片进行分块采样
    if height % final_size[1] == 0:
        num_h_tiles = int(height / final_size[1])
    else:
        num_h_tiles = int(height / final_size[1]) + 1
    if width % final_size[1] == 0:
        num_w_tiles = int(width / final_size[1])
    else:
        num_w_tiles = int(width / final_size[1]) + 1
    # 根据取样的滑动窗口算出来的一个图像大小,对于多出来的部分，算作加大的padding，用对称映射的方式解决
    rounded_height = num_h_tiles * final_size[0]
    rounded_width = num_w_tiles * final_size[0]

    padded_height = rounded_height + 2 * shift
    padded_width = rounded_width + 2 * shift

    padded = np.zeros((num_channels, padded_height, padded_width))
    # 边界以外部分赋值
    padded[:, shift:shift + height, shift: shift + width] = X_train

    # add mirror reflections to the padded areas, 将靠近padding 16个像素的位置翻转，然后赋值给padding，下同
    up = padded[:, shift:2 * shift, shift:-shift][:, ::-1]
    padded[:, :shift, shift:-shift] = up

    lag = padded.shape[1] - height - shift
    bottom = padded[:, height + shift - lag:shift + height, shift:-shift][:, ::-1]
    padded[:, height + shift:, shift:-shift] = bottom

    left = padded[:, :, shift:2 * shift][:, :, ::-1]
    padded[:, :, :shift] = left

    lag = padded.shape[2] - width - shift
    right = padded[:, :, width + shift - lag:shift + width][:, :, ::-1]
    padded[:, :, width + shift:] = right

    # 含义就是取原始数据的除最后一个元素之外的值
    h_start = range(0, padded_height, final_size[0])[:-1]
    assert len(h_start) == num_h_tiles

    w_start = range(0, padded_width, final_size[0])[:-1]
    assert len(w_start) == num_w_tiles

    temp = []
    for h in h_start:
        for w in w_start:
            temp += [padded[:, h:h + initial_size[0], w:w + initial_size[0]]]

    prediction = model.predict(np.array(temp))

    predicted_mask = np.zeros((num_masks, rounded_height, rounded_width))

    for j_h, h in enumerate(h_start):
         for j_w, w in enumerate(w_start):
             i = len(w_start) * j_h + j_w
             predicted_mask[:, h: h + final_size[0], w: w + final_size[0]] = prediction[i]

    return predicted_mask[:, :height, :width]


def make_prediction_cropped3(model, X_train, initial_size=(572, 572), final_size=(388, 388), num_channels=19, num_masks=10):
    shift = int((initial_size[0] - final_size[0]) / 2)

    height = X_train.shape[1]
    width = X_train.shape[2]

    if height % final_size[1] == 0:
        num_h_tiles = int(height / final_size[1])
    else:
        num_h_tiles = int(height / final_size[1]) + 1

    if width % final_size[1] == 0:
        num_w_tiles = int(width / final_size[1])
    else:
        num_w_tiles = int(width / final_size[1]) + 1

    rounded_height = num_h_tiles * final_size[0]
    rounded_width = num_w_tiles * final_size[0]

    padded_height = rounded_height + 2 * shift
    padded_width = rounded_width + 2 * shift

    padded = np.zeros((num_channels, padded_height, padded_width))

    padded[:, shift:shift + height, shift: shift + width] = X_train

    # add mirror reflections to the padded areas
    up = padded[:, shift:2 * shift, shift:-shift][:, ::-1]
    padded[:, :shift, shift:-shift] = up

    lag = padded.shape[1] - height - shift
    bottom = padded[:, height + shift - lag:shift + height, shift:-shift][:, ::-1]
    padded[:, height + shift:, shift:-shift] = bottom

    left = padded[:, :, shift:2 * shift][:, :, ::-1]
    padded[:, :, :shift] = left

    lag = padded.shape[2] - width - shift
    right = padded[:, :, width + shift - lag:shift + width][:, :, ::-1]

    padded[:, :, width + shift:] = right

    h_start = range(0, padded_height, final_size[0])[:-1]
    assert len(h_start) == num_h_tiles

    w_start = range(0, padded_width, final_size[0])[:-1]
    assert len(w_start) == num_w_tiles

    temp = []
    for h in h_start:
        for w in w_start:
            temp += [padded[:, h:h + initial_size[0], w:w + initial_size[0]]]

    prediction = model.predict(np.array(temp))

    predicted_mask = np.zeros((num_masks, rounded_height, rounded_width))

    for j_h, h in enumerate(h_start):
         for j_w, w in enumerate(w_start):
             i = len(w_start) * j_h + j_w
             predicted_mask[:, h: h + final_size[0], w: w + final_size[0]] = prediction[i]

    return predicted_mask[:, :height, :width]
