# coding:utf-8
"""
Script that scans 3 band tiff files and creates csv file with columns:
image_id, width, height
"""
from __future__ import division
from owslib.wms import WebMapService
import extra_functions
import tifffile as tiff
import os
from tqdm import tqdm
import pandas as pd
import cv2
import numpy as np
import shapely

data_path = '../data'

three_band_path = os.path.join(data_path, 'three_band')

testData_path = '../testData'
test_testData_path = os.path.join(testData_path, 'test')
file_names = []
widths_3 = []
heights_3 = []


def downloadMapData():
    ImageURL = "https://geodata.nationaalgeoregister.nl/luchtfoto/rgb/wms?request=GetCapabilities"
    ImageWms = WebMapService(ImageURL, version='1.1.1')
    ContourURL = "https://geodata.nationaalgeoregister.nl/aan/wms?request=GetCapabilities"
    ContounrWms = WebMapService(ContourURL,version='1.1.1')
    OUTPUT_DIRECTORY = '../testData/image_tiles/'
    result = []
    x_min = 4.536343
    y_min = 52.289726
    dx, dy = 0.005, 0.005
    no_tiles_x = 1
    no_tiles_y = 1
    total_no_tiles = no_tiles_x * no_tiles_y

    x_max = x_min + no_tiles_x * dx
    y_max = y_min + no_tiles_y * dy
    BOUNDING_BOX = [x_min, y_min, x_max, y_max]

    for ii in tqdm(range(0, no_tiles_x)):
        for jj in tqdm(range(0, no_tiles_y)):
            ll_x_ = x_min + ii * dx
            ll_y_ = y_min + jj * dy
            bbox = (ll_x_, ll_y_, ll_x_ + dx, ll_y_ + dy)
            img = ImageWms.getmap(layers=['Actueel_ortho25'], srs='EPSG:4326', bbox=bbox, size=(1024, 1024),
                             format='image/GeoTIFF', transparent=True)
            ContourImg = ContounrWms.getmap(layers=['aan'], srs='EPSG:4326', bbox=bbox, size=(1024, 1024),
                             format='image/png', transparent=True)
            filename = "{}_{}_{}_{}.tif".format(bbox[0], bbox[1], bbox[2], bbox[3])
            filename2 = "{}_{}_{}_{}.png".format(bbox[0], bbox[1], bbox[2], bbox[3])
            out = open(OUTPUT_DIRECTORY + filename, 'wb')
            out.write(img.read())
            out.close()
            out1 = open(OUTPUT_DIRECTORY + filename2, 'wb')
            out1.write(ContourImg.read())
            out1.close()
            ContourImg = cv2.imread(OUTPUT_DIRECTORY + filename2)
            ContourImg = ContourImg[:, :, 0]
            polygons=extra_functions.png2polygons_layer(ContourImg)
            result += [(filename, shapely.wkt.dumps(polygons))]
    contoursCSV = pd.DataFrame(result, columns=['file_name', 'MultipolygonWKT'])
    contoursCSV.to_csv('../testData/contours.csv', index=False)


if __name__ == '__main__':
    downloadMapData()
    # for file_name in tqdm(sorted(os.listdir(test_testData_path))):
    #     # TODO: crashes if there anything except tiff files in folder (for ex, QGIS creates a lot of aux files)
    #     # 读取灰度图
    #     img = cv2.imread(file_name)
    #     img = img[:, :, 0]
    #     # 利用Canny算法进行检测边缘
    #     img = cv2.Canny(img, 150, 180)
    #     # 获取轮廓
    #     _, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     # df = pd.DataFrame({'file_name': file_names, 'width': widths_3, 'height': heights_3})
    #     #
    #     # df['image_id'] = df['file_name'].apply(lambda x: x.split('.')[0])
    #     #
    #     # df.to_csv(os.path.join(data_path, '3_shapes.csv'), index=False)
