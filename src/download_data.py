# coding:utf-8
"""
Script that scans 3 band tiff files and creates csv file with columns:
image_id, width, height
"""
from __future__ import division
from owslib.wms import WebMapService
import extra_functions
import os
from tqdm import tqdm
import pandas as pd
import cv2
import numpy as np
import shapely
import threading
try:  # python3
    import queue as Queue
except ImportError:  # python2
    import Queue
import time
import tifffile as tiff
data_path = '../data'

three_band_path = os.path.join(data_path, 'three_band')

testData_path = '../data/'
test_testData_path = os.path.join(testData_path, 'test')
file_names = []
widths_3 = []
heights_3 = []

ImageURL = "https://geodata.nationaalgeoregister.nl/luchtfoto/rgb/wms?request=GetCapabilities"
ContourURL = "https://geodata.nationaalgeoregister.nl/aan/wms?request=GetCapabilities"
ImageOutDirectory = '../data/image_tiles/'
ContourOutDirectory = '../data/contour_tiles/'


class DownloadThread(threading.Thread):
    def __init__(self, parameter, queue):
        threading.Thread.__init__(self)
        self.queue = queue
        self.ImageURL = parameter['ImageURL']
        self.ContourURL = parameter['ContourURL']
        self.ImageOutDirectory = parameter['ImageOutDirectory']
        self.ContourOutDirectory = parameter['ContourOutDirectory']
        self.x_num = parameter['x_num']
        self.y_num = parameter['y_num']
        self.x_stride = parameter['x_stride']
        self.y_stride = parameter['y_stride']
        self.x_start = parameter['x_start']
        self.y_start = parameter['y_start']

    def run(self):
        time.sleep(np.random.randint(0, 40))
        ImageWms = WebMapService(self.ImageURL, version='1.1.1', timeout=200)
        ContourWms = WebMapService(self.ContourURL, version='1.1.1', timeout=200)
        x_min = self.x_start
        y_min = self.y_start
        for ii in range(0, self.x_num):
            for jj in range(0, self.y_num):
                ll_x_ = x_min + ii * self.x_stride
                ll_y_ = y_min + jj * self.y_stride
                bbox = (ll_x_, ll_y_, ll_x_ + self.x_stride, ll_y_ + self.y_stride)
                try:
                    img = ImageWms.getmap(layers=['Actueel_ortho25'], srs='EPSG:4326', bbox=bbox, size=(1024, 1024),
                                          format='image/GeoTIFF', transparent=True)
                except BaseException:
                    self.queue.put(0)
                    continue
                try:
                    ContourImg = ContourWms.getmap(layers=['aan'], srs='EPSG:4326', bbox=bbox, size=(4096, 4096),
                                 format='image/png', transparent=True)
                except BaseException:
                    self.queue.put(0)
                    continue
                filename = "{}_{}_{}_{}.tif".format(bbox[0], bbox[1], bbox[2], bbox[3])
                filename2 = "{}_{}_{}_{}.png".format(bbox[0], bbox[1], bbox[2], bbox[3])
                with open(self.ImageOutDirectory + filename, 'wb') as out:
                    out.write(img.read())
                with open(self.ContourOutDirectory + filename2, 'wb') as out1:
                    out1.write(ContourImg.read())
                self.queue.put(1)
                # time.sleep(np.random.randint(10, 20))


def download_map_data(ImageURL, ContourURL, ImageOutDirectory, ContourOutDirectory):
    parameter = {}
    q = Queue.Queue()
    thead_num = 100
    x_min = 4.536343
    y_min = 52.289726
    parameter['ImageURL'] = ImageURL
    parameter['ContourURL'] = ContourURL
    parameter['ImageOutDirectory'] = ImageOutDirectory
    parameter['ContourOutDirectory'] = ContourOutDirectory
    parameter['x_stride'] = 0.005
    parameter['y_stride'] = 0.005
    no_tiles_x = 100
    no_tiles_y = 10
    total_no_tiles = no_tiles_x * no_tiles_y
    # 确保可以整除
    x_block = no_tiles_x / thead_num
    for ii in range(0, thead_num):
        parameter['x_start'] = x_min + ii*x_block*parameter['x_stride']
        parameter['y_start'] = y_min
        parameter['x_num'] = int(x_block)
        parameter['y_num'] = no_tiles_y
        t = DownloadThread(parameter, q)
        t.start()
    pbar = tqdm(total=total_no_tiles)
    number = 0
    k = 0
    while number < total_no_tiles:
        i = q.get()
        if i==1:
            pbar.update(i)
        else:
            k += 1
            print("下载出现错误已经跳过")
        number += 1
    pbar.close()
    print("预备下载：%d" % total_no_tiles)
    print("实际下载：%d" % (total_no_tiles-k))
    print("下面开始生成contour.csv")


# 针对有些数据没有现在完全，将其检查出来并重新下载。
def downloadcheck(ImageOutDirectory, ContourOutDirectory):
    num = 0
    for file_name in tqdm(sorted(os.listdir(ContourOutDirectory))):
        try:
            img_3 = tiff.imread("%s%s.tif" %(ImageOutDirectory,file_name[0:-4]))
            if img_3.size != 1024*1024*3:
                bbox = file_name[0:-4].split("_")
                singleDownloadmap(bbox, ImageURL, ContourURL, ImageOutDirectory, ContourOutDirectory)
                num+=1
        except BaseException:
            bbox = file_name[0:-4].split("_")
            for i in range(len(bbox)):
            	bbox[i]=float(bbox[i])
            singleDownloadmap(bbox, ImageURL, ContourURL, ImageOutDirectory, ContourOutDirectory)
            num+=1
    print("已经修复%d张缺失或下载错误的tif卫星图" % num)


def singleDownloadmap(bbox,ImageURL, ContourURL, ImageOutDirectory, ContourOutDirectory):
    ImageWms = WebMapService(ImageURL, version='1.1.1', timeout=200)
    ContourWms = WebMapService(ContourURL, version='1.1.1', timeout=200)
    img = ImageWms.getmap(layers=['Actueel_ortho25'], srs='EPSG:4326', bbox=bbox, size=(1024, 1024),
                                      format='image/GeoTIFF', transparent=True)
    ContourImg = ContourWms.getmap(layers=['aan'], srs='EPSG:4326', bbox=bbox, size=(4096, 4096),
                             format='image/png', transparent=True)
    filename = "{}_{}_{}_{}.tif".format(bbox[0], bbox[1], bbox[2], bbox[3])
    filename2 = "{}_{}_{}_{}.png".format(bbox[0], bbox[1], bbox[2], bbox[3])
    with open(ImageOutDirectory + filename, 'wb') as out:
        out.write(img.read())
    with open(ContourOutDirectory + filename2, 'wb') as out1:
        out1.write(ContourImg.read())
                

def create_contour_csv(contourImagePath, outputPath):
    result = []
    for file_name in tqdm(sorted(os.listdir(contourImagePath))):
        ContourImg = cv2.imread(contourImagePath + file_name)
        # print(file_name)
        ContourImg = ContourImg[:, :, 0]
        polygons = extra_functions.png2polygons_layer(ContourImg)
        result += [(file_name[0:-4], shapely.wkt.dumps(polygons))]
    contoursCSV = pd.DataFrame(result, columns=['file_name', 'MultipolygonWKT'])
    contoursCSV.to_csv(os.path.join(outputPath, 'contours.csv'), index=False)
    print("csv文件生成成功！")


if __name__ == '__main__':
    #download_map_data(ImageURL, ContourURL, ImageOutDirectory, ContourOutDirectory)
    downloadcheck(ImageOutDirectory, ContourOutDirectory)
    #create_contour_csv(ContourOutDirectory, testData_path)

