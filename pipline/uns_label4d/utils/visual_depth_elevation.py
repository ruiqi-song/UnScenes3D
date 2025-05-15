#!/usr/bin/env python3
# coding=utf-8
'''
   brief: 
   Version: v1.0.0
   Author: Baiyong Ding  && baiyong.ding@waytous.com
   Date: 2024-11-20 15:43:23
   Description: 
   LastEditors: Baiyong Ding
   LastEditTime: 2024-11-20 15:43:23
   FilePath: /Uni3DScenes_dev_toolkit_/tools/visualization/depth_elev_visual.py
   Copyright 2024  by Inc, All Rights Reserved. 
   2024-11-20 15:43:23
'''
import os
import cv2
import imageio
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_files(dir_path, ext=None):
    files = []
    for entry in os.scandir(dir_path):
        if entry.is_file() and (entry.name.endswith(ext) if ext else True):
            files.append(entry.path)
        elif entry.is_dir():
            files.extend(get_files(entry.path, ext))
    return files


if __name__ == '__main__':
    DATA_DIR = './data'
    image_paths = get_files(DATA_DIR+'/images', '.jpg')
    for image_path in tqdm(image_paths):
        image_path = './data/images/1635039340.857420.jpg'
        depth_path = image_path.replace(
            'images', 'depths').replace('.jpg', '.png')
        elevation_path = image_path.replace(
            'images', 'elevation').replace('.jpg', '.png')
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)/256.
        # 使用 matplotlib 生成热力图并将其转换为 RGB 格式
        colormap = plt.get_cmap('viridis')  # 选择一个 colormap，比如 'plasma'
        depth_image_colored = colormap(
            depth_image / depth_image.max())  # 归一化并应用 colormap
        depth_image_colored = (
            depth_image_colored[:, :, :3] * 255).astype(np.uint8)
        cv2.imwrite('./results/depth_visual.png',
                    depth_image_colored[:, :, ::-1])

        elevation_gt = cv2.imread(elevation_path, cv2.IMREAD_UNCHANGED)/256.
        colormap = plt.get_cmap('plasma')  # 选择一个 colormap，比如 'plasma'
        elevation_gt[elevation_gt > 0] = elevation_gt.max() - \
            elevation_gt[elevation_gt > 0]
        depth_image_colored = colormap(
            elevation_gt / elevation_gt.max())  # 归一化并应用 colormap
        depth_image_colored = (
            depth_image_colored[:, :, :3] * 255).astype(np.uint8)

        cv2.imwrite('./results/elevation_visual.png',
                    depth_image_colored[:, :, ::-1])
        print(image_path, depth_path, elevation_path)
        break
