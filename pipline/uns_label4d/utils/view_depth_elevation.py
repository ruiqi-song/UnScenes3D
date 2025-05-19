#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2024-11-20 15:43:23
Description: 
LastEditors: knightdby
LastEditTime: 2025-05-19 17:17:07
FilePath: /UnScenes3D/pipline/uns_label4d/utils/view_depth_elevation.py
Copyright 2025 by Inc, All Rights Reserved. 
2024-11-20 15:43:23
"""

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
    DATA_DIR = './data/raw_data/scene_00000/camera_1'
    image_paths = get_files(DATA_DIR, '.jpg')
    for image_path in tqdm(image_paths):
        depth_path = image_path.replace(
            'camera_1', 'pc_depth').replace('.jpg', '.png')
        elevation_path = image_path.replace(
            'camera_1', 'pc_height').replace('.jpg', '.png')
        if not os.path.exists(depth_path):
            continue
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)/256.
        colormap = plt.get_cmap('viridis')
        depth_image_colored = colormap(
            depth_image / depth_image.max())
        depth_image_colored = (
            depth_image_colored[:, :, :3] * 255).astype(np.uint8)
        os.makedirs(os.path.dirname(depth_path.replace(
            'pc_depth', 'pc_depth_view')), exist_ok=True)
        cv2.imwrite(depth_path.replace(
            'pc_depth', 'pc_depth_view'),
            depth_image_colored[:, :, ::-1])

        elevation_gt = cv2.imread(elevation_path, cv2.IMREAD_UNCHANGED)/256.
        colormap = plt.get_cmap('plasma')
        elevation_gt[elevation_gt > 0] = elevation_gt.max() - \
            elevation_gt[elevation_gt > 0]
        depth_image_colored = colormap(
            elevation_gt / elevation_gt.max())
        depth_image_colored = (
            depth_image_colored[:, :, :3] * 255).astype(np.uint8)
        os.makedirs(os.path.dirname(elevation_path.replace(
            'pc_height', 'pc_height_view')), exist_ok=True)
        cv2.imwrite(elevation_path.replace(
            'pc_height', 'pc_height_view'),
            depth_image_colored[:, :, ::-1])
