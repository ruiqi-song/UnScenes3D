#!/usr/bin/env python3
# coding=utf-8
'''
   brief: 
   Version: v1.0.0
   Author: Baiyong Ding  && baiyong.ding@waytous.com
   Date: 2024-11-20 14:55:02
   Description: 
   LastEditors: Baiyong Ding
   LastEditTime: 2024-11-20 15:36:10
   FilePath: /Uni3DScenes_dev_toolkit_/tools/processing/gengert_elevation_label.py
   Copyright 2024  by Inc, All Rights Reserved. 
   2024-11-20 14:55:02
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


class Calibration:
    def __init__(self, calib_filepath):
        calibs = self.read_calib_file(calib_filepath)

        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])

        self.L2C = calibs['Tr_velo_to_cam']
        self.L2C = np.reshape(self.L2C, [3, 4])

        self.L2M = calibs['Tr_velo_to_imu']
        self.L2M = np.reshape(self.L2C, [3, 4])
        # self.M2L = np.linalg.inv(self.L2M)

        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

    @staticmethod
    def read_calib_file(filepath):
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data
    # From LiDAR coordinate system to Camera Coordinate system

    def lidar2cam(self, pts_3d_lidar):
        n = pts_3d_lidar.shape[0]
        pts_3d_hom = np.hstack((pts_3d_lidar, np.ones((n, 1))))
        pts_3d_cam_ref = np.dot(pts_3d_hom, np.transpose(self.L2C))
        pts_3d_cam_rec = np.transpose(
            np.dot(self.R0, np.transpose(pts_3d_cam_ref)))
        return pts_3d_cam_rec

    def map2lidar(self, pts_3d_lidar):
        n = pts_3d_lidar.shape[0]
        pts_3d_hom = np.hstack((pts_3d_lidar, np.ones((n, 1))))
        pts_3d_cam_lidar = self.L2C.dot(pts_3d_hom.T).T
        return pts_3d_cam_lidar

    # From Camera Coordinate system to Image frame
    def rect2Img(self, rect_pts, img_width, img_height):
        n = rect_pts.shape[0]
        points_hom = np.hstack((rect_pts, np.ones((n, 1))))
        points_2d = np.dot(points_hom, np.transpose(self.P))  # nx3
        points_2d[:, 0] /= points_2d[:, 2]
        points_2d[:, 1] /= points_2d[:, 2]

        mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] <= img_width) & (
            points_2d[:, 1] >= 0) & (points_2d[:, 1] <= img_height)
        mask = mask & (rect_pts[:, 2] > 2) & (rect_pts[:, 2] < 80)
        return points_2d, points_2d[mask, 0:2], mask


def get_gt_elevation(xyz):
    N, _ = xyz.shape
    points_y = xyz[:, 2]  # points, m --> cm
    points_xz = xyz[:, [1, 0]]
    grids_y = np.zeros(
        (num_grids_z, num_grids_x), dtype=np.float32)
    grids_count = np.zeros(
        (num_grids_z, num_grids_x), dtype=np.uint8)

    for xz, y in zip(points_xz, points_y):
        if (xz[0] < roi_x[0]) or (xz[1] < roi_z[0]) or (xz[0] > roi_x[1]) or (xz[1] > roi_z[1]):
            continue
        idx_x = num_grids_x - 1 - int((xz[0] - roi_x[0]) / grid_res[0])
        idx_z = num_grids_z - 1-int((xz[1] - roi_z[0]) / grid_res[2])
        grids_y[idx_z, idx_x] += base_height - y
        grids_count[idx_z, idx_x] += 1
    mask = grids_count > 0
    grids_y[mask] = grids_y[mask] / grids_count[mask]

    return grids_y, mask


base_height = 1.0  # in meter, the reference height of the camera w.r.t. road surface
# in meter, the range of interest above and below the base height， i.e., [-20cm, 20cm]
y_range = 0.2
# in meter, the lateral range of interest (in the horizontal coordinate of camera)
roi_x = np.array([-10.88, 10.88])
# in meter, the longitudinal range of interest
roi_z = np.array([25, 45.48])
#######################
# in [x, y(vertical), z] order. The range of interest above should be integer times of resolution here
grid_res = np.array([0.04, 0.01, 0.04])
num_grids_x = int((roi_x[1] - roi_x[0]) / grid_res[0])
num_grids_z = int((roi_z[1] - roi_z[0]) / grid_res[2])
num_grids_y = int(y_range*2 / grid_res[1])

if __name__ == '__main__':
    DATA_DIR = './data'
    localmap_cloud_paths = get_files(DATA_DIR+'/localmap_clouds', '.bin')
    for localmap_cloud_path in tqdm(localmap_cloud_paths):
        localmap_cloud_path = './data/localmap_clouds/1635039340.857420.bin'

        stamp_name = os.path.basename(localmap_cloud_path)[:-4]
        calib_path = localmap_cloud_path.replace(
            'localmap_clouds', 'calibs').replace('.bin', '.txt')
        img_path = localmap_cloud_path.replace(
            'localmap_clouds', 'images').replace('.bin', '.jpg')
        height_save_path = localmap_cloud_path.replace(
            'localmap_clouds', 'elevation').replace('.bin', '.png')
        # if os.path.exists(height_save_path):
        #     continue
        img = cv2.imread(img_path)
        calib = Calibration(calib_path)
        lidar = np.fromfile(localmap_cloud_path, dtype=np.float32)
        lidar = lidar.reshape(-1, 3)
        lidar_rect = calib.lidar2cam(lidar[:, 0:3])
        points_2d, _,  mask_inimg = calib.rect2Img(
            lidar_rect, img.shape[1], img.shape[0])
        mask_roi = (lidar[:, 0] > roi_z[0]) & (lidar[:, 0] < roi_z[1]) & (
            lidar[:, 1] > roi_x[0]) & (lidar[:, 1] < roi_x[1])

        ele_gt, _ = get_gt_elevation(lidar[mask_inimg & mask_roi, :])
        height = ele_gt * 256.
        height = height.astype(np.uint16)
        os.makedirs(os.path.dirname(height_save_path), exist_ok=True)
        # imageio.imwrite(height_save_path, height)

        for i, point in enumerate(points_2d[mask_roi]):
            u, v = int(point[0]), int(point[1])
            cv2.circle(img, (u, v), 1, (0, 255, 0), -1)
        # img = img[500:-150, :, :]
        cv2.imwrite('./results/height_proj_visual.png', img)

        break
