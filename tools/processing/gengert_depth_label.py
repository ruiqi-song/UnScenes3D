#!/usr/bin/env python3
# coding=utf-8
'''
   brief: 
   Version: v1.0.0
   Author: Baiyong Ding  && baiyong.ding@waytous.com
   Date: 2024-11-20 15:36:15
   Description: 
   LastEditors: Baiyong Ding
   LastEditTime: 2024-11-20 15:36:31
   FilePath: /Uni3DScenes_dev_toolkit_/tools/processing/gengert_depth_label.py
   Copyright 2024  by Inc, All Rights Reserved. 
   2024-11-20 15:36:15
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


def get_gt_depth(Pts, n, m, grid):
    ng = 2 * grid + 1

    mX = np.zeros((m, n)) + np.float64("inf")
    mY = np.zeros((m, n)) + np.float64("inf")
    mD = np.zeros((m, n))
    mX[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
    mY[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
    mD[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[2]

    KmX = np.zeros((ng, ng, m - ng, n - ng))
    KmY = np.zeros((ng, ng, m - ng, n - ng))
    KmD = np.zeros((ng, ng, m - ng, n - ng))

    for i in range(ng):
        for j in range(ng):
            KmX[i, j] = mX[i: (m - ng + i), j: (n - ng + j)] - grid - 1 + i
            KmY[i, j] = mY[i: (m - ng + i), j: (n - ng + j)] - grid - 1 + i
            KmD[i, j] = mD[i: (m - ng + i), j: (n - ng + j)]
    S = np.zeros_like(KmD[0, 0])
    Y = np.zeros_like(KmD[0, 0])

    for i in range(ng):
        for j in range(ng):
            s = 1/np.sqrt(KmX[i, j] * KmX[i, j] + KmY[i, j] * KmY[i, j])
            Y = Y + s * KmD[i, j]
            S = S + s

    S[S == 0] = 1
    out = np.zeros((m, n))
    out[grid + 1: -grid, grid + 1: -grid] = Y/S
    return out


if __name__ == '__main__':
    DATA_DIR = './data'
    localmap_cloud_paths = get_files(DATA_DIR+'/localmap_clouds', '.bin')
    for localmap_cloud_path in tqdm(localmap_cloud_paths):
        stamp_name = os.path.basename(localmap_cloud_path)[:-4]
        calib_path = localmap_cloud_path.replace(
            'localmap_clouds', 'calibs').replace('.bin', '.txt')
        img_path = localmap_cloud_path.replace(
            'localmap_clouds', 'images').replace('.bin', '.jpg')
        depth_save_path = localmap_cloud_path.replace(
            'localmap_clouds', 'depths').replace('.bin', '.png')
        if os.path.exists(depth_save_path):
            continue
        img = cv2.imread(img_path)
        calib = Calibration(calib_path)
        lidar = np.fromfile(localmap_cloud_path, dtype=np.float32)
        lidar = lidar.reshape(-1, 3)
        lidar = lidar[lidar[:, 0] < 80, :]
        lidar_rect = calib.lidar2cam(lidar[:, 0:3])
        points_2d, lidarOnImage,  mask = calib.rect2Img(
            lidar_rect, img.shape[1], img.shape[0])
        lidarOnImagewithDepth = np.concatenate(
            (lidarOnImage, lidar_rect[mask, 2].reshape(-1, 1)), 1)
        depth_im = get_gt_depth(lidarOnImagewithDepth.T,
                                img.shape[1], img.shape[0], 4)

        depth = depth_im * 256.
        depth = depth.astype(np.uint16)
        os.makedirs(os.path.dirname(
            depth_save_path), exist_ok=True)
        imageio.imwrite(depth_save_path, depth)
