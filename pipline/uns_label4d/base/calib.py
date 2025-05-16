#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2025-03-25 20:23:21
Description: 
LastEditors: knightdby
LastEditTime: 2025-03-25 20:35:14
FilePath: /UniOcc/uniocc/base/calib.py
Copyright 2025 by Inc, All Rights Reserved. 
2025-03-25 20:23:21
"""
import os
import numpy as np
from pyquaternion import Quaternion

kitti_to_nu_lidar = Quaternion(axis=(0, 0, 1), angle=np.pi / 2)
kitti_to_nu_lidar_inv = kitti_to_nu_lidar.inverse


class Calibration:
    def __init__(self, filepath):
        calibs = self.read_calib_file(filepath)
        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])

        self.L2C = calibs['Tr_velo_to_cam']
        self.L2C = np.reshape(self.L2C, [3, 4])

        self.L2M = calibs['Tr_velo_to_imu']
        self.L2M = np.reshape(self.L2M, [3, 4])

        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0, [3, 3])
        c2l_tf = np.linalg.inv(self.L2C[0:3, 0:3])
        c2l_trans = -np.dot(c2l_tf, self.L2C[0:3, 3])
        c2l = np.eye(4)
        c2l[:3, :3] = c2l_tf
        c2l[:3, 3] = c2l_trans
        self.cam2map = np.dot(self.L2M, c2l)

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

    def lidar2map(self, pts_3d_lidar):
        n = pts_3d_lidar.shape[0]
        pts_3d_hom = np.hstack((pts_3d_lidar, np.ones((n, 1))))
        pts_3d_cam_lidar = np.dot(pts_3d_hom, np.transpose(self.L2M))
        # pts_3d_cam_lidar = self.L2M.dot(pts_3d_hom.T).T
        return pts_3d_cam_lidar

    def lidar2nuscene(self, pts_3d_lidar):
        n = pts_3d_lidar.shape[0]
        pts_3d_hom = np.hstack((pts_3d_lidar, np.ones((n, 1))))
        rt = np.hstack(
            (kitti_to_nu_lidar.rotation_matrix, np.zeros((3, 1))))
        pts_3d_cam_lidar = rt.dot(pts_3d_hom.T).T
        return pts_3d_cam_lidar
    # From Camera Coordinate system to Image frame

    def rect2img(self, rect_pts, img_width, img_height):
        n = rect_pts.shape[0]
        points_hom = np.hstack((rect_pts, np.ones((n, 1))))
        points_2d = np.dot(points_hom, np.transpose(self.P))  # nx3
        points_2d[:, 0] /= points_2d[:, 2]
        points_2d[:, 1] /= points_2d[:, 2]

        mask = (points_2d[:, 0] >= 0) & (points_2d[:, 0] <= img_width) & (
            points_2d[:, 1] >= 0) & (points_2d[:, 1] <= img_height)
        # mask = mask & (rect_pts[:, 2] > 2) & (rect_pts[:, 2] < 80)
        return points_2d[mask, 0:2], mask
