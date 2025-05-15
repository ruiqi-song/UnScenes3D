#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2025-04-15 13:43:15
Description: 
LastEditors: knightdby
LastEditTime: 2025-05-03 23:34:45
FilePath: /UniOcc/uniocc/semantic/get_label_depth.py
Copyright 2025 by Inc, All Rights Reserved. 
2025-04-15 13:43:15
"""


from manifast import *
from pipline.uns_label4d.base.database import Database, cloud_viewer
import imageio


class Calibration:
    def __init__(self, calib_filepath):
        calibs = self.read_calib_file(calib_filepath)

        self.P = calibs['P2']
        self.P = np.reshape(self.P, [3, 4])

        self.L2C = calibs['Tr_velo_to_cam']
        self.L2C = np.reshape(self.L2C, [3, 4])

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
        return points_2d[mask, 0:2], mask


def dense_map(Pts, n, m, grid):
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


if __name__ == "__main__":

    db = Database('/media/knight/disk2knight/htmine_occ',
                  sweep=False)
    img = None
    for clip_name in tqdm(list(db.clip_stamps.keys())[0:]):
        stamps = db.clip_stamps[clip_name]
        # print(clip_name, stamps[0], stamps[-1])
        is_build = False
        for stamp in tqdm(stamps):
            depth_save_path = os.path.join(
                db.data_dir, f'labels/pc_depth/{stamp}.png')
            if not osp.exists(depth_save_path):
                is_build = True
            try:
                img = cv2.imread(depth_save_path)
                if img is None:
                    is_build = True
            except:
                is_build = True
        if not is_build:
            continue
        cloud_map = db.build_static_cloudmap4clip(clip_name)
        # cloud_viewer(cloud_map)
        for stamp in tqdm(stamps):
            depth_save_path = os.path.join(
                db.data_dir, f'labels/pc_depth/{stamp}.png')
            if osp.exists(depth_save_path):
                continue
            calib_path = os.path.join(
                db.data_dir, f'samples/calibs/{stamp}.txt')
            if not osp.exists(calib_path):
                calib_path = calib_path.replace('samples', 'sweeps')
            calib = Calibration(calib_path)
            if img is None:
                img_path = os.path.join(
                    db.data_dir, f'samples/images/{stamp}.jpg')
                if not osp.exists(img_path):
                    img_path = img_path.replace('samples', 'sweeps')
                img = cv2.imread(img_path)
            lidar = db.transform_pc(
                cloud_map, np.linalg.inv(db.load_calib(stamp)[0].T_lidar_odom))[:, 0:3]
            lidar_pc = db.load_lidar(stamp)[:, 0:3]
            if len(lidar_pc) > 0:
                lidar = np.concatenate(
                    (lidar, lidar_pc), axis=0)
            lidar = lidar[lidar[:, 0] < 80, :]
            # From LiDAR coordinate system to Camera Coordinate system
            lidar_rect = calib.lidar2cam(lidar[:, 0:3])
            # From Camera Coordinate system to Image frame
            lidarOnImage, mask = calib.rect2Img(
                lidar_rect, img.shape[1], img.shape[0])
            # Concatenate LiDAR position with the intesity (3), with (2) we would have the depth
            lidarOnImagewithDepth = np.concatenate(
                (lidarOnImage, lidar_rect[mask, 2].reshape(-1, 1)), 1)
            depth_im = dense_map(lidarOnImagewithDepth.T,
                                 img.shape[1], img.shape[0], 4)
            depth = depth_im * 256.
            depth = depth.astype(np.uint16)
            make_path_dirs(depth_save_path)
            imageio.imwrite(depth_save_path, depth)
            # height = height_im * 256.
            # height = height.astype(np.uint16)
            # height_save_path = os.path.join(
            #     db.data_dir, f'pc_height/{stamp}.png')
            # make_path_dirs(height_save_path)
            # imageio.imwrite(height_save_path, height)
            # break
        # break
