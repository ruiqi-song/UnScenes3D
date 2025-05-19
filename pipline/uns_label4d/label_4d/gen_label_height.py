#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2025-04-15 13:58:54
Description: 
LastEditors: knightdby
LastEditTime: 2025-05-19 17:09:22
FilePath: /UnScenes3D/pipline/uns_label4d/label_4d/gen_label_height.py
Copyright 2025 by Inc, All Rights Reserved. 
2025-04-15 13:58:54
"""


from manifast import *
from pipline.uns_label4d.base.database import Database
import imageio


base_height = 1.0  # in meter, the reference height of the camera w.r.t. road surface
# in meter, the range of interest above and below the base height， i.e., [-20cm, 20cm]
y_range = 0.2
# in meter, the lateral range of interest (in the horizontal coordinate of camera)
roi_x = np.array([-10.88, 10.88])
# in meter, the longitudinal range of interest
roi_z = np.array([25, 45.48])
#######################
# in [x, y(vertical), z] order. The range of interest above should be integer times of resolution here
grid_res = np.array([0.1, 0.1, 0.1])
num_grids_x = int((roi_x[1] - roi_x[0]) / grid_res[0])
num_grids_z = int((roi_z[1] - roi_z[0]) / grid_res[2])
num_grids_y = int(y_range*2 / grid_res[1])


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

        mask_inimg = (points_2d[:, 0] >= 0) & (points_2d[:, 0] <= img_width) & (
            points_2d[:, 1] >= 0) & (points_2d[:, 1] <= img_height)

        return points_2d[mask_inimg, 0:2], mask_inimg


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


if __name__ == "__main__":

    db = Database('./data/raw_data',
                  sweep=True
                  )
    db_keys = Database('./data/raw_data',
                       )
    img = None

    for clip_name in tqdm(list(db.clip_stamps.keys())[0:]):
        stamps = db_keys.clip_stamps[clip_name]
        if len(stamps) < 5:
            continue
        is_build = False
        for stamp in tqdm(stamps):
            depth_save_path = os.path.join(
                db.data_dir, clip_name, f'pc_height/{stamp}.png')
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
        cloud_map = db.build_semantic_map(clip_name)

        for stamp in tqdm(stamps):
            height_save_path = os.path.join(
                db.data_dir, clip_name, f'pc_height/{stamp}.png')
            if osp.exists(height_save_path):
                continue
            calib_path = os.path.join(
                db.data_dir, clip_name, f'calib/{stamp}.txt')
            calib = Calibration(calib_path)
            img = None
            if img is None:
                img_path = os.path.join(
                    db.data_dir, clip_name, f'camera_1/{stamp}.jpg')
                img = cv2.imread(img_path)
            lidar = db.transform_pc(
                cloud_map, np.linalg.inv(db.load_calib(clip_name, stamp).T_lidar_odom))[:, 0:3]

            lidar_pc = db.load_lidar(clip_name, stamp)[:, 0:3]

            if len(lidar_pc) > 0:
                lidar = np.concatenate(
                    (lidar, lidar_pc), axis=0)
            lidar_rect = calib.lidar2cam(lidar[:, 0:3])
            _,  mask_inimg = calib.rect2Img(
                lidar_rect, img.shape[1], img.shape[0])
            mask_roi = (lidar[:, 0] > roi_z[0]) & (lidar[:, 0] < roi_z[1]) & (
                lidar[:, 1] > roi_x[0]) & (lidar[:, 1] < roi_x[1])

            ele_gt, _ = get_gt_elevation(lidar[mask_inimg & mask_roi, :])
            height = ele_gt * 256.
            height = height.astype(np.uint16)
            make_path_dirs(height_save_path)
            imageio.imwrite(height_save_path, height)

            pc_inimg = lidar[mask_inimg, :]
            pc_inimg = pc_inimg.astype(np.float32)
            pc_proj_save_path = os.path.join(
                db.data_dir, clip_name, f'pc_inimg/{stamp}.bin')
            make_path_dirs(pc_proj_save_path)
            pc_inimg.tofile(pc_proj_save_path)
