#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2025-05-16 10:35:00
Description: 
LastEditors: knightdby
LastEditTime: 2025-05-16 10:35:02
FilePath: /UnScenes3D/pipline/uns_label4d/genert_odom_fine.py
Copyright 2025 by Inc, All Rights Reserved. 
2025-05-16 10:35:00
"""
from manifast import *

from kiss_icp.config import load_config
from kiss_icp.datasets.generic import GenericDataset
from kiss_icp.kiss_icp import KissICP
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R

odom_cfg = load_config(None, max_range=None)
view_odom = False


def maiin():
    dataset_dir = "/media/knight/disk2knight/htmine_occ_2"
    lidar_dir = dataset_dir + '/raw_data/lidar_1'
    lidar_paths = get_files(lidar_dir, '.pcd')
    _dataset = GenericDataset(lidar_dir)
    assert len(_dataset) == len(lidar_paths)
    odometry = KissICP(config=odom_cfg)
    poses = np.zeros((len(lidar_paths), 4, 4))
    tum_data = np.zeros((len(poses), 8))
    # view_odom = True
    if view_odom:
        import open3d as o3d
        combined = o3d.geometry.PointCloud()
    for idx in tqdm(range(len(lidar_paths))):
        stamp = osp.basename(lidar_paths[idx])[:-4]
        raw_frame, timestamps_ = _dataset[idx]
        source, keypoints = odometry.register_frame(raw_frame, timestamps_)
        poses[idx] = odometry.last_pose
        tx, ty, tz = poses[idx, :3, -1].flatten()
        qw, qx, qy, qz = Quaternion(matrix=poses[idx], atol=0.01).elements
        tum_data[idx] = np.r_[float(stamp), tx, ty, tz, qx, qy, qz, qw]
        odom_pose_path = osp.join(dataset_dir, 'odom_pose', f"{stamp}.txt")
        make_path_dirs(odom_pose_path)
        np.savetxt(odom_pose_path, X=[tum_data[idx]], fmt="%.4f")
        if view_odom:
            rot = R.from_quat([qx, qy, qz, qw]).as_matrix()
            trans = np.eye(4)
            trans[:3, :3] = rot
            trans[:3, 3] = [tx, ty, tz]
            pcd = o3d.io.read_point_cloud(lidar_paths[idx])
            pcd.transform(trans)  # 应用 TUM 位姿变换
            combined += pcd

    if view_odom:
        o3d.visualization.draw_geometries([combined])


if __name__ == '__main__':
    maiin()
