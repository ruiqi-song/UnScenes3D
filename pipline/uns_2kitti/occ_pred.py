#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2025-05-16 09:33:15
Description: 
LastEditors: knightdby
LastEditTime: 2025-05-16 09:34:29
FilePath: /UnScenes3D/pipline/uns2kitti/occ_pred.py
Copyright 2025 by Inc, All Rights Reserved. 
2025-05-16 09:33:15
"""

import os
import json
from tqdm import tqdm
import os.path as osp
import shutil
import numpy as np


class Database:
    def __init__(self, data_dir, sweep=False):
        self.data_dir = data_dir
        self._load_timestamps(sweep)

    def _load_timestamps(self, sweep):
        clip_info_path = os.path.join(
            self.data_dir, 'imagesets/scene_info.json')
        with open(clip_info_path, "r", encoding='utf-8') as f:
            json_data = json.load(f)
        self.clip_info = {}
        for clip, value in json_data.items():
            frames = []
            frames.extend(value['samples'])
            if sweep:
                frames.extend(value['sweeps'])
            frames = sorted(frames)
            self.clip_info[clip] = frames


if __name__ == "__main__":

    db = Database('./data', sweep=False)
    occ_kitti_save_dir = './datasets/occ_task'
    clip_sequence_idx = 0
    clip_freq_dict = {}
    for clip in tqdm(list(db.clip_info.keys())[0:]):
        class_frequencies = np.zeros(12)
        save_seq_name = f'{clip_sequence_idx:04}'
        clip_sequence_idx += 1
        stamps = sorted(db.clip_info[clip])
        print(clip, stamps[0], stamps[-1])
        save_name_idx = 0
        for stamp in stamps:
            save_frame_name = f'{save_name_idx:06}'
            save_name_idx += 1
            image_path = osp.join(
                db.data_dir, 'images', f'{stamp}.jpg')
            lidar_path = osp.join(
                db.data_dir, 'clouds', f'{stamp}.bin')
            calib_path = osp.join(
                db.data_dir, 'calibs', f'{stamp}.txt')
            occ_path = osp.join(
                db.data_dir, 'occ', f'{stamp}.npy')

            lidar_save_path = osp.join(
                occ_kitti_save_dir, 'sequences', save_seq_name, 'velodyne', f'{save_frame_name}.bin')
            os.makedirs(os.path.dirname(lidar_save_path), exist_ok=True)
            shutil.copy(lidar_path, lidar_save_path)
            img_save_path = osp.join(
                occ_kitti_save_dir, 'sequences', save_seq_name, 'image_2', f'{save_frame_name}.jpg')
            os.makedirs(os.path.dirname(img_save_path), exist_ok=True)
            shutil.copy(image_path, img_save_path)
            calib_save_path = osp.join(
                occ_kitti_save_dir, 'sequences', save_seq_name, 'calibs', f'{save_frame_name}.txt')
            os.makedirs(os.path.dirname(calib_save_path), exist_ok=True)
            shutil.copy(calib_path, calib_save_path)

            point_cloud_range = [0, -38.4, -4, 76.8, 38.4, 5.6]
            occ_size = [256, 256, 32]
            voxel_size = [0.3, 0.3, 0.3]

            fov_voxels = np.load(occ_path)
            fov_voxels[..., 3][fov_voxels[..., 3] == 0] = 255
            voxel = np.zeros(np.array(occ_size))
            voxel[fov_voxels[:, 0].astype(np.int32), fov_voxels[:, 1].astype(
                np.int32), fov_voxels[:, 2].astype(np.int32)] = fov_voxels[:, 3]
            occ_save_path = osp.join(
                occ_kitti_save_dir, 'sequences', save_seq_name, 'voxels', f'{save_frame_name}.npy')
            os.makedirs(os.path.dirname(occ_save_path), exist_ok=True)
            np.save(occ_save_path, voxel)
