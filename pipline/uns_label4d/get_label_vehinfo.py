#!/usr/bin/env python3
# coding=utf-8
"""
brief: 
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2025-05-15 13:48:06
Description: 
LastEditors: knightdby
LastEditTime: 2025-05-15 18:19:34
FilePath: /UnScenes3D/pipline/uns_label4d/publish/publish_vehinfo.py
Copyright 2025 by Inc, All Rights Reserved. 
2025-05-15 13:48:06
"""

from manifast import *
from pipline.uns_label4d.base.database import Database
from scipy.spatial.transform import Rotation as R


def get_current_speed(stamp):
    data_dir = '/media/knight/disk2knight/htmine_occ'
    ego_pose_path = osp.join(
        data_dir, 'samples/ego_pose', f'{stamp}.txt')
    if not osp.exists(ego_pose_path):
        ego_pose_path = ego_pose_path.replace('samples', 'sweeps')
    lines = []
    with open(ego_pose_path, 'r') as file_to_read:
        while True:
            line = file_to_read.readline()  # 整行读取数据
            lines.append(line)
            if not line:
                break
    # print(lines)
    k, rt = lines[0].split(' ', 1)  # 四元数格式 (x, y, z, w)
    quaternion = np.array([float(x) for x in rt.split()])[-4:]
    # print(quaternion)
    # 将四元数转换为旋转矩阵
    rotation_matrix = R.from_quat(quaternion).as_matrix()
    # 世界坐标系下的线速度 (示例数据，单位 m/s)
    key, value = lines[1].split(' ', 1)
    velocity_world = np.array([float(x) for x in value.split()])[:3]
    # 将世界坐标系下的线速度转换为车体坐标系下的线速度
    velocity_body = rotation_matrix.T @ velocity_world
    # print(velocity_world, velocity_body)
    return key, velocity_body


if __name__ == "__main__":

    db = Database('/media/knight/disk2knight/htmine_occ',
                  sweep=True)
    #   sweep=False)
    dataset_save_dir = '/media/knight/disk2knight/htmine_occ/HTMINE_Occ2pub'
    clip_sequence_idx = 0
    clip_freq_dict = {}
    for clip_idx in tqdm(range(0, len(db.clip_stamps.keys()), 1)):
        clip_name = list(db.clip_stamps.keys())[clip_idx]
        stamps = db.clip_stamps[clip_name]
        if len(stamps) < 10:
            continue
        class_frequencies = np.zeros(12)
        save_seq_name = f'{clip_sequence_idx:04}'
        clip_sequence_idx += 1
        print(clip_name, stamps[0], stamps[-1])
        save_name_idx = 0
        for idx in range(len(stamps)):
            if idx == len(stamps) - 1:
                st_1, velo_1 = get_current_speed(stamps[idx-1])
                st_2, velo_2 = get_current_speed(stamps[idx])
                acc = (velo_2 - velo_1) / (float(st_2) - float(st_1))
                velo = velo_2
            elif idx == 0:
                st_1, velo_1 = get_current_speed(stamps[idx])
                st_2, velo_2 = get_current_speed(stamps[idx+1])
                acc = (velo_2 - velo_1) / (float(st_2) - float(st_1))
                velo = velo_1
            else:
                st_1, velo_1 = get_current_speed(stamps[idx-1])
                st_2, velo_2 = get_current_speed(stamps[idx])
                st_3, velo_3 = get_current_speed(stamps[idx+1])
                acc_2 = (velo_3 - velo_2) / (float(st_3) - float(st_2))
                acc_1 = (velo_2 - velo_1) / (float(st_2) - float(st_1))
                acc = (acc_2 + acc_1) / 2.0
                velo = velo_2

            # 保留两位小数
            ego_pose_path = osp.join(
                db.data_dir, 'samples/ego_pose', f'{stamps[idx]}.txt')
            if not osp.exists(ego_pose_path):
                ego_pose_path = ego_pose_path.replace('samples', 'sweeps')
            lines = []
            with open(ego_pose_path, 'r') as file_to_read:
                while True:
                    line = file_to_read.readline()  # 整行读取数据
                    lines.append(line)
                    if not line:
                        break
            lines = lines[:2]
            save_path = osp.join(
                dataset_save_dir, 'vehicle_infos',  f'{stamps[idx]}.txt')
            rstent = lines[1].split()[0]
            lines[1] = rstent + f' {velo[0]:.4f} {velo[1]:.4f} {velo[2]:.4f} ' + \
                f'{acc[0]:.4f} {acc[1]:.4f} {acc[2]:.4f}'
            make_path_dirs(save_path)
            with open(save_path, 'w') as file_to_write:
                for line in lines:
                    file_to_write.write(line)
        #     break
        # break
