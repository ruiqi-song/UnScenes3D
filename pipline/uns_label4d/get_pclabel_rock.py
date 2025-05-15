#!/usr/bin/env python3
# coding=utf-8
"""
brief:
Version: v0.0.1
Author: knightdby  && knightdby@163.com
Date: 2025-04-16 11:55:30
Description:
LastEditors: knightdby
LastEditTime: 2025-05-15 18:17:38
FilePath: /UnScenes3D/pipline/uns_label4d/semantic/get_pclabel_rock.py
Copyright 2025 by Inc, All Rights Reserved.
2025-04-16 11:55:30
"""

import open3d as o3d
from manifast import *
from pipline.uns_label4d.base.database import Database, cloud_viewer
from shapely.geometry import Point, Polygon
from pipline.uns_label4d.base.database import palette, class_to_name_seg, name_to_class_seg, cloud_rgb_viewer_save, cloud_viewer, cloud_viewer_rgb_id, cloud_viewer_rgb


make_path_dirs('tlogs_exp/calib_check')
view = False


def get_bbox(points, label='null'):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    obb = pcd.get_oriented_bounding_box()
    if label == 'warningsign':
        labels = np.array(pcd.cluster_dbscan(eps=0.4, min_points=4))
        if len(np.bincount(labels[labels >= 0])) == 0:
            return '', None
        largest_label = np.argmax(np.bincount(labels[labels >= 0]))
        main_points = points[labels == largest_label]
        main_pcd = o3d.geometry.PointCloud()
        main_pcd.points = o3d.utility.Vector3dVector(main_points)
        obb = main_pcd.get_oriented_bounding_box()
        h, l, w = obb.extent
        w = max(w, 0.3)
        l = max(l, 0.3)
        h = max(h, 1.5)
        obb.extent = [h, l, w]
    if label == 'rock':
        labels = np.array(pcd.cluster_dbscan(eps=0.3, min_points=4))
        if len(np.bincount(labels[labels >= 0])) == 0:
            return '', None
        largest_label = np.argmax(np.bincount(labels[labels >= 0]))
        main_points = points[labels == largest_label]
        main_pcd = o3d.geometry.PointCloud()
        main_pcd.points = o3d.utility.Vector3dVector(main_points)
        obb = main_pcd.get_oriented_bounding_box()
        h, l, w = obb.extent
        if h < 0.3:
            label = 'null'
        w = max(w, 0.3)
        l = max(l, 0.3)
        h = max(h, 0.3)
        obb.extent = [h, l, w]
    # 取中心、尺寸
    center = obb.center             # (x, y, z)
    extent = obb.extent            # (width, height, length)

    # 获取朝向：绕 Y 轴的旋转角（KITTI 的 rotation_y）
    R = obb.R                      # 3x3 旋转矩阵
    forward_vec = R[:, 2]         # Z 轴方向是前向
    rotation_y = np.arctan2(forward_vec[0], forward_vec[2])  # Yaw angle

    # 按照 KITTI 格式的顺序输出
    line = f"{label} 0 0 -1 0 0 0 0 {extent[1]:.2f} {extent[0]:.2f} {extent[2]:.2f} {center[0]:.2f} {center[1]:.2f} {center[2]:.2f} {rotation_y:.2f}"
    return line, obb


view = False


def maiin(dataset_dir):
    # view = True
    db = Database(dataset_dir, sweep=False)
    for clip_idx in tqdm(range(0, len(db.clip_stamps.keys()))):
        save_path = None
        clip_name = list(db.clip_stamps.keys())[clip_idx]
        stamps = db.clip_stamps[clip_name]
        if len(stamps) < 5:
            continue
        print(clip_name, stamps[0], stamps[-1])
        obs_lines = []
        for stamp in tqdm(stamps):
            calib, calib_pixel = db.load_calib(stamp)
            lidar_pc = db.load_lidar(stamp)
            label_2d = db.load_label2d(stamp)
            img_width = label_2d['image_info']['width']
            img_height = label_2d['image_info']['height']
            lidar_pc = lidar_pc[lidar_pc[:, 0] < 80, :]
            lidar_rect = calib_pixel.lidar2cam(lidar_pc[:, 0:3])
            points_2d, mask_inimg = calib_pixel.rect2img(
                lidar_rect, img_width, img_height)
            lidar = lidar_pc[mask_inimg, :]
            pc_rocks = []
            pc_warnings = []
            for ins in label_2d['instance']:
                if ins['obj_type'] == 'Rock':
                    x, y, w, h = ins['bbox'][0]*img_width, ins['bbox'][1] * \
                        img_height, ins['bbox'][2] * \
                        img_width, ins['bbox'][3]*img_height
                    mask_rock = (points_2d[:, 0] >= x-w/2) & (points_2d[:, 0] <= x+w/2) & (
                        points_2d[:, 1] >= y-h/2) & (points_2d[:, 1] <= y+h/2)
                    pc_rock = lidar[mask_rock, :]
                    if len(pc_rock) > 5:
                        pc_rocks.append(pc_rock)
                elif ins['obj_type'] == 'Warningsign':
                    x, y, w, h = ins['bbox'][0]*img_width, ins['bbox'][1] * \
                        img_height, ins['bbox'][2] * \
                        img_width, ins['bbox'][3]*img_height
                    mask_warn = (points_2d[:, 0] >= x-w/2) & (points_2d[:, 0] <= x+w/2) & (
                        points_2d[:, 1] >= y-h/2) & (points_2d[:, 1] <= y+h/2)
                    pc_warn = lidar[mask_warn, :]
                    if len(pc_warn) > 5:
                        pc_warnings.append(pc_warn)
            if len(pc_warnings) == 0 and len(pc_rocks) == 0:
                continue
            if view:
                vis = o3d.visualization.Visualizer()
                vis.create_window()
            for rock in pc_rocks:
                pc_map_ = db.transform_pc(rock[:, :3], calib.T_lidar_odom)
                bbox, obb = get_bbox(pc_map_, 'rock')
                if 'rock' in bbox:
                    obs_lines.append(bbox)
                if view:
                    print(bbox)
                    vis.add_geometry(obb)
            for warn in pc_warnings:
                pc_map_ = db.transform_pc(warn[:, :3], calib.T_lidar_odom)
                bbox, obb = get_bbox(pc_map_, 'warningsign')
                obs_lines.append(bbox)
                if view:
                    print(bbox)
                    vis.add_geometry(obb)
            if view:
                point_cloud = o3d.geometry.PointCloud()
                lidar = db.transform_pc(lidar[:, :3], calib.T_lidar_odom)
                point_cloud.points = o3d.utility.Vector3dVector(lidar)
                vis.add_geometry(point_cloud)
                vis.get_render_option().background_color = np.asarray(
                    [0, 0, 0])  # you can set the bg color
                vis.run()
                vis.destroy_window()
        label_path = os.path.join(
            dataset_dir, f'labels/label_4d_invis/{clip_name}.txt')
        make_path_dirs(label_path)
        with open(label_path, "w") as f:
            for line in obs_lines:
                if line:
                    f.write(line + "\n")
        # break


if __name__ == "__main__":
    dataset_dir = '/media/knight/disk2knight/htmine_occ'
    maiin(dataset_dir)
